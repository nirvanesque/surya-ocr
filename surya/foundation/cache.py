from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import StaticCache
from transformers import PretrainedConfig

"""
Special cache class for the surya foundation model that supports - 
1) Static shape
2) A custom sliding window, where image tokens stay in cache, and text tokens are popped
3) Continuous batching - merging etc
4) Attention mask management - To match with what's currently in the cache

Heavily inspired from https://github.com/huggingface/transformers/blob/0725cd6953803b8aacfc85288cbfb83dea30c469/src/transformers/cache_utils.py#L1079
"""


class ContinuousBatchingLayerCache(StaticCache):
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int,
        max_cache_len: int,
        text_sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        # No need for the super class call, it just overwrites the caches
        # At some point, we should consider not inheriting from StaticCache
        self.max_cache_len = max_cache_len
        self.max_batch_size = batch_size

        self.head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        cache_shape = (
            self.max_batch_size,
            self.num_key_value_heads,
            self.max_cache_len,
            self.head_dim,
        )
        self.key_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

        self.text_sliding_window = text_sliding_window
        self.text_token_counts = torch.zeros(
            self.max_batch_size, dtype=torch.long, device=device
        )
        self.cache_image_end = self.max_cache_len - self.text_sliding_window

        self.dtype = dtype
        self.device = device

    # This is used by HF models to determine the causal relationship between new tokens and cache
    # Our cache is left padded - So all tokens should always be visible to new tokens
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.max_cache_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prefill = cache_kwargs.get("prefill", False)
        if prefill:
            return self._prefill_update(
                self.key_cache,
                self.value_cache,
                key_states,
                value_states,
                cache_kwargs,
            )
        else:
            return self._decode_update(key_states, value_states, cache_kwargs)

    def _prefill_update(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        cache_idxs: List[int] = cache_kwargs.get("cache_idxs", None)
        cache_idx_length: int = cache_kwargs.get("cache_idx_length", None)
        text_lengths: List[int] = cache_kwargs.get("text_lengths", None)
        assert cache_idxs is not None, "cache_idxs must be specified during prefill"
        assert text_lengths is not None, "text_lengths must be specified during prefill"

        # Get cache dimensions
        seq_len = key_states.shape[2]

        for batch_idx, cache_idx in enumerate(cache_idxs):
            # If the cache_idx is out of bounds, skip it
            # Cannot continue here due to compile
            if batch_idx < cache_idx_length:
                text_len = text_lengths[batch_idx]
                state_image_end = seq_len - text_len
                cache_image_start = self.cache_image_end - state_image_end

                # Assign image tokens (always static) to the cache
                key_cache[cache_idx, :, cache_image_start : self.cache_image_end] = (
                    key_states[batch_idx, :, :state_image_end, :]
                )
                value_cache[cache_idx, :, cache_image_start : self.cache_image_end] = (
                    value_states[batch_idx, :, :state_image_end, :]
                )

                # Assign text tokens (sliding window) to the cache
                cache_text_len = min(text_len, self.text_sliding_window)
                cache_text_end = self.cache_image_end + cache_text_len
                key_cache[cache_idx, :, self.cache_image_end : cache_text_end] = (
                    key_states[batch_idx, :, -cache_text_len:, :]
                )
                value_cache[cache_idx, :, self.cache_image_end : cache_text_end] = (
                    value_states[batch_idx, :, -cache_text_len:, :]
                )
                self.text_token_counts[cache_idx] = (
                    cache_text_len  # Track the position of our last text token
                )

        return key_states, value_states

    def _decode_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Static cache update
        - respects per-batch text token limits
        - per-batch valid token lengths (right-padded inputs)

        kv states are expected to have shape [batch_size, kv_heads, T_pad, head_dim]
        They may have different `true` lengths, to account for multi token preds, or beacon tokens
        Expects `num_valid_tokens` in cache_kwargs: a tensor of shape (B,) indicating the number
        of actual (non-padded) tokens to add per batch element.
        """

        num_valid_tokens: torch.Tensor = cache_kwargs.get(
            "num_valid_tokens"
        )  # shape: (B,)
        assert num_valid_tokens is not None, (
            "`num_valid_tokens` must be provided in `cache_kwargs`"
        )

        k_cache = self.key_cache  # (B, H, L, D)
        v_cache = self.value_cache  # (B, H, L, D)

        existing_space = (
            self.text_sliding_window - self.text_token_counts
        )  # free room per batch
        end_insert_pos = (
            self.text_sliding_window + self.cache_image_end - num_valid_tokens
        )

        shifts = (num_valid_tokens - existing_space).clamp(
            min=torch.zeros_like(num_valid_tokens), max=num_valid_tokens
        )
        # ─► 0  when enough space
        # ─► new_token_count-existing_space when partial space
        # ─► new_token_count when no space

        insert_positions = torch.where(
            existing_space >= num_valid_tokens,  # “no-shift” case
            self.text_token_counts + self.cache_image_end,  # append after current text
            end_insert_pos,
        )

        shifts = shifts.view(-1, 1, 1).expand(
            -1, k_cache.shape[1], self.text_sliding_window
        )

        # Shift the text tokens if needed
        indices = (
            torch.arange(self.text_sliding_window, device=k_cache.device)
            .view(1, 1, -1)
            .expand(k_cache.shape[0], k_cache.shape[1], -1)
        )
        shifted_indices = (
            (indices + shifts).clamp(0, self.text_sliding_window - 1).unsqueeze(-1)
        )
        shifted_indices = shifted_indices.expand(-1, -1, -1, k_cache.shape[-1])
        v_cache[:, :, self.cache_image_end :, :] = torch.gather(
            v_cache[:, :, self.cache_image_end :, :], dim=2, index=shifted_indices
        )
        k_cache[:, :, self.cache_image_end :, :] = torch.gather(
            k_cache[:, :, self.cache_image_end :, :], dim=2, index=shifted_indices
        )

        for batch_idx, new_text_len in enumerate(num_valid_tokens):
            cache_idx = batch_idx
            new_text_len = num_valid_tokens[batch_idx]

            insert_position = insert_positions[batch_idx]
            end_position = insert_position + new_text_len

            k_new = key_states[batch_idx, :, -new_text_len:, :]
            v_new = value_states[batch_idx, :, -new_text_len:, :]

            if new_text_len > 0:
                candidate_tokens = self.text_token_counts[cache_idx] + new_text_len
                total_text_tokens = (
                    candidate_tokens
                    if candidate_tokens <= self.text_sliding_window
                    else self.text_sliding_window
                )
                k_cache[cache_idx, :, insert_position:end_position] = k_new
                v_cache[cache_idx, :, insert_position:end_position] = v_new
                self.text_token_counts[cache_idx] = total_text_tokens

        self.key_cache = k_cache
        self.value_cache = v_cache

        return self.key_cache, self.value_cache


class ContinuousBatchingCache:
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int,
        max_cache_len: int,
        text_sliding_window: int,
        device: int,
        dtype: int,
    ):
        self.text_sliding_window = text_sliding_window
        self.num_layers = config.num_hidden_layers
        self.max_cache_len = max_cache_len
        self.max_batch_size = batch_size
        self.cache_image_end = self.max_cache_len - self.text_sliding_window

        self.attention_mask = torch.zeros(
            (self.max_batch_size, self.max_cache_len), device=device, dtype=torch.long
        )
        self.layer_caches = [
            ContinuousBatchingLayerCache(
                config,
                batch_size=batch_size,
                max_cache_len=max_cache_len,
                text_sliding_window=text_sliding_window,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.num_layers)
        ]

        self.dtype = dtype
        self.device = device

    def decode_attention_mask_update(
        self, num_valid_tokens: torch.Tensor, cache_idxs: List[int]
    ):
        num_valid_tokens_list = num_valid_tokens.tolist()
        for batch_idx, cache_idx in enumerate(cache_idxs):
            existing_token_count = self.layer_caches[0].text_token_counts[cache_idx]
            new_text_len = num_valid_tokens_list[batch_idx]

            cache_text_len = min(
                self.text_sliding_window, new_text_len + existing_token_count
            )
            cache_text_end = self.cache_image_end + cache_text_len
            self.attention_mask[cache_idx, self.cache_image_end : cache_text_end] = 1

    # Mirrors the logic from _prefill_update
    def prefill_attention_mask_update(
        self,
        attention_mask: torch.Tensor,
        cache_idxs: List[int],
        text_lengths: List[int],
        image_lengths: List[int],
    ):
        for batch_idx, cache_idx in enumerate(cache_idxs):
            text_len = text_lengths[batch_idx]
            image_len = image_lengths[batch_idx]
            valid_mask_length = text_len + image_len
            self.attention_mask[cache_idx] = 0  # Set default

            # Assign image mask
            cache_image_start = self.cache_image_end - image_len
            self.attention_mask[cache_idx, cache_image_start : self.cache_image_end] = (
                attention_mask[batch_idx, -valid_mask_length:-text_len]
            )

            # Assign text mask
            cache_text_length = min(text_len, self.text_sliding_window)
            cache_text_end = self.cache_image_end + cache_text_length
            self.attention_mask[cache_idx, self.cache_image_end : cache_text_end] = (
                attention_mask[batch_idx, -cache_text_length:]
            )

    # This is used by HF models to determine the causal relationship between new tokens and cache
    # Our cache is left padded - So all tokens should always be visible to new tokens
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.max_cache_len
