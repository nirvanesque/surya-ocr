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


class ContinuousBatchingCache(StaticCache):
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int,
        max_cache_len: int,
        text_sliding_window: int,
        device: int,
        dtype: int,
    ):
        # batch_size is deprecated in newer versions
        super().__init__(
            config,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
            max_batch_size=batch_size,
        )
        self.text_sliding_window = text_sliding_window
        self.num_layers = config.num_hidden_layers

        self.attention_mask = torch.zeros(
            (self.max_batch_size, self.max_cache_len), device=device, dtype=torch.long
        )
        self.text_token_counts = [
            torch.zeros(self.max_batch_size, dtype=torch.long, device=device)
            for _ in range(self.num_layers)
        ]

        self.dtype = dtype
        self.device = device

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prefill = cache_kwargs.get("prefill", False)
        update_fn = self._prefill_update if prefill else self._decode_update
        return update_fn(
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            key_states,
            value_states,
            self.text_token_counts[layer_idx],
            cache_kwargs,
        )

    def update_text_counts(self, cache_idxs: List[int], new_text_lens: List[int]):
        assert len(cache_idxs) == len(new_text_lens)
        new_text_len_tensor = torch.tensor(
            new_text_lens, dtype=torch.long, device=self.device
        )
        for layer_idx in range(self.num_layers):
            self.text_token_counts[layer_idx][cache_idxs] = new_text_len_tensor


    # Mirrors the logic from _prefill_update
    # Logic is better explained in this funcrtion
    def prefill_attention_mask_update(
        self,
        prefill_attention_mask: torch.Tensor,
        cache_idxs: List[int],
        text_lengths: List[int],
    ):
        seq_len = prefill_attention_mask.shape[1]
        sliding_window = self.text_sliding_window
        total_cache_len = self.max_cache_len
        prefix_cache_space = total_cache_len - sliding_window

        for batch_idx, cache_idx in enumerate(cache_idxs):
            text_len = text_lengths[batch_idx]
            prefix_len = seq_len - text_len
            self.attention_mask[cache_idx] = 0  # Set default

            assert prefix_len > 0, "There are no prefix (image) tokens!"
            
            end_pos = prefix_cache_space
            # Handle prefix part - Which may be left padded
            if prefix_len <= prefix_cache_space:
                start_pos = prefix_cache_space - prefix_len
                self.attention_mask[cache_idx, start_pos: end_pos] = prefill_attention_mask[batch_idx, :prefix_len]
            else:
                self.attention_mask[cache_idx, :end_pos] = prefill_attention_mask[batch_idx, prefix_len - prefix_cache_space: prefix_len]

            # Handle text part, keeping sliding window in consideration
            # All of the left padding is before the prefix, so we can ignore the prefill_attention_mask here
            if text_len > 0:
                text_cache_start = prefix_cache_space
                if text_len <= sliding_window:
                    self.attention_mask[cache_idx, text_cache_start: text_cache_start + text_len] = 1
                else:
                    self.attention_mask[cache_idx, -sliding_window: ] = 1
                
    # Slow impl for now - Prefill time is dominated by the large sequence length forward pass
    def _prefill_update(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        text_token_counts: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        cache_idxs: List[int] = cache_kwargs.get("cache_idxs", None)
        text_lengths: List[int] = cache_kwargs.get("text_lengths", None)
        assert cache_idxs is not None, "cache_idxs must be specified during prefill"
        assert text_lengths is not None, "text_lengths must be specified during prefill"

        _, _, seq_len, _ = key_states.shape
        total_cache_len = self.max_cache_len
        sliding_window = self.text_sliding_window
        prefix_cache_space = total_cache_len - sliding_window
        
        for batch_idx, cache_idx in enumerate(cache_idxs):
            text_len = text_lengths[batch_idx]
            prefix_len = seq_len - text_len
            
            ###### Handle Image Tokens (Prefix) #####
            # Place image tokens in appropriate cache space, aligned to the **right edge**
            assert prefix_len > 0, "There are no prefix (image) tokens!"
            
            # prefix_len may be greater than the prefix cache space due to left padding - This happens when
            # a different batch element has a large input text during prefill, causing others to have a lot of 
            # left padding. We can safely take the last `prefix_cache_space` elements from the kv states, since
            # `prefix_cache_space` is large enough to fit any image, and the rest **has to be** padding
            end_pos = prefix_cache_space
            if prefix_len <= prefix_cache_space:
                start_pos = prefix_cache_space - prefix_len
                key_cache[cache_idx, :, start_pos:end_pos] = key_states[batch_idx, :, :prefix_len]
                value_cache[cache_idx, :, start_pos:end_pos] = value_states[batch_idx, :, :prefix_len]
            else:
                key_cache[cache_idx, :, :end_pos] = key_states[batch_idx, :, prefix_len - prefix_cache_space: prefix_len]
                value_cache[cache_idx, :, :end_pos] = value_states[batch_idx, :, prefix_len - prefix_cache_space: prefix_len]


            ###### Handle Text Tokens #####
            # Text tokens start at the **left edge** of sliding window cache space
            if text_len > 0:
                text_cache_start = prefix_cache_space
                
                if text_len <= sliding_window:
                    key_cache[cache_idx, :, text_cache_start:text_cache_start + text_len] = key_states[batch_idx, :, prefix_len:prefix_len + text_len]
                    value_cache[cache_idx, :, text_cache_start:text_cache_start + text_len] = value_states[batch_idx, :, prefix_len:prefix_len + text_len]
                else:
                    start_in_text = text_len - sliding_window
                    key_cache[cache_idx, :, text_cache_start:text_cache_start + sliding_window] = key_states[batch_idx, :, prefix_len + start_in_text:prefix_len + text_len]
                    value_cache[cache_idx, :, text_cache_start:text_cache_start + sliding_window] = value_states[batch_idx, :, prefix_len + start_in_text:prefix_len + text_len]

        # Return the full key/value states (not just cached) for use in subsequent layers
        return key_states, value_states

    # """
    # Matches the logic of the decode update, but needs to be called before the updates
    # since some parts of the model depend on the attention mask
    # """
    # def decode_attention_mask_update(
    #     self, num_valid_tokens: List[int], cache_idxs: List[int]
    # ):
    #     sliding_window = self.text_sliding_window
    #     text_cache_start = self.max_cache_len - sliding_window

    #     # Using text_token_counts of first layer, should be same for all though
    #     current_text_lens = self.text_token_counts[0]
    #     new_text_lens = current_text_lens + num_valid_tokens
    #     is_full = new_text_lens > sliding_window

    #     # For caches that become full, attention mask is all 1s
    #     # For caches that remain empty, we only need to add 1s in the appropriate places
    #     for batch_idx, cache_idx in enumerate(cache_idxs):
    #         valid_tokens = num_valid_tokens[batch_idx].item()
    #         text_len = current_text_lens[batch_idx]
    #         if is_full[batch_idx]:
    #             self.attention_mask[cache_idx, text_cache_start:] = 1
    #         else:
    #             self.attention_mask[cache_idx, text_len: text_len + valid_tokens] = 1

    def decode_attention_mask_update(
        self, num_valid_tokens: List[int], cache_idxs: List[int]
    ):
        sliding_window = self.text_sliding_window
        text_cache_start = self.max_cache_len - sliding_window

        # Using text_token_counts of first layer, should be same for all though
        current_text_lens = self.text_token_counts[0]
        
        # Convert to tensors for vectorized operations
        num_valid_tokens_tensor = torch.tensor(num_valid_tokens, device=current_text_lens.device)
        cache_idxs_tensor = torch.tensor(cache_idxs, device=current_text_lens.device)
        
        # Get current text lengths for the relevant cache indices
        current_lens = current_text_lens[cache_idxs_tensor]
        new_text_lens = current_lens + num_valid_tokens_tensor
        is_full = new_text_lens > sliding_window

        # Handle full caches - set entire sliding window to 1
        if is_full.any():
            full_mask = is_full
            full_cache_idxs = cache_idxs_tensor[full_mask]
            self.attention_mask[full_cache_idxs, text_cache_start:] = 1

        # Handle non-full caches - set specific ranges to 1
        if (~is_full).any():
            non_full_mask = ~is_full
            non_full_cache_idxs = cache_idxs_tensor[non_full_mask]
            non_full_current_lens = current_lens[non_full_mask]
            non_full_valid_tokens = num_valid_tokens_tensor[non_full_mask]
            
            # Create a range tensor for advanced indexing
            max_valid_tokens = non_full_valid_tokens.max().item() if len(non_full_valid_tokens) > 0 else 0
            if max_valid_tokens > 0:
                # Create offset matrix for each batch item
                batch_size = len(non_full_cache_idxs)
                offset_range = torch.arange(max_valid_tokens, device=current_text_lens.device)
                
                # Broadcast to create indices for each batch item
                batch_offsets = offset_range.unsqueeze(0).expand(batch_size, -1)  # [batch_size, max_valid_tokens]
                start_positions = non_full_current_lens.unsqueeze(1)  # [batch_size, 1]
                valid_token_counts = non_full_valid_tokens.unsqueeze(1)  # [batch_size, 1]
                
                # Create mask for valid positions
                position_indices = start_positions + batch_offsets  # [batch_size, max_valid_tokens]
                valid_mask = batch_offsets < valid_token_counts  # [batch_size, max_valid_tokens]
                
                # Get the row indices (cache indices) for each valid position
                row_indices = non_full_cache_idxs.unsqueeze(1).expand(-1, max_valid_tokens)[valid_mask]
                col_indices = position_indices[valid_mask]
                
                # Set the attention mask values
                self.attention_mask[row_indices, col_indices] = 1
            

        def maybe_shift_attention_mask(
            self, num_valid_tokens: List[int], cache_idxs: List[int]
        ):
            for batch_idx, cache_idx in enumerate(cache_idxs):
                new_text_len = num_valid_tokens[batch_idx]
                if new_text_len == 0:
                    continue  # skip padded batch entry

                # Same token counts for all layers when we start, so we take 0th
                curr_text_cache_len = self.text_token_counts[0][cache_idx].item()

                if curr_text_cache_len + new_text_len <= self.text_sliding_window:
                    # If we are under the sliding window length, shift the entire cache left
                    # Since we setup the max cache length with enough buffer, this will ONLY drop
                    # left padding tokens out
                    shift = new_text_len
                    self._shift_attention_mask_left(cache_idx, shift)
                else:
                    # Shift entire cache left to make room for full text sliding window
                    shift_amount = self.text_sliding_window - curr_text_cache_len
                    # If this is <=0, we are already above the sliding window, so the attention mask stays the same
                    if shift_amount > 0:
                        self._shift_attention_mask_left(cache_idx, shift_amount)

    
    """
    Static cache update
    - respects per-batch text token limits
    - per-batch valid token lengths (right-padded inputs)

    kv states are expected to have shape [batch_size, kv_heads, T_pad, head_dim]
    They may have different `true` lengths, to account for multi token preds, or beacon tokens
    Expects `num_valid_tokens` in cache_kwargs: a tensor of shape (B,) indicating the number
    of actual (non-padded) tokens to add per batch element.
    """
    #TODO: Ensure the in-place edit for text_token_counts actually works
    def _decode_update(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        text_token_counts: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_valid_tokens: torch.Tensor = cache_kwargs.get("num_valid_tokens")  # shape: (B,)
        assert num_valid_tokens is not None, ("`num_valid_tokens` must be provided in `cache_kwargs`")
        device = key_states.device

        batch_size, num_head, seq_len, head_dim = key_states.shape
        sliding_window = self.text_sliding_window
        max_cache_len = self.max_cache_len
        cache_text_start = max_cache_len - sliding_window
        new_text_lengths = text_token_counts + num_valid_tokens
        slide_amounts = torch.clamp(new_text_lengths - sliding_window, min=0)
        needs_rotate = slide_amounts > 0

        # Rotate the cache if needed
        if torch.any(needs_rotate):
            k_slice = key_cache[:, :, -sliding_window:]  # shape: [B, H, W, D]
            v_slice = value_cache[:, :, -sliding_window:]  # same shape

            cache_indices = torch.arange(sliding_window, device=device).unsqueeze(0).repeat(batch_size, 1)  # [B, W]
            rolled_indices = (cache_indices + slide_amounts.unsqueeze(1)) % sliding_window  # [B, W]

            # We need to expand indices to shape: [B, 1, W, 1] to broadcast with k_slice
            rolled_indices = rolled_indices.unsqueeze(1).unsqueeze(-1).expand(-1, num_head, -1, head_dim)

            k_slice_rolled = k_slice.gather(dim=2, index=rolled_indices)
            v_slice_rolled = v_slice.gather(dim=2, index=rolled_indices)

            key_cache[:, :, -sliding_window:] = k_slice_rolled
            value_cache[:, :, -sliding_window:] = v_slice_rolled

        # Insert new tokens
        insert_indices = torch.where(
            needs_rotate,
            max_cache_len - num_valid_tokens,
            text_token_counts + cache_text_start
        )
        ## TODO Actually do the insertion for differing num_valid_tokens
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, seq_len)
        seq_indices = insert_indices.unsqueeze(1) + torch.arange(seq_len, device=device)
        expanded_seq_indices = seq_indices.unsqueeze(1).unsqueeze(-1).expand(-1, num_head, -1, head_dim)
        key_cache.scatter_(2, expanded_seq_indices, key_states)
        value_cache.scatter_(2, expanded_seq_indices, value_states)

        # In-place edit - Mutates
        text_token_counts += num_valid_tokens
        text_token_counts.clamp_(sliding_window)

        return key_cache, value_cache

    # The attention mask managed by our kv cache automatically masks the tokens
    # in the cache, so we can return full length for HF to use in other places
    # This is mainly utilized in the cache_positions creation
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.max_cache_len
