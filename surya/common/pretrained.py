from typing import Optional

from transformers import PreTrainedModel


class SuryaPreTrainedModel(PreTrainedModel):
    # No-op, so we can set attention however we want in the config
    def _check_and_adjust_attn_implementation(
        self, attn_implementation: Optional[str], **kwargs
    ):
        return attn_implementation
