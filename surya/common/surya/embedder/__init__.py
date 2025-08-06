import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTokenEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.bbox_embed = nn.ModuleList(
            [
                nn.Embedding(
                    config.bbox_size + config.special_token_count,
                    config.bbox_embed_size,
                )
                for _ in range(6)
            ]
        )

    def embed(
        self,
        input_tokens: torch.Tensor,
        input_bboxes: torch.Tensor | None,
        embed_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        # Embed tokens
        token_embeds = self.token_embed(input_tokens)

        # Optionally embed boxes
        if input_bboxes is not None:  # Is none in prefill
            input_bboxes = input_bboxes.to(torch.long)
            bbox_loss_ignore_mask = (input_bboxes[:, :, 0] < 0).unsqueeze(-1)
            input_bboxes = torch.where(
                input_bboxes > 0, input_bboxes, torch.zeros_like(input_bboxes)
            )

            bbox_embeds = torch.sum(
                torch.stack(
                    [
                        self.bbox_embed[i](input_bboxes[:, :, i])
                        for i in range(len(self.bbox_embed))
                    ],
                    dim=-1,
                ),
                dim=-1,
            )

            bbox_embeds = F.pad(
                bbox_embeds, (token_embeds.shape[-1] - bbox_embeds.shape[-1], 0)
            )
            embed_bboxes = embed_bboxes.unsqueeze(1).unsqueeze(1).expand_as(bbox_embeds)
            bbox_loss_ignore_mask = bbox_loss_ignore_mask.expand_as(bbox_embeds)

            zero_boxes = torch.zeros_like(token_embeds)
            bbox_embeds = torch.where(embed_bboxes, bbox_embeds, zero_boxes)
            bbox_embeds = torch.where(bbox_loss_ignore_mask, zero_boxes, bbox_embeds)

            token_embeds = token_embeds + bbox_embeds

        return token_embeds
