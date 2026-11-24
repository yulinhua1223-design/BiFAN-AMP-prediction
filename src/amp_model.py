from __future__ import annotations
import torch
from torch import nn
from transformers import AutoModel

from fusion_variants import build_fusion


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class HandcraftedFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            SEBlock(64),
            nn.AdaptiveMaxPool1d(1)
        )

        self.projector = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.unsqueeze(1)
        x = self.conv_branch(x).squeeze(-1)
        return self.projector(x)


class AmpPredictor(nn.Module):
    """Dual-branch AMP classifier.

    ``fusion`` selects the fusion operator (see ``fusion_variants``), ``branch``
    selects which representation feeds the classifier for the component ablation
    (``full`` / ``esm_only`` / ``hand_only``) and ``aux_lambda > 0`` adds an
    auxiliary classification loss on the handcrafted branch alone so that the
    branch has to stay predictive on its own.
    """

    def __init__(self, esm_model_name: str, handcrafted_input_dim: int,
                 projection_dim: int = 256, dropout: float = 0.1,
                 fusion: str = "add", branch: str = "full",
                 aux_lambda: float = 0.0) -> None:
        super().__init__()
        if branch not in {"full", "esm_only", "hand_only"}:
            raise ValueError(f"unknown branch mode {branch!r}")

        self.esm = AutoModel.from_pretrained(esm_model_name)
        self.attention_query = nn.Linear(self.esm.config.hidden_size, 1)

        self.global_projection = nn.Sequential(
            nn.Linear(self.esm.config.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

        self.handcrafted_encoder = HandcraftedFeatureEncoder(handcrafted_input_dim, projection_dim, dropout)
        self.fusion_engine = build_fusion(
            fusion,
            d_model=projection_dim,
            dropout=dropout,
            token_dim=self.esm.config.hidden_size,
        )

        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 2),
        )

        self.fusion_name = fusion
        self.branch = branch
        self.aux_lambda = float(aux_lambda)
        # the component ablation that drops the sequence branch does not need the
        # language model at all: skipping it removes ~99% of the compute and all
        # of its parameters are frozen so the optimiser never touches them
        self.skip_esm = branch == "hand_only"
        if self.skip_esm:
            for parameter in self.esm.parameters():
                parameter.requires_grad = False
        self.handcrafted_head = None
        if self.aux_lambda > 0.0:
            self.handcrafted_head = nn.Sequential(
                nn.Linear(projection_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, 2),
            )

    def attention_pooling(self, token_embeddings, attention_mask):
        scores = self.attention_query(token_embeddings)
        mask = attention_mask.unsqueeze(-1).float()
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(token_embeddings * weights, dim=1)

    def forward(self, input_ids, attention_mask, handcrafted_features):
        if self.skip_esm:
            token_embeddings = None
            global_feat = None
        else:
            outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            global_feat = self.global_projection(
                self.attention_pooling(token_embeddings, attention_mask)
            )

        handcrafted_feat = self.handcrafted_encoder(handcrafted_features)

        if self.branch == "esm_only":
            fused = global_feat
            logits = self.classifier(fused)
        elif self.branch == "hand_only":
            fused = handcrafted_feat
            logits = self.classifier(fused)
        else:
            fused = self.fusion_engine(
                global_feat,
                handcrafted_feat,
                tokens=token_embeddings,
                token_mask=attention_mask,
            )
            logits = self.classifier(fused)

        result = {
            "logits": logits,
            "fused_feature": fused,
            "global_feature": global_feat,
            "handcrafted_feature": handcrafted_feat,
        }
        if self.handcrafted_head is not None:
            result["aux_logits"] = self.handcrafted_head(handcrafted_feat)
        return result
