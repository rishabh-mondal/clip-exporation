import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

try:
    from transformers import AutoConfig, AutoModel
except Exception:
    AutoConfig = None
    AutoModel = None


def basic_tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    # Keep alphanumeric spans as tokens; simple and stable for geospatial captions.
    return re.findall(r"[a-z0-9]+", text)


class SimpleTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(
        self,
        vocab: Dict[str, int],
        max_length: int = 32,
    ) -> None:
        self.vocab = dict(vocab)
        self.max_length = max_length

        required = [self.PAD, self.UNK, self.BOS, self.EOS]
        for tok in required:
            if tok not in self.vocab:
                raise ValueError(f"Tokenizer vocab missing required token: {tok}")

        self.pad_id = self.vocab[self.PAD]
        self.unk_id = self.vocab[self.UNK]
        self.bos_id = self.vocab[self.BOS]
        self.eos_id = self.vocab[self.EOS]

    @classmethod
    def build_from_texts(
        cls,
        texts: Sequence[str],
        max_length: int = 32,
        min_freq: int = 1,
        max_vocab_size: Optional[int] = 30000,
    ) -> "SimpleTokenizer":
        if min_freq < 1:
            raise ValueError("min_freq must be >= 1")

        counter: Counter = Counter()
        for text in texts:
            counter.update(basic_tokenize(str(text)))

        special = [cls.PAD, cls.UNK, cls.BOS, cls.EOS]
        vocab: Dict[str, int] = {tok: idx for idx, tok in enumerate(special)}

        candidates = [(token, freq) for token, freq in counter.items() if freq >= min_freq]
        candidates.sort(key=lambda x: (-x[1], x[0]))

        if max_vocab_size is not None:
            max_core = max(max_vocab_size - len(special), 0)
            candidates = candidates[:max_core]

        for token, _ in candidates:
            vocab[token] = len(vocab)

        return cls(vocab=vocab, max_length=max_length)

    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        tokens = basic_tokenize(str(text))
        token_ids = [self.vocab.get(tok, self.unk_id) for tok in tokens]

        token_ids = token_ids[: max(self.max_length - 2, 0)]
        ids = [self.bos_id] + token_ids + [self.eos_id]
        attn = [1] * len(ids)

        pad_needed = max(self.max_length - len(ids), 0)
        if pad_needed:
            ids = ids + [self.pad_id] * pad_needed
            attn = attn + [0] * pad_needed
        else:
            ids = ids[: self.max_length]
            attn = attn[: self.max_length]

        return ids, attn

    def batch_encode(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        all_ids: List[List[int]] = []
        all_attn: List[List[int]] = []
        for text in texts:
            ids, attn = self.encode(text)
            all_ids.append(ids)
            all_attn.append(attn)
        input_ids = torch.tensor(all_ids, dtype=torch.long)
        attention_mask = torch.tensor(all_attn, dtype=torch.long)
        return input_ids, attention_mask

    def save(self, path: str) -> None:
        payload = {"vocab": self.vocab, "max_length": self.max_length}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(vocab=payload["vocab"], max_length=int(payload["max_length"]))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def build_image_transform(image_size: int = 224) -> transforms.Compose:
    # CLIP-like normalization values.
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class ScratchCLIP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        text_width: int = 256,
        text_heads: int = 8,
        text_layers: int = 4,
        text_max_len: int = 32,
        dropout: float = 0.1,
        image_encoder_name: str = "vit_b_16",
        image_backbone_pretrained: bool = False,
        text_encoder_name: str = "scratch",
        text_backbone_pretrained: bool = False,
        text_backbone_trainable: bool = True,
        text_local_files_only: bool = False,
    ) -> None:
        super().__init__()

        self.image_encoder_name = image_encoder_name
        self._image_encoder_family = "unknown"
        self.image_backbone_pretrained = image_backbone_pretrained

        image_weights = None
        if image_backbone_pretrained:
            try:
                image_weights = torchvision.models.get_model_weights(image_encoder_name).DEFAULT
            except Exception as exc:
                raise ValueError(
                    f"Could not resolve pretrained weights for image_encoder_name='{image_encoder_name}'"
                ) from exc

        if image_encoder_name == "resnet18":
            backbone = torchvision.models.resnet18(weights=image_weights)
            self._image_encoder_family = "resnet"
        elif image_encoder_name == "resnet34":
            backbone = torchvision.models.resnet34(weights=image_weights)
            self._image_encoder_family = "resnet"
        elif image_encoder_name == "vit_b_16":
            backbone = torchvision.models.vit_b_16(weights=image_weights)
            self._image_encoder_family = "vit"
        elif image_encoder_name == "vit_b_32":
            backbone = torchvision.models.vit_b_32(weights=image_weights)
            self._image_encoder_family = "vit"
        else:
            raise ValueError(f"Unsupported image_encoder_name: {image_encoder_name}")

        if self._image_encoder_family == "resnet":
            image_width = backbone.fc.in_features
            self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        elif self._image_encoder_family == "vit":
            image_width = int(backbone.hidden_dim)
            backbone.heads = nn.Identity()
            self.image_encoder = backbone
        else:
            raise ValueError(f"Invalid image encoder family: {self._image_encoder_family}")

        self.image_proj = nn.Linear(image_width, embed_dim, bias=False)

        self.text_encoder_name = str(text_encoder_name)
        self.text_backbone_pretrained = bool(text_backbone_pretrained)
        self.text_backbone_trainable = bool(text_backbone_trainable)
        self.text_max_len = int(text_max_len)
        self._text_encoder_family = "scratch" if self.text_encoder_name.lower() == "scratch" else "hf"

        if self._text_encoder_family == "scratch":
            if vocab_size <= 0:
                raise ValueError("vocab_size must be > 0 when text_encoder_name='scratch'")
            self.token_embed = nn.Embedding(vocab_size, text_width)
            self.pos_embed = nn.Parameter(torch.empty(self.text_max_len, text_width))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=text_width,
                nhead=text_heads,
                dim_feedforward=4 * text_width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.text_transformer = nn.TransformerEncoder(encoder_layer, num_layers=text_layers)
            self.text_ln = nn.LayerNorm(text_width)
            text_feature_dim = text_width
        else:
            if AutoModel is None or AutoConfig is None:
                raise ImportError("transformers is required for pretrained HF text encoders")

            if self.text_backbone_pretrained:
                self.text_encoder = AutoModel.from_pretrained(
                    self.text_encoder_name,
                    local_files_only=text_local_files_only,
                )
            else:
                text_config = AutoConfig.from_pretrained(
                    self.text_encoder_name,
                    local_files_only=text_local_files_only,
                )
                self.text_encoder = AutoModel.from_config(text_config)

            if not self.text_backbone_trainable:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False

            text_feature_dim = int(self.text_encoder.config.hidden_size)

        self.text_proj = nn.Linear(text_feature_dim, embed_dim, bias=False)

        # logit_scale = log(1/0.07) initialization as in CLIP.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))
        self._init_parameters()

    def _init_parameters(self) -> None:
        if self._text_encoder_family == "scratch":
            nn.init.normal_(self.pos_embed, std=0.01)
            nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.image_proj.weight, std=0.02)
        nn.init.normal_(self.text_proj.weight, std=0.02)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.image_encoder(pixel_values)
        if self._image_encoder_family == "resnet":
            x = x.flatten(start_dim=1)
        elif self._image_encoder_family == "vit":
            # With heads=Identity(), torchvision ViT returns CLS features [B, hidden_dim].
            pass
        else:
            raise ValueError(f"Unsupported image encoder family: {self._image_encoder_family}")
        x = self.image_proj(x)
        return x

    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self._text_encoder_family == "scratch":
            if input_ids.shape[1] > self.text_max_len:
                raise ValueError(
                    f"Input text length {input_ids.shape[1]} exceeds configured text_max_len={self.text_max_len}"
                )

            x = self.token_embed(input_ids) + self.pos_embed[: input_ids.shape[1], :].unsqueeze(0)
            key_padding_mask = attention_mask == 0
            x = self.text_transformer(x, src_key_padding_mask=key_padding_mask)
            x = self.text_ln(x)

            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            batch_idx = torch.arange(x.size(0), device=x.device)
            pooled = x[batch_idx, lengths]
            return self.text_proj(pooled)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if not hasattr(outputs, "last_hidden_state"):
            raise ValueError("HF text encoder output does not contain last_hidden_state")
        pooled = outputs.last_hidden_state[:, 0]
        return self.text_proj(pooled)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.get_image_features(pixel_values)
        text_features = self.get_text_features(input_ids, attention_mask)

        image_embeds = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        text_embeds = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * (image_embeds @ text_embeds.t())
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text, image_embeds, text_embeds


def get_default_model_config() -> Dict[str, object]:
    return {
        "embed_dim": 256,
        "text_width": 256,
        "text_heads": 8,
        "text_layers": 4,
        "text_max_len": 32,
        "dropout": 0.1,
        "image_encoder_name": "vit_b_16",
        "image_backbone_pretrained": False,
        "text_encoder_name": "BAAI/bge-base-en-v1.5",
        "text_backbone_pretrained": True,
        "text_backbone_trainable": True,
        "text_local_files_only": False,
        "image_size": 224,
    }


def build_model_from_config(
    vocab_size: int,
    config: Dict[str, object],
    image_backbone_pretrained: Optional[bool] = None,
    text_backbone_pretrained: Optional[bool] = None,
    text_local_files_only: Optional[bool] = None,
) -> ScratchCLIP:
    if image_backbone_pretrained is None:
        image_backbone_pretrained = bool(config.get("image_backbone_pretrained", False))
    if text_backbone_pretrained is None:
        text_backbone_pretrained = bool(config.get("text_backbone_pretrained", False))
    if text_local_files_only is None:
        text_local_files_only = bool(config.get("text_local_files_only", False))

    return ScratchCLIP(
        vocab_size=vocab_size,
        embed_dim=int(config["embed_dim"]),
        text_width=int(config.get("text_width", 256)),
        text_heads=int(config.get("text_heads", 8)),
        text_layers=int(config.get("text_layers", 4)),
        text_max_len=int(config.get("text_max_len", 32)),
        dropout=float(config.get("dropout", 0.1)),
        image_encoder_name=str(config["image_encoder_name"]),
        image_backbone_pretrained=bool(image_backbone_pretrained),
        text_encoder_name=str(config.get("text_encoder_name", "scratch")),
        text_backbone_pretrained=bool(text_backbone_pretrained),
        text_backbone_trainable=bool(config.get("text_backbone_trainable", True)),
        text_local_files_only=bool(text_local_files_only),
    )


def captions_from_entries(entries: Sequence[object]) -> List[str]:
    captions: List[str] = []
    for entry in entries:
        entry_captions = getattr(entry, "captions", None)
        if entry_captions is None:
            continue
        captions.extend([str(c).strip() for c in entry_captions if str(c).strip()])
    return captions
