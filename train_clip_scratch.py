import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from clip_scratch_model import (
    SimpleTokenizer,
    build_image_transform,
    build_model_from_config,
    captions_from_entries,
    get_default_model_config,
)
from clip_utils import (
    CLIPTrainDataset,
    build_retrieval_benchmark,
    compute_retrieval_metrics,
    load_image_caption_entries,
    load_rgb_image,
    plot_embedding_projection,
    plot_image_to_text_examples,
    plot_text_to_image_examples,
    plot_training_curves,
    sample_image_to_text_retrieval,
    sample_text_to_image_retrieval,
    save_json,
    set_seed,
    split_entries,
)

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIP-style model with ViT image encoder and configurable text encoder.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="esri_rgb_esa_landcover_zoom_17_patch_224_captions_internvl38b.csv",
    )
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs_clip_scratch")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--samples_per_image", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda:0", "cuda:1","cuda:2","cuda:3", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_each_epoch", action="store_true")
    parser.add_argument("--vis_queries", type=int, default=6)
    parser.add_argument("--vis_top_k", type=int, default=5)
    parser.add_argument("--max_train_steps_per_epoch", type=int, default=None)

    parser.add_argument("--max_text_len", type=int, default=32)
    parser.add_argument("--min_token_freq", type=int, default=1)
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--text_width", type=int, default=256)
    parser.add_argument("--text_heads", type=int, default=8)
    parser.add_argument("--text_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--image_encoder_name",
        type=str,
        default="vit_b_16",
        choices=["vit_b_16", "vit_b_32", "resnet18", "resnet34"],
    )
    parser.add_argument(
        "--image_backbone_pretrained",
        action="store_true",
        help="Initialize image backbone with torchvision pretrained weights (optional).",
    )
    parser.add_argument(
        "--text_encoder_name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="HF text model name/path, or 'scratch' for the custom transformer tokenizer+encoder.",
    )
    parser.add_argument(
        "--text_backbone_pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained weights for HF text encoder.",
    )
    parser.add_argument(
        "--text_backbone_trainable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If false, freeze HF text encoder and train only projection/image branch.",
    )
    parser.add_argument(
        "--text_local_files_only",
        action="store_true",
        help="Load HF text assets only from local cache/files (no network).",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def is_scratch_text_encoder(text_encoder_name: str) -> bool:
    return str(text_encoder_name).lower() == "scratch"


def clip_symmetric_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i + loss_t)


def make_collate_fn(text_tokenizer: Any, image_transform, max_text_len: int, use_scratch_tokenizer: bool):
    def collate_fn(batch):
        images = torch.stack([image_transform(sample["image"]) for sample in batch], dim=0)
        texts = [sample["text"] for sample in batch]
        if use_scratch_tokenizer:
            input_ids, attention_mask = text_tokenizer.batch_encode(texts)
        else:
            encoded = text_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_text_len,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        return {
            "pixel_values": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    return collate_fn


def _iter_batches(items: Sequence, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.no_grad()
def encode_image_paths_scratch(
    model,
    image_paths: Sequence[str],
    image_transform,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    features: List[torch.Tensor] = []
    for batch_paths in _iter_batches(list(image_paths), batch_size):
        images = torch.stack([image_transform(load_rgb_image(path)) for path in batch_paths], dim=0).to(device)
        batch_features = model.get_image_features(images)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        features.append(batch_features.cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def encode_texts_model(
    model,
    text_tokenizer: Any,
    texts: Sequence[str],
    batch_size: int,
    device: torch.device,
    max_text_len: int,
    use_scratch_tokenizer: bool,
) -> torch.Tensor:
    model.eval()
    features: List[torch.Tensor] = []
    for batch_texts in _iter_batches(list(texts), batch_size):
        if use_scratch_tokenizer:
            input_ids, attention_mask = text_tokenizer.batch_encode(batch_texts)
        else:
            encoded = text_tokenizer(
                list(batch_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_text_len,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        batch_features = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        features.append(batch_features.cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def evaluate_retrieval(
    model,
    text_tokenizer: Any,
    image_transform,
    entries,
    device: torch.device,
    eval_batch_size: int,
    max_text_len: int,
    use_scratch_tokenizer: bool,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    image_paths, class_names, captions, caption_to_image_idx, image_to_caption_idxs = build_retrieval_benchmark(entries)

    image_embeds = encode_image_paths_scratch(
        model=model,
        image_paths=image_paths,
        image_transform=image_transform,
        batch_size=eval_batch_size,
        device=device,
    )
    text_embeds = encode_texts_model(
        model=model,
        text_tokenizer=text_tokenizer,
        texts=captions,
        batch_size=eval_batch_size,
        device=device,
        max_text_len=max_text_len,
        use_scratch_tokenizer=use_scratch_tokenizer,
    )
    metrics = compute_retrieval_metrics(
        image_embeds=image_embeds,
        text_embeds=text_embeds,
        caption_to_image_idx=caption_to_image_idx,
        image_to_caption_idxs=image_to_caption_idxs,
        ks=(1, 5, 10),
    )
    extras = {
        "image_paths": image_paths,
        "class_names": class_names,
        "captions": captions,
        "caption_to_image_idx": caption_to_image_idx,
        "image_to_caption_idxs": image_to_caption_idxs,
        "image_embeds": image_embeds,
        "text_embeds": text_embeds,
    }
    return metrics, extras


def build_model_config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    config = get_default_model_config()
    config["embed_dim"] = args.embed_dim
    config["text_width"] = args.text_width
    config["text_heads"] = args.text_heads
    config["text_layers"] = args.text_layers
    config["text_max_len"] = args.max_text_len
    config["dropout"] = args.dropout
    config["image_encoder_name"] = args.image_encoder_name
    config["image_backbone_pretrained"] = args.image_backbone_pretrained
    config["text_encoder_name"] = args.text_encoder_name
    config["text_backbone_pretrained"] = args.text_backbone_pretrained
    config["text_backbone_trainable"] = args.text_backbone_trainable
    config["text_local_files_only"] = args.text_local_files_only
    config["image_size"] = args.image_size
    return config


def save_checkpoint(
    ckpt_path: Path,
    model,
    model_config: Dict[str, object],
    epoch: int,
    best_score: float,
    tokenizer_vocab_size: int,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "epoch": epoch,
        "best_score": best_score,
        "tokenizer_vocab_size": tokenizer_vocab_size,
    }
    torch.save(payload, ckpt_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    entries = load_image_caption_entries(args.csv_path, image_root=args.image_root, min_captions=1)
    train_entries, val_entries, test_entries = split_entries(
        entries,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Total images: {len(entries)} | Train: {len(train_entries)} | Val: {len(val_entries)} | Test: {len(test_entries)}")
    print(f"Using device: {device}")

    use_scratch_tokenizer = is_scratch_text_encoder(args.text_encoder_name)
    if use_scratch_tokenizer:
        train_texts = captions_from_entries(train_entries)
        text_tokenizer = SimpleTokenizer.build_from_texts(
            train_texts,
            max_length=args.max_text_len,
            min_freq=args.min_token_freq,
            max_vocab_size=args.max_vocab_size,
        )
        tokenizer_vocab_size = text_tokenizer.vocab_size
        text_tokenizer.save(str(out_dir / "tokenizer.json"))
        print(f"Text encoder: scratch tokenizer+transformer | vocab_size={tokenizer_vocab_size}")
    else:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is required for HF tokenizer loading")
        try:
            text_tokenizer = AutoTokenizer.from_pretrained(
                args.text_encoder_name,
                local_files_only=args.text_local_files_only,
                use_fast=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load HF text tokenizer. Check --text_encoder_name or --text_local_files_only."
            ) from exc
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token or text_tokenizer.cls_token
        tokenizer_vocab_size = 0
        text_tokenizer.save_pretrained(out_dir / "text_tokenizer")
        print(f"Text encoder: {args.text_encoder_name}")

    model_config = build_model_config_from_args(args)
    with open(out_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    try:
        model = build_model_from_config(
            vocab_size=tokenizer_vocab_size,
            config=model_config,
            image_backbone_pretrained=bool(model_config.get("image_backbone_pretrained", False)),
            text_backbone_pretrained=bool(model_config.get("text_backbone_pretrained", False)),
            text_local_files_only=bool(model_config.get("text_local_files_only", False)),
        ).to(device)
    except Exception as exc:
        if bool(model_config.get("image_backbone_pretrained", False)):
            raise RuntimeError(
                "Failed to load torchvision pretrained image backbone. "
                "If running offline, remove --image_backbone_pretrained."
            ) from exc
        if not use_scratch_tokenizer:
            raise RuntimeError(
                "Failed to build HF text encoder. If running offline, use --text_local_files_only with cached assets."
            ) from exc
        raise

    image_transform = build_image_transform(image_size=int(model_config["image_size"]))

    train_dataset = CLIPTrainDataset(train_entries, samples_per_image=args.samples_per_image)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_collate_fn(
            text_tokenizer=text_tokenizer,
            image_transform=image_transform,
            max_text_len=args.max_text_len,
            use_scratch_tokenizer=use_scratch_tokenizer,
        ),
        drop_last=False,
    )
    if len(train_loader) == 0:
        raise ValueError("No training steps available. Check batch size and dataset.")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    effective_steps_per_epoch = (
        min(len(train_loader), args.max_train_steps_per_epoch)
        if args.max_train_steps_per_epoch
        else len(train_loader)
    )
    total_steps = args.epochs * effective_steps_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    history = {"epoch": [], "train_loss": [], "val_t2i_r@1": [], "val_i2t_r@1": []}
    best_dir = out_dir / "best_model"
    best_ckpt_path = best_dir / "model.pt"
    best_score = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for step_idx, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(args.amp and device.type == "cuda")):
                logits_i, logits_t, _, _ = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = clip_symmetric_loss(logits_i, logits_t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            if args.max_train_steps_per_epoch and step_idx >= args.max_train_steps_per_epoch:
                break

        train_loss = running_loss / max(effective_steps_per_epoch, 1)
        val_metrics, _ = evaluate_retrieval(
            model=model,
            text_tokenizer=text_tokenizer,
            image_transform=image_transform,
            entries=val_entries,
            device=device,
            eval_batch_size=args.eval_batch_size,
            max_text_len=args.max_text_len,
            use_scratch_tokenizer=use_scratch_tokenizer,
        )
        score = val_metrics["t2i_r@1"] + val_metrics["i2t_r@1"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_t2i_r@1"].append(val_metrics["t2i_r@1"])
        history["val_i2t_r@1"].append(val_metrics["i2t_r@1"])

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_t2i_r@1={val_metrics['t2i_r@1']:.4f} "
            f"val_i2t_r@1={val_metrics['i2t_r@1']:.4f}"
        )

        plot_training_curves(history, str(plot_dir / "training_curves.png"))
        save_json(history, str(out_dir / "history.json"))

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                ckpt_path=best_ckpt_path,
                model=model,
                model_config=model_config,
                epoch=epoch,
                best_score=best_score,
                tokenizer_vocab_size=tokenizer_vocab_size,
            )
            if use_scratch_tokenizer:
                text_tokenizer.save(str(best_dir / "tokenizer.json"))
            else:
                text_tokenizer.save_pretrained(str(best_dir / "text_tokenizer"))
            with open(best_dir / "model_config.json", "w", encoding="utf-8") as f:
                json.dump(model_config, f, indent=2)
            save_json({"best_epoch": best_epoch, "best_score": best_score}, str(out_dir / "best_summary.json"))

        if args.save_each_epoch:
            epoch_dir = out_dir / f"epoch_{epoch:02d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                ckpt_path=epoch_dir / "model.pt",
                model=model,
                model_config=model_config,
                epoch=epoch,
                best_score=score,
                tokenizer_vocab_size=tokenizer_vocab_size,
            )
            if use_scratch_tokenizer:
                text_tokenizer.save(str(epoch_dir / "tokenizer.json"))
            else:
                text_tokenizer.save_pretrained(str(epoch_dir / "text_tokenizer"))
            with open(epoch_dir / "model_config.json", "w", encoding="utf-8") as f:
                json.dump(model_config, f, indent=2)

    print(f"Best epoch: {best_epoch}, best R@1 sum: {best_score:.4f}")

    ckpt = torch.load(best_ckpt_path, map_location=device)
    load_model_config = dict(ckpt["model_config"])
    use_scratch_eval = is_scratch_text_encoder(load_model_config.get("text_encoder_name", "scratch"))

    if use_scratch_eval:
        eval_tokenizer = SimpleTokenizer.load(str(best_dir / "tokenizer.json"))
        eval_vocab_size = eval_tokenizer.vocab_size
    else:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is required for HF tokenizer loading")
        eval_tokenizer = AutoTokenizer.from_pretrained(
            str(best_dir / "text_tokenizer"), local_files_only=True, use_fast=True,
        )
        eval_vocab_size = 0

    # Rebuild model architecture then load trained weights from checkpoint.
    # text_backbone_pretrained=True so the HF model structure is created from
    # the hub config; the actual weights come from load_state_dict right after.
    eval_model = build_model_from_config(
        vocab_size=eval_vocab_size,
        config=load_model_config,
        image_backbone_pretrained=False,
        text_backbone_pretrained=(not use_scratch_eval),
        text_local_files_only=False,
    ).to(device)
    eval_model.load_state_dict(ckpt["model_state_dict"])
    eval_model.eval()
    eval_transform = build_image_transform(image_size=int(load_model_config["image_size"]))
    eval_max_text_len = int(load_model_config.get("text_max_len", args.max_text_len))

    test_metrics, test_extras = evaluate_retrieval(
        model=eval_model,
        text_tokenizer=eval_tokenizer,
        image_transform=eval_transform,
        entries=test_entries,
        device=device,
        eval_batch_size=args.eval_batch_size,
        max_text_len=eval_max_text_len,
        use_scratch_tokenizer=use_scratch_eval,
    )
    print("Test metrics:")
    for key, value in sorted(test_metrics.items()):
        print(f"  {key}: {value:.4f}")
    save_json(test_metrics, str(out_dir / "test_metrics.json"))

    plot_embedding_projection(
        image_embeds=test_extras["image_embeds"],
        class_names=test_extras["class_names"],
        output_path=str(plot_dir / "test_embedding_projection.png"),
        seed=args.seed,
    )

    text_query_idxs, t2i_retrieved, t2i_gt = sample_text_to_image_retrieval(
        text_embeds=test_extras["text_embeds"],
        image_embeds=test_extras["image_embeds"],
        caption_to_image_idx=test_extras["caption_to_image_idx"],
        num_queries=args.vis_queries,
        top_k=args.vis_top_k,
        seed=args.seed,
    )
    query_texts = [test_extras["captions"][idx] for idx in text_query_idxs]
    plot_text_to_image_examples(
        query_texts=query_texts,
        retrieved_image_indices=t2i_retrieved,
        image_paths=test_extras["image_paths"],
        gt_image_indices=t2i_gt,
        output_path=str(plot_dir / "qualitative_text_to_image.png"),
    )

    image_query_idxs, i2t_retrieved = sample_image_to_text_retrieval(
        image_embeds=test_extras["image_embeds"],
        text_embeds=test_extras["text_embeds"],
        image_to_caption_idxs=test_extras["image_to_caption_idxs"],
        num_queries=args.vis_queries,
        top_k=args.vis_top_k,
        seed=args.seed,
    )
    plot_image_to_text_examples(
        query_image_indices=image_query_idxs,
        retrieved_caption_indices=i2t_retrieved,
        image_paths=test_extras["image_paths"],
        captions=test_extras["captions"],
        image_to_caption_idxs=test_extras["image_to_caption_idxs"],
        output_path=str(plot_dir / "qualitative_image_to_text.png"),
    )

    with open(out_dir / "run_args.txt", "w", encoding="utf-8") as f:
        for key, value in vars(args).items():
            f.write(f"{key}={value}\n")
    print(f"Saved artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
