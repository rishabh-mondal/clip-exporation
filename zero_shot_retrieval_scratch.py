import argparse
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import torch

from clip_scratch_model import SimpleTokenizer, build_image_transform, build_model_from_config
from clip_utils import (
    build_retrieval_benchmark,
    compute_retrieval_metrics,
    load_image_caption_entries,
    load_rgb_image,
    plot_image_to_text_examples,
    plot_text_to_image_examples,
    save_json,
    set_seed,
    split_entries,
)

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot retrieval with scratch CLIP checkpoint.")
    parser.add_argument("--model_dir", type=str, default="outputs_clip_scratch/best_model")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="esri_rgb_esa_landcover_zoom_17_patch_224_captions_internvl38b.csv",
    )
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--query_text", type=str, default=None)
    parser.add_argument("--query_image", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:3", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default="outputs_clip_scratch/retrieval_demo")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter an interactive loop where you can type text or image-path queries.",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def is_scratch_text_encoder(text_encoder_name: str) -> bool:
    return str(text_encoder_name).lower() == "scratch"


def pick_entries(args, all_entries):
    train_entries, val_entries, test_entries = split_entries(
        all_entries,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if args.split == "train":
        return train_entries
    if args.split == "val":
        return val_entries
    if args.split == "test":
        return test_entries
    return all_entries


def _iter_batches(items: Sequence, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def tokenize_texts(
    text_tokenizer: Any,
    texts: Sequence[str],
    max_text_len: int,
    use_scratch_tokenizer: bool,
):
    if use_scratch_tokenizer:
        return text_tokenizer.batch_encode(texts)
    encoded = text_tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_len,
    )
    return encoded["input_ids"], encoded["attention_mask"]


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
        input_ids, attention_mask = tokenize_texts(
            text_tokenizer=text_tokenizer,
            texts=batch_texts,
            max_text_len=max_text_len,
            use_scratch_tokenizer=use_scratch_tokenizer,
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        batch_features = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        features.append(batch_features.cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def embed_single_text(
    model,
    text_tokenizer: Any,
    text: str,
    device: torch.device,
    max_text_len: int,
    use_scratch_tokenizer: bool,
) -> torch.Tensor:
    input_ids, attention_mask = tokenize_texts(
        text_tokenizer=text_tokenizer,
        texts=[text],
        max_text_len=max_text_len,
        use_scratch_tokenizer=use_scratch_tokenizer,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    feat = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return feat.cpu()


@torch.no_grad()
def embed_single_image(model, image_transform, image_path: str, device: torch.device) -> torch.Tensor:
    image = image_transform(load_rgb_image(image_path)).unsqueeze(0).to(device)
    feat = model.get_image_features(pixel_values=image)
    feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return feat.cpu()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    load_model_config = dict(ckpt["model_config"])
    use_scratch_tokenizer = is_scratch_text_encoder(load_model_config.get("text_encoder_name", "scratch"))

    if use_scratch_tokenizer:
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Missing tokenizer file: {tokenizer_path}")
        text_tokenizer = SimpleTokenizer.load(str(tokenizer_path))
        tokenizer_vocab_size = text_tokenizer.vocab_size
    else:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is required for HF tokenizer loading")
        tokenizer_dir = model_dir / "text_tokenizer"
        if not tokenizer_dir.exists():
            raise FileNotFoundError(f"Missing tokenizer directory: {tokenizer_dir}")
        text_tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir), local_files_only=True, use_fast=True,
        )
        tokenizer_vocab_size = 0

    # Rebuild model architecture then load trained weights from checkpoint.
    model = build_model_from_config(
        vocab_size=tokenizer_vocab_size,
        config=load_model_config,
        image_backbone_pretrained=False,
        text_backbone_pretrained=(not use_scratch_tokenizer),
        text_local_files_only=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    max_text_len = int(load_model_config.get("text_max_len", 32))
    image_transform = build_image_transform(image_size=int(load_model_config["image_size"]))

    all_entries = load_image_caption_entries(args.csv_path, image_root=args.image_root)
    entries = pick_entries(args, all_entries)
    if len(entries) == 0:
        raise ValueError(f"No entries found for split '{args.split}'")

    image_paths, class_names, captions, caption_to_image_idx, image_to_caption_idxs = build_retrieval_benchmark(entries)
    image_embeds = encode_image_paths_scratch(model, image_paths, image_transform, batch_size=args.batch_size, device=device)
    text_embeds = encode_texts_model(
        model=model,
        text_tokenizer=text_tokenizer,
        texts=captions,
        batch_size=args.batch_size,
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
    print(f"Indexed split: {args.split}")
    print(f"Images: {len(image_paths)}, Captions: {len(captions)}")
    print("Recall metrics on indexed split:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")
    save_json(metrics, str(out_dir / f"{args.split}_metrics.json"))

    # --- helper closures for running a single query -------------------------
    top_k_image = min(args.top_k, len(image_paths))
    top_k_text = min(args.top_k, len(captions))
    query_counter = {"text": 0, "image": 0}

    def run_text_query(query: str) -> None:
        query_counter["text"] += 1
        text_feat = embed_single_text(
            model=model,
            text_tokenizer=text_tokenizer,
            text=query,
            device=device,
            max_text_len=max_text_len,
            use_scratch_tokenizer=use_scratch_tokenizer,
        )
        sims = (text_feat @ image_embeds.T).squeeze(0)
        retrieved = torch.topk(sims, k=top_k_image).indices.cpu().numpy()

        print(f"\nText query: {query}")
        print("Top retrieved images:")
        for rank, image_idx in enumerate(retrieved, start=1):
            print(f"  {rank}. {image_paths[int(image_idx)]} | class={class_names[int(image_idx)]}")

        tag = f"text_query_{query_counter['text']:03d}"
        plot_text_to_image_examples(
            query_texts=[query],
            retrieved_image_indices=np.expand_dims(retrieved, axis=0),
            image_paths=image_paths,
            gt_image_indices=[-1],
            output_path=str(out_dir / f"{tag}.png"),
        )
        print(f"  Saved visualization -> {out_dir / f'{tag}.png'}")

    def run_image_query(img_path: str) -> None:
        import os
        if not os.path.exists(img_path):
            print(f"  [error] File not found: {img_path}")
            return
        query_counter["image"] += 1
        image_feat = embed_single_image(model, image_transform, img_path, device=device)
        sims = (image_feat @ text_embeds.T).squeeze(0)
        retrieved = torch.topk(sims, k=top_k_text).indices.cpu().numpy()

        print(f"\nImage query: {img_path}")
        print("Top retrieved captions:")
        for rank, caption_idx in enumerate(retrieved, start=1):
            print(f"  {rank}. {captions[int(caption_idx)]}")

        tag = f"image_query_{query_counter['image']:03d}"
        plot_image_to_text_examples(
            query_image_indices=[0],
            retrieved_caption_indices=np.expand_dims(retrieved, axis=0),
            image_paths=[img_path],
            captions=captions,
            image_to_caption_idxs=[[]],
            output_path=str(out_dir / f"{tag}.png"),
        )
        print(f"  Saved visualization -> {out_dir / f'{tag}.png'}")

    # --- run CLI queries if provided ----------------------------------------
    if args.query_text:
        run_text_query(args.query_text)
    if args.query_image:
        run_image_query(args.query_image)

    # --- interactive mode ----------------------------------------------------
    if args.interactive or (not args.query_text and not args.query_image):
        print("\n" + "=" * 60)
        print("INTERACTIVE RETRIEVAL MODE")
        print("=" * 60)
        print("Type a text prompt  -> retrieves matching images  (text-to-image)")
        print("Type an image path  -> retrieves matching captions (image-to-text)")
        print("Commands:  quit / exit / q  ->  stop")
        print("=" * 60)

        import os
        while True:
            try:
                user_input = input("\n[query] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive mode.")
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            # Heuristic: if it looks like a file path, treat as image query
            if os.path.exists(user_input) or user_input.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                run_image_query(user_input)
            else:
                run_text_query(user_input)

    print(f"\nAll visualizations saved in: {out_dir}")


if __name__ == "__main__":
    main()
