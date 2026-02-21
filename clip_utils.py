import json
import os
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class ImageCaptionEntry:
    image_path: str
    class_name: str
    captions: List[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_image_path(raw_path: str, class_name: str, image_root: Optional[str]) -> str:
    raw_path = str(raw_path)
    if os.path.exists(raw_path):
        return raw_path

    if image_root:
        candidate = os.path.join(image_root, class_name, os.path.basename(raw_path))
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Image path does not exist: {raw_path}")


def _deduplicate_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def load_image_caption_entries(
    csv_path: str,
    image_root: Optional[str] = None,
    min_captions: int = 1,
) -> List[ImageCaptionEntry]:
    df = pd.read_csv(csv_path)
    required_cols = {"image_path", "class_name", "caption"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    sort_cols = ["image_path"]
    if "caption_id" in df.columns:
        sort_cols.append("caption_id")
    else:
        sort_cols.append("caption")
    df = df.sort_values(sort_cols)

    entries: List[ImageCaptionEntry] = []
    grouped = df.groupby(["image_path", "class_name"], sort=False)
    for (raw_path, class_name), group in grouped:
        captions = [
            str(caption).strip()
            for caption in group["caption"].tolist()
            if isinstance(caption, str) and str(caption).strip()
        ]
        captions = _deduplicate_keep_order(captions)
        if len(captions) < min_captions:
            continue

        image_path = _resolve_image_path(str(raw_path), str(class_name), image_root)
        entries.append(
            ImageCaptionEntry(
                image_path=image_path,
                class_name=str(class_name),
                captions=captions,
            )
        )

    if not entries:
        raise ValueError("No valid image-caption entries were loaded from CSV.")
    return entries


def split_entries(
    entries: Sequence[ImageCaptionEntry],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[ImageCaptionEntry], List[ImageCaptionEntry], List[ImageCaptionEntry]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must satisfy: train_ratio > 0, val_ratio >= 0, train+val < 1.")

    indices = list(range(len(entries)))
    random.Random(seed).shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_entries = [entries[i] for i in train_idx]
    val_entries = [entries[i] for i in val_idx]
    test_entries = [entries[i] for i in test_idx]
    return train_entries, val_entries, test_entries


def load_rgb_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB")


class CLIPTrainDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[ImageCaptionEntry],
        samples_per_image: int = 1,
    ) -> None:
        if not entries:
            raise ValueError("CLIPTrainDataset requires at least one entry.")
        if samples_per_image < 1:
            raise ValueError("samples_per_image must be >= 1")
        self.entries = list(entries)
        self.samples_per_image = samples_per_image

    def __len__(self) -> int:
        return len(self.entries) * self.samples_per_image

    def __getitem__(self, idx: int) -> Dict[str, object]:
        entry = self.entries[idx % len(self.entries)]
        caption = random.choice(entry.captions)
        image = load_rgb_image(entry.image_path)
        return {
            "image": image,
            "text": caption,
            "image_path": entry.image_path,
            "class_name": entry.class_name,
        }


def build_retrieval_benchmark(
    entries: Sequence[ImageCaptionEntry],
) -> Tuple[List[str], List[str], List[str], List[int], List[List[int]]]:
    image_paths: List[str] = []
    class_names: List[str] = []
    captions: List[str] = []
    caption_to_image_idx: List[int] = []
    image_to_caption_idxs: List[List[int]] = []

    for image_idx, entry in enumerate(entries):
        image_paths.append(entry.image_path)
        class_names.append(entry.class_name)
        gt_caption_idxs: List[int] = []
        for caption in entry.captions:
            caption_idx = len(captions)
            captions.append(caption)
            caption_to_image_idx.append(image_idx)
            gt_caption_idxs.append(caption_idx)
        image_to_caption_idxs.append(gt_caption_idxs)

    return image_paths, class_names, captions, caption_to_image_idx, image_to_caption_idxs


def _iter_batches(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.no_grad()
def encode_image_paths(
    model,
    processor,
    image_paths: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    model.eval()
    for batch_paths in _iter_batches(list(image_paths), batch_size):
        images = [load_rgb_image(path) for path in batch_paths]
        batch = processor(images=images, return_tensors="pt")
        pixel_values = batch["pixel_values"].to(device)

        batch_features = model.get_image_features(pixel_values=pixel_values)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        features.append(batch_features.cpu())

    return torch.cat(features, dim=0)


@torch.no_grad()
def encode_texts(
    model,
    processor,
    texts: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    model.eval()
    for batch_texts in _iter_batches(list(texts), batch_size):
        batch = processor(
            text=list(batch_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        batch_features = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        features.append(batch_features.cpu())

    return torch.cat(features, dim=0)


def compute_retrieval_metrics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    caption_to_image_idx: Sequence[int],
    image_to_caption_idxs: Sequence[Sequence[int]],
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    if image_embeds.ndim != 2 or text_embeds.ndim != 2:
        raise ValueError("Embeddings must be rank-2 tensors.")

    ks = sorted({int(k) for k in ks if k > 0})
    if not ks:
        raise ValueError("At least one positive k is required.")

    sim_t2i = text_embeds @ image_embeds.T
    sim_i2t = image_embeds @ text_embeds.T

    max_k_t2i = min(max(ks), sim_t2i.shape[1])
    max_k_i2t = min(max(ks), sim_i2t.shape[1])

    top_t2i = torch.topk(sim_t2i, k=max_k_t2i, dim=1).indices.cpu().numpy()
    top_i2t = torch.topk(sim_i2t, k=max_k_i2t, dim=1).indices.cpu().numpy()

    caption_to_image = np.asarray(caption_to_image_idx, dtype=np.int64)
    metrics: Dict[str, float] = {}

    for k in ks:
        k_t2i = min(k, top_t2i.shape[1])
        k_i2t = min(k, top_i2t.shape[1])

        t2i_hits = (top_t2i[:, :k_t2i] == caption_to_image[:, None]).any(axis=1)
        metrics[f"t2i_r@{k}"] = float(t2i_hits.mean())

        i2t_hits: List[bool] = []
        for image_idx, gt_caption_idxs in enumerate(image_to_caption_idxs):
            retrieved = top_i2t[image_idx, :k_i2t]
            gt = set(gt_caption_idxs)
            i2t_hits.append(any(caption_idx in gt for caption_idx in retrieved))
        metrics[f"i2t_r@{k}"] = float(np.mean(i2t_hits))

    return metrics


def save_json(data: Dict[str, object], output_path: str) -> None:
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_training_curves(history: Dict[str, List[float]], output_path: str) -> None:
    if not history.get("epoch"):
        return

    epochs = history["epoch"]
    train_loss = history.get("train_loss", [])
    val_t2i = history.get("val_t2i_r@1", [])
    val_i2t = history.get("val_i2t_r@1", [])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, marker="o", color="#1f77b4")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.2)

    axes[1].plot(epochs, val_t2i, marker="o", label="Text->Image R@1", color="#2ca02c")
    axes[1].plot(epochs, val_i2t, marker="o", label="Image->Text R@1", color="#d62728")
    axes[1].set_title("Validation Retrieval")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Recall")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_embedding_projection(
    image_embeds: torch.Tensor,
    class_names: Sequence[str],
    output_path: str,
    max_points: int = 1000,
    seed: int = 42,
) -> None:
    vectors = image_embeds.cpu().numpy()
    labels = np.asarray(class_names)
    if len(vectors) == 0:
        return

    if len(vectors) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(vectors), size=max_points, replace=False)
        vectors = vectors[keep]
        labels = labels[keep]

    try:
        from sklearn.manifold import TSNE

        perplexity = max(5, min(30, len(vectors) - 1))
        projected = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=seed,
        ).fit_transform(vectors)
        title = "t-SNE of Image Embeddings"
    except Exception:
        centered = vectors - vectors.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        projected = u[:, :2] * s[:2]
        title = "PCA of Image Embeddings (fallback)"

    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(unique_labels):
        label_mask = labels == label
        ax.scatter(
            projected[label_mask, 0],
            projected[label_mask, 1],
            s=18,
            alpha=0.8,
            color=cmap(idx % 20),
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def sample_text_to_image_retrieval(
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    caption_to_image_idx: Sequence[int],
    num_queries: int,
    top_k: int,
    seed: int = 42,
) -> Tuple[List[int], np.ndarray, List[int]]:
    rng = np.random.default_rng(seed)
    n_text = text_embeds.shape[0]
    num_queries = min(num_queries, n_text)
    query_indices = rng.choice(n_text, size=num_queries, replace=False).tolist()

    sims = text_embeds[query_indices] @ image_embeds.T
    k = min(top_k, sims.shape[1])
    retrieved = torch.topk(sims, k=k, dim=1).indices.cpu().numpy()
    gt = [caption_to_image_idx[idx] for idx in query_indices]
    return query_indices, retrieved, gt


def sample_image_to_text_retrieval(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    image_to_caption_idxs: Sequence[Sequence[int]],
    num_queries: int,
    top_k: int,
    seed: int = 42,
) -> Tuple[List[int], np.ndarray]:
    rng = np.random.default_rng(seed)
    n_image = image_embeds.shape[0]
    num_queries = min(num_queries, n_image)
    query_indices = rng.choice(n_image, size=num_queries, replace=False).tolist()

    sims = image_embeds[query_indices] @ text_embeds.T
    k = min(top_k, sims.shape[1])
    retrieved = torch.topk(sims, k=k, dim=1).indices.cpu().numpy()
    return query_indices, retrieved


def plot_text_to_image_examples(
    query_texts: Sequence[str],
    retrieved_image_indices: np.ndarray,
    image_paths: Sequence[str],
    gt_image_indices: Sequence[int],
    output_path: str,
) -> None:
    rows = len(query_texts)
    if rows == 0:
        return
    cols = retrieved_image_indices.shape[1] + 1

    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.8 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r in range(rows):
        axes[r, 0].axis("off")
        wrapped = textwrap.fill(query_texts[r], width=35)
        axes[r, 0].text(0.02, 0.5, wrapped, va="center", fontsize=10)
        axes[r, 0].set_title("Query Text", fontsize=10)

        for c, image_idx in enumerate(retrieved_image_indices[r], start=1):
            ax = axes[r, c]
            ax.axis("off")
            try:
                image = load_rgb_image(image_paths[int(image_idx)])
                ax.imshow(image)
            except Exception:
                ax.text(0.5, 0.5, "Image load error", ha="center", va="center", fontsize=8)
            title = f"Top-{c}"
            if int(image_idx) == int(gt_image_indices[r]):
                title += " (GT)"
            ax.set_title(title, fontsize=9)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_image_to_text_examples(
    query_image_indices: Sequence[int],
    retrieved_caption_indices: np.ndarray,
    image_paths: Sequence[str],
    captions: Sequence[str],
    image_to_caption_idxs: Sequence[Sequence[int]],
    output_path: str,
) -> None:
    rows = len(query_image_indices)
    if rows == 0:
        return

    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, image_idx in enumerate(query_image_indices):
        ax_image = axes[r, 0]
        ax_text = axes[r, 1]

        ax_image.axis("off")
        try:
            image = load_rgb_image(image_paths[int(image_idx)])
            ax_image.imshow(image)
        except Exception:
            ax_image.text(0.5, 0.5, "Image load error", ha="center", va="center")
        ax_image.set_title("Query Image", fontsize=10)

        gt_set = set(image_to_caption_idxs[int(image_idx)])
        lines = []
        for rank, caption_idx in enumerate(retrieved_caption_indices[r], start=1):
            caption = captions[int(caption_idx)]
            marker = " [GT]" if int(caption_idx) in gt_set else ""
            lines.append(f"{rank}. {caption}{marker}")

        ax_text.axis("off")
        wrapped_lines = [textwrap.fill(line, width=75) for line in lines]
        ax_text.text(0.0, 1.0, "\n\n".join(wrapped_lines), va="top", fontsize=9)
        ax_text.set_title("Top Retrieved Captions", fontsize=10)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
