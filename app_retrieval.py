"""
🛰️ CLIP Zero-Shot Retrieval Explorer
=====================================
Beautiful, fast, interactive demo for satellite imagery CLIP models.

Launch:
    streamlit run app_retrieval.py

Features:
  - 4 model configuration tabs with unique widget keys
  - Multicore parallel image encoding (64 cores)
  - Beautiful modern UI with custom CSS
  - GPU-accelerated inference
  - Training curves, qualitative results, t-SNE visualizations
"""

import json
import os
import textwrap
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Set multiprocessing for data loading
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(16)  # Use multiple cores for torch operations

# ---------------------------------------------------------------------------
# Imports from the project
# ---------------------------------------------------------------------------
from clip_scratch_model import SimpleTokenizer, build_image_transform, build_model_from_config
from clip_utils import (
    build_retrieval_benchmark,
    compute_retrieval_metrics,
    load_image_caption_entries,
    load_rgb_image,
    set_seed,
    split_entries,
)

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


# ---------------------------------------------------------------------------
# Custom CSS for beautiful light styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Main background - clean white */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 50%, #e9ecef 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #e8eaf6 0%, #ede7f6 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #d1c4e9;
    }
    
    .main-header h1 {
        color: #5c6bc0;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: none;
    }
    
    .main-header p {
        color: #7e57c2;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eaf6 100%);
        border: 1px solid rgba(92, 107, 192, 0.2);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #5c6bc0, #7e57c2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(92, 107, 192, 0.08);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.8);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #424242;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #5c6bc0, #7e57c2);
        color: white;
    }
    
    /* Cards */
    .model-card {
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(92, 107, 192, 0.15);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    
    /* Image hover effect */
    .stImage img {
        border-radius: 10px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(0,0,0,0.08);
    }
    
    .stImage img:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 30px rgba(92, 107, 192, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #5c6bc0, #7e57c2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(92, 107, 192, 0.35);
    }
    
    /* Status badges */
    .status-ready {
        background: linear-gradient(90deg, #43a047, #66bb6a);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-pending {
        background: linear-gradient(90deg, #fb8c00, #ffb74d);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f7fa 0%, #e8eaf6 100%);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border: none;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #5c6bc0, #7e57c2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(92, 107, 192, 0.08);
        border-radius: 10px;
    }
    
    /* Caption styling */
    .image-caption {
        font-size: 0.85rem;
        color: #616161;
        text-align: center;
        padding: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #757575;
        font-size: 0.9rem;
    }
    
    /* Text colors for light theme */
    h1, h2, h3, h4 {
        color: #37474f;
    }
    
    p, span, div {
        color: #424242;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""


# ---------------------------------------------------------------------------
# Configuration for 4 model setups
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Full Scratch": {
        "dir": "outputs_clip_scratch/best_model",
        "description": "Both ViT and Text Encoder trained from scratch",
        "icon": "🔧",
        "color": "#FF6B6B",
        "gradient": "linear-gradient(135deg, #FF6B6B, #ee5a5a)",
    },
    "Full Pretrained": {
        "dir": "outputs_clip_pretrained/best_model",
        "description": "Pretrained ViT-B/16 + Pretrained BGE text encoder",
        "icon": "⚡",
        "color": "#4ECDC4",
        "gradient": "linear-gradient(135deg, #4ECDC4, #44a08d)",
    },
    "Pretrained ViT Only": {
        "dir": "outputs_clip_pretrained_vit_only/best_model",
        "description": "Pretrained ViT-B/16 + Text Encoder from scratch",
        "icon": "🖼️",
        "color": "#45B7D1",
        "gradient": "linear-gradient(135deg, #45B7D1, #2e86ab)",
    },
    "Pretrained Text Only": {
        "dir": "outputs_clip_pretrained_text_only/best_model",
        "description": "ViT from scratch + Pretrained BGE text encoder",
        "icon": "📝",
        "color": "#96CEB4",
        "gradient": "linear-gradient(135deg, #96CEB4, #88b89a)",
    },
}

# Default paths
DEFAULT_CSV = "esri_rgb_esa_landcover_zoom_17_patch_224_captions_internvl38b.csv"
DEFAULT_IMAGE_ROOT = "/home/rishabh.mondal/data/esri_rgb_esa_landcover_zoom_17_patch_224"
NUM_WORKERS = 32  # Number of parallel workers for image loading


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _iter_batches(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def is_scratch_text_encoder(name: str) -> bool:
    return str(name).lower() == "scratch"


def tokenize_texts(text_tokenizer, texts, max_text_len, use_scratch):
    if use_scratch:
        return text_tokenizer.batch_encode(texts)
    encoded = text_tokenizer(
        list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_text_len,
    )
    return encoded["input_ids"], encoded["attention_mask"]


def load_and_transform_image(path, transform):
    """Load and transform a single image (for parallel processing)."""
    return transform(load_rgb_image(path))


@torch.no_grad()
def encode_image_paths_parallel(model, image_paths, image_transform, batch_size, device, keep_on_gpu=True):
    """Encode images with parallel loading using ThreadPoolExecutor."""
    model.eval()
    features = []
    
    # Create partial function with transform
    load_fn = partial(load_and_transform_image, transform=image_transform)
    
    for batch_paths in _iter_batches(list(image_paths), batch_size):
        # Parallel image loading
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            batch_tensors = list(executor.map(load_fn, batch_paths))
        
        imgs = torch.stack(batch_tensors).to(device)
        f = model.get_image_features(imgs)
        f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        features.append(f if keep_on_gpu else f.cpu())
        
    return torch.cat(features, dim=0)


@torch.no_grad()
def encode_texts(model, tokenizer, texts, batch_size, device, max_len, use_scratch, keep_on_gpu=True):
    model.eval()
    features = []
    for batch_texts in _iter_batches(list(texts), batch_size):
        ids, mask = tokenize_texts(tokenizer, batch_texts, max_len, use_scratch)
        f = model.get_text_features(input_ids=ids.to(device), attention_mask=mask.to(device))
        f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        features.append(f if keep_on_gpu else f.cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def embed_single_text(model, tokenizer, text, device, max_len, use_scratch, keep_on_gpu=True):
    ids, mask = tokenize_texts(tokenizer, [text], max_len, use_scratch)
    f = model.get_text_features(input_ids=ids.to(device), attention_mask=mask.to(device))
    f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return f if keep_on_gpu else f.cpu()


@torch.no_grad()
def embed_single_image(model, image_transform, image, device, keep_on_gpu=True):
    if isinstance(image, str):
        image = load_rgb_image(image)
    pixel = image_transform(image).unsqueeze(0).to(device)
    f = model.get_image_features(pixel_values=pixel)
    f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return f if keep_on_gpu else f.cpu()


def load_json_safe(path: Path) -> dict:
    """Load JSON file safely, return empty dict if not found."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def check_model_available(model_dir: str) -> bool:
    """Check if model checkpoint exists."""
    model_path = Path(model_dir) / "model.pt"
    return model_path.exists()


# ---------------------------------------------------------------------------
# Cached model loading with progress
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_context(model_dir, csv_path, image_root, split, batch_size, device_str):
    """Load model and build embedding index with multicore acceleration."""
    device = torch.device(device_str)
    model_dir = Path(model_dir)
    output_dir = model_dir.parent
    model_path = model_dir / "model.pt"
    
    if not model_path.exists():
        return None

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = dict(ckpt["model_config"])
    use_scratch = is_scratch_text_encoder(cfg.get("text_encoder_name", "scratch"))

    # Tokenizer
    if use_scratch:
        tok_path = model_dir / "tokenizer.json"
        if not tok_path.exists():
            tok_path = output_dir / "tokenizer.json"
        tok = SimpleTokenizer.load(str(tok_path))
        vocab_size = tok.vocab_size
    else:
        tok_dir = model_dir / "text_tokenizer"
        if not tok_dir.exists():
            tok_dir = output_dir / "text_tokenizer"
        tok = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True, use_fast=True)
        vocab_size = 0

    # Build model (without loading pretrained backbone - we have trained weights)
    model = build_model_from_config(
        vocab_size=vocab_size,
        config=cfg,
        image_backbone_pretrained=False,
        text_backbone_pretrained=False,  # Don't load HF weights, we have our own
        text_local_files_only=True,
    ).to(device)
    
    # Load trained weights (strict=False for flexibility)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    max_text_len = int(cfg.get("text_max_len", 32))
    image_transform = build_image_transform(image_size=int(cfg["image_size"]))

    # Load data entries
    all_entries = load_image_caption_entries(csv_path, image_root=image_root)
    set_seed(42)
    train_e, val_e, test_e = split_entries(all_entries, train_ratio=0.8, val_ratio=0.1, seed=42)
    split_map = {"train": train_e, "val": val_e, "test": test_e, "all": all_entries}
    entries = split_map.get(split, test_e)

    image_paths, class_names, captions, cap2img, img2caps = build_retrieval_benchmark(entries)

    # Build embeddings with parallel loading
    keep_on_gpu = (device_str == "cuda")
    img_embeds = encode_image_paths_parallel(model, image_paths, image_transform, batch_size, device, keep_on_gpu=keep_on_gpu)
    txt_embeds = encode_texts(model, tok, captions, batch_size, device, max_text_len, use_scratch, keep_on_gpu=keep_on_gpu)

    metrics = compute_retrieval_metrics(img_embeds, txt_embeds, cap2img, img2caps, ks=(1, 5, 10))

    # Load training history and config
    history = load_json_safe(output_dir / "history.json")
    best_summary = load_json_safe(output_dir / "best_summary.json")
    test_metrics = load_json_safe(output_dir / "test_metrics.json")
    model_config = load_json_safe(output_dir / "model_config.json")
    
    # Get plot paths
    plots_dir = output_dir / "plots"
    plots = {}
    for plot_name in ["training_curves.png", "qualitative_text_to_image.png", 
                      "qualitative_image_to_text.png", "test_embedding_projection.png"]:
        plot_path = plots_dir / plot_name
        if plot_path.exists():
            plots[plot_name.replace(".png", "")] = str(plot_path)

    return {
        "model": model,
        "tokenizer": tok,
        "use_scratch": use_scratch,
        "max_text_len": max_text_len,
        "image_transform": image_transform,
        "device": device,
        "keep_on_gpu": keep_on_gpu,
        "image_paths": image_paths,
        "class_names": class_names,
        "captions": captions,
        "cap2img": cap2img,
        "img2caps": img2caps,
        "img_embeds": img_embeds,
        "txt_embeds": txt_embeds,
        "metrics": metrics,
        "history": history,
        "best_summary": best_summary,
        "test_metrics": test_metrics,
        "model_config": model_config,
        "plots": plots,
        "output_dir": str(output_dir),
    }


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
def render_header():
    """Render beautiful header."""
    st.markdown("""
    <div class="main-header">
        <h1>🛰️ CLIP Zero-Shot Retrieval Explorer</h1>
        <p>Interactive exploration of CLIP models trained on satellite imagery</p>
    </div>
    """, unsafe_allow_html=True)


def render_training_plots(ctx: dict, setup_name: str):
    """Render training visualization plots."""
    plots = ctx.get("plots", {})
    history = ctx.get("history", {})
    
    if not plots and not history:
        st.info("📊 No training plots available. Train the model first to see visualizations.")
        return
    
    # Training curves
    if "training_curves" in plots:
        st.markdown("### 📈 Training Curves")
        st.image(plots["training_curves"], use_column_width=True)
    elif history:
        st.markdown("### 📈 Training Progress")
        import matplotlib.pyplot as plt
        
        plt.style.use('seaborn-v0_8-whitegrid')
        epochs = history.get("epoch", [])
        train_loss = history.get("train_loss", [])
        val_t2i = history.get("val_t2i_r@1", [])
        val_i2t = history.get("val_i2t_r@1", [])
        
        if epochs and train_loss:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor('#ffffff')
            
            for ax in axes:
                ax.set_facecolor('#f8f9fa')
                ax.tick_params(colors='#424242')
                ax.xaxis.label.set_color('#424242')
                ax.yaxis.label.set_color('#424242')
                ax.title.set_color('#37474f')
            
            # Loss plot
            axes[0].plot(epochs, train_loss, color='#5c6bc0', linewidth=2.5, label='Train Loss')
            axes[0].fill_between(epochs, train_loss, alpha=0.2, color='#5c6bc0')
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.4, color='#bdbdbd')
            axes[0].legend(facecolor='#ffffff', edgecolor='#e0e0e0')
            
            # Retrieval metrics plot
            if val_t2i:
                axes[1].plot(epochs, [v*100 for v in val_t2i], color='#26a69a', linewidth=2.5, label='T2I R@1')
            if val_i2t:
                axes[1].plot(epochs, [v*100 for v in val_i2t], color='#ef5350', linewidth=2.5, label='I2T R@1')
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Recall@1 (%)', fontsize=12)
            axes[1].set_title('Validation Retrieval Metrics', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.4, color='#bdbdbd')
            axes[1].legend(facecolor='#ffffff', edgecolor='#e0e0e0')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # Qualitative results in expanders
    col1, col2 = st.columns(2)
    
    with col1:
        if "qualitative_text_to_image" in plots:
            with st.expander("🔤→🖼️ Text to Image Examples", expanded=True):
                st.image(plots["qualitative_text_to_image"], use_column_width=True)
    
    with col2:
        if "qualitative_image_to_text" in plots:
            with st.expander("🖼️→🔤 Image to Text Examples", expanded=True):
                st.image(plots["qualitative_image_to_text"], use_column_width=True)
    
    # Embedding projection
    if "test_embedding_projection" in plots:
        st.markdown("### 🎯 Embedding Space Visualization")
        st.image(plots["test_embedding_projection"], use_column_width=True)
        st.caption("t-SNE projection of image and text embeddings. Aligned clusters indicate good multimodal learning.")


def render_metrics_dashboard(ctx: dict, setup_name: str, key_prefix: str):
    """Render metrics dashboard."""
    metrics = ctx.get("metrics", {})
    test_metrics = ctx.get("test_metrics", {})
    best_summary = ctx.get("best_summary", {})
    model_config = ctx.get("model_config", {})
    
    # Best epoch badge
    if best_summary:
        best_epoch = best_summary.get('best_epoch', 'N/A')
        best_score = best_summary.get('best_score', 0)
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #e8eaf6, #ede7f6); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem;
                    border-left: 4px solid #5c6bc0;">
            <b style="color: #37474f;">🏆 Best Epoch:</b> {best_epoch} &nbsp;&nbsp;|&nbsp;&nbsp; 
            <b style="color: #37474f;">Combined Score (T2I+I2T R@1):</b> {best_score:.2%}
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics grid
    st.markdown("#### 📊 Retrieval Metrics")
    cols = st.columns(6)
    metric_order = ["t2i_r@1", "t2i_r@5", "t2i_r@10", "i2t_r@1", "i2t_r@5", "i2t_r@10"]
    display_names = ["T2I R@1", "T2I R@5", "T2I R@10", "I2T R@1", "I2T R@5", "I2T R@10"]
    
    for i, (key, name) in enumerate(zip(metric_order, display_names)):
        with cols[i]:
            value = metrics.get(key, test_metrics.get(key, 0))
            st.metric(label=name, value=f"{value:.1%}")
    
    # Model configuration cards
    st.markdown("#### ⚙️ Model Configuration")
    config_cols = st.columns(4)
    
    configs = [
        ("🖼️ Image Encoder", model_config.get("image_encoder_name", "N/A"), 
         "✅ Pretrained" if model_config.get("image_backbone_pretrained") else "🔧 Scratch"),
        ("📝 Text Encoder", model_config.get("text_encoder_name", "N/A"),
         "✅ Pretrained" if model_config.get("text_backbone_pretrained") else "🔧 Scratch"),
        ("📐 Embed Dim", str(model_config.get("embed_dim", "N/A")), "dimensions"),
        ("🖼️ Image Size", f"{model_config.get('image_size', 'N/A')}px", "input resolution"),
    ]
    
    for col, (title, value, subtitle) in zip(config_cols, configs):
        with col:
            st.markdown(f"""
            <div style="background: #f5f7fa; padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0;">
                <div style="font-size: 0.85rem; color: #757575;">{title}</div>
                <div style="font-size: 1.1rem; font-weight: bold; margin: 0.3rem 0; color: #37474f;">{value}</div>
                <div style="font-size: 0.75rem; color: #9e9e9e;">{subtitle}</div>
            </div>
            """, unsafe_allow_html=True)


def render_retrieval_interface(ctx: dict, top_k: int, key_prefix: str):
    """Render interactive retrieval interface with unique keys per model."""
    tab_t2i, tab_i2t, tab_browse = st.tabs([
        "📝 Text → Image",
        "🖼️ Image → Text", 
        "📂 Browse Index",
    ])

    # ================================================================
    # TAB 1: Text-to-Image
    # ================================================================
    with tab_t2i:
        st.markdown("### 🔍 Find images matching your description")
        
        # Query input with unique key
        query_text = st.text_input(
            "Enter text query:", 
            key=f"{key_prefix}_t2i_query",
            placeholder="e.g., green agricultural field with crops, urban area with buildings..."
        )
        
        # Example buttons
        st.markdown("**Quick examples:**")
        example_cols = st.columns(5)
        examples = [
            ("🌾 Agriculture", "green agricultural field with crops"),
            ("🏙️ Urban", "urban buildings and roads"),
            ("🌲 Forest", "dense forest with trees"),
            ("💧 Water", "water body like river or lake"),
            ("🏜️ Barren", "barren land with no vegetation"),
        ]
        
        for col, (label, query) in zip(example_cols, examples):
            with col:
                if st.button(label, key=f"{key_prefix}_ex_{label}"):
                    query_text = query

        if query_text:
            with st.spinner("🔍 Searching..."):
                feat = embed_single_text(
                    ctx["model"], ctx["tokenizer"], query_text,
                    ctx["device"], ctx["max_text_len"], ctx["use_scratch"],
                    keep_on_gpu=ctx["keep_on_gpu"],
                )
                sims = (feat @ ctx["img_embeds"].T).squeeze(0)
                k = min(top_k, len(ctx["image_paths"]))
                topk = torch.topk(sims, k=k)
                indices = topk.indices.cpu().numpy()
                scores = topk.values.cpu().numpy()

            st.success(f"✨ Found top {k} matches for: *\"{query_text}\"*")
            
            # Results grid
            for row_start in range(0, k, 5):
                cols = st.columns(min(5, k - row_start))
                for i, col in enumerate(cols):
                    idx = row_start + i
                    if idx < k:
                        with col:
                            img = load_rgb_image(ctx["image_paths"][int(indices[idx])])
                            st.image(img, use_column_width=True)
                            st.markdown(f"""
                            <div style="text-align: center; font-size: 0.85rem;">
                                <b>#{idx+1}</b> | Sim: {scores[idx]:.3f}<br>
                                <span style="color: #00897b;">{ctx['class_names'][int(indices[idx])]}</span>
                            </div>
                            """, unsafe_allow_html=True)

    # ================================================================
    # TAB 2: Image-to-Text
    # ================================================================
    with tab_i2t:
        st.markdown("### 🔍 Find captions matching your image")
        
        i2t_mode = st.radio(
            "Image source:", 
            ["📤 Upload image", "📁 Pick from index"], 
            horizontal=True, 
            key=f"{key_prefix}_i2t_mode"
        )

        query_image_pil = None
        query_image_path = None

        if i2t_mode == "📤 Upload image":
            uploaded = st.file_uploader(
                "Upload an image:", 
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key=f"{key_prefix}_uploader"
            )
            if uploaded is not None:
                query_image_pil = Image.open(uploaded).convert("RGB")
        else:
            col_picker, col_preview = st.columns([1, 2])
            with col_picker:
                idx_pick = st.number_input(
                    f"Image index (0–{len(ctx['image_paths'])-1}):",
                    min_value=0, 
                    max_value=len(ctx["image_paths"]) - 1, 
                    value=0, 
                    step=1,
                    key=f"{key_prefix}_img_idx"
                )
                st.info(f"**Class:** {ctx['class_names'][idx_pick]}")
            
            query_image_path = ctx["image_paths"][idx_pick]
            query_image_pil = load_rgb_image(query_image_path)

        if query_image_pil is not None:
            col_img, col_results = st.columns([1, 2])
            
            with col_img:
                st.image(query_image_pil, caption="Query Image", use_column_width=True)

            with col_results:
                with st.spinner("🔍 Searching..."):
                    if query_image_path:
                        feat = embed_single_image(
                            ctx["model"], ctx["image_transform"], 
                            query_image_path, ctx["device"],
                            keep_on_gpu=ctx["keep_on_gpu"]
                        )
                    else:
                        feat = embed_single_image(
                            ctx["model"], ctx["image_transform"], 
                            query_image_pil, ctx["device"],
                            keep_on_gpu=ctx["keep_on_gpu"]
                        )
                    
                    sims = (feat @ ctx["txt_embeds"].T).squeeze(0)
                    k = min(top_k, len(ctx["captions"]))
                    topk = torch.topk(sims, k=k)
                    indices = topk.indices.cpu().numpy()
                    scores = topk.values.cpu().numpy()

                st.markdown("**🏆 Top retrieved captions:**")
                for rank, (cidx, score) in enumerate(zip(indices, scores)):
                    caption = ctx["captions"][int(cidx)]
                    gt_img_idx = ctx["cap2img"][int(cidx)]
                    gt_class = ctx["class_names"][gt_img_idx]
                    
                    st.markdown(f"""
                    <div style="background: #f5f7fa; padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid #e0e0e0;">
                        <b style="color: #37474f;">#{rank+1}</b> (sim={score:.3f}) 
                        <span style="background: #e8f5e9; color: #2e7d32; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">{gt_class}</span>
                        <p style="margin: 0.5rem 0 0 0; color: #424242;">{caption}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # ================================================================
    # TAB 3: Browse Index
    # ================================================================
    with tab_browse:
        st.markdown("### 📂 Browse the indexed dataset")

        unique_classes = sorted(set(ctx["class_names"]))
        selected_class = st.selectbox(
            "Filter by class:", 
            ["(all)"] + unique_classes,
            key=f"{key_prefix}_class_filter"
        )

        if selected_class == "(all)":
            filtered = list(range(len(ctx["image_paths"])))
        else:
            filtered = [i for i, c in enumerate(ctx["class_names"]) if c == selected_class]

        st.info(f"📊 Showing **{len(filtered)}** images | Total: {len(ctx['image_paths'])} images, {len(ctx['captions'])} captions")

        page_size = 12
        total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
        page = st.number_input(
            "Page", 
            min_value=1, 
            max_value=total_pages, 
            value=1, 
            step=1,
            key=f"{key_prefix}_page"
        )
        page_items = filtered[(page - 1) * page_size : page * page_size]

        cols = st.columns(4)
        for i, img_idx in enumerate(page_items):
            with cols[i % 4]:
                img = load_rgb_image(ctx["image_paths"][img_idx])
                cap_idxs = ctx["img2caps"][img_idx]
                first_cap = ctx["captions"][cap_idxs[0]] if cap_idxs else ""
                short_cap = textwrap.shorten(first_cap, width=50, placeholder="…")
                st.image(img, use_column_width=True)
                st.caption(f"**{ctx['class_names'][img_idx]}**\n{short_cap}")


def render_comparison_dashboard():
    """Render comparison across all model setups."""
    st.markdown("## 📊 Model Comparison Dashboard")
    
    # Collect metrics from all available models
    all_metrics = {}
    
    for setup_name, config in MODEL_CONFIGS.items():
        output_dir = Path(config["dir"]).parent
        test_metrics = load_json_safe(output_dir / "test_metrics.json")
        best_summary = load_json_safe(output_dir / "best_summary.json")
        history = load_json_safe(output_dir / "history.json")
        
        if test_metrics or best_summary:
            all_metrics[setup_name] = {
                "test_metrics": test_metrics,
                "best_summary": best_summary,
                "history": history,
                "icon": config["icon"],
                "color": config["color"],
                "available": check_model_available(config["dir"]),
            }
    
    if not all_metrics:
        st.warning("⚠️ No trained models found. Train at least one model configuration first.")
        return
    
    # Comparison table
    st.markdown("### 📋 Performance Comparison")
    
    import pandas as pd
    
    comparison_data = []
    for setup_name, data in all_metrics.items():
        if data["available"]:
            metrics = data["test_metrics"]
            best = data["best_summary"]
            comparison_data.append({
                "Model": f"{data['icon']} {setup_name}",
                "T2I R@1": f"{metrics.get('t2i_r@1', 0):.1%}",
                "T2I R@5": f"{metrics.get('t2i_r@5', 0):.1%}",
                "I2T R@1": f"{metrics.get('i2t_r@1', 0):.1%}",
                "I2T R@5": f"{metrics.get('i2t_r@5', 0):.1%}",
                "Best Epoch": best.get("best_epoch", "N/A"),
                "Score": f"{best.get('best_score', 0):.1%}",
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Visual comparison charts
    st.markdown("### 📈 Visual Comparison")
    
    import matplotlib.pyplot as plt
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    setups = []
    t2i_r1, t2i_r5, i2t_r1, i2t_r5 = [], [], [], []
    colors = []
    
    for setup_name, data in all_metrics.items():
        if data["available"] and data["test_metrics"]:
            setups.append(setup_name.replace(" ", "\n"))
            t2i_r1.append(data["test_metrics"].get("t2i_r@1", 0) * 100)
            t2i_r5.append(data["test_metrics"].get("t2i_r@5", 0) * 100)
            i2t_r1.append(data["test_metrics"].get("i2t_r@1", 0) * 100)
            i2t_r5.append(data["test_metrics"].get("i2t_r@5", 0) * 100)
            colors.append(data["color"])
    
    if setups:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#ffffff')
        
        for ax in axes:
            ax.set_facecolor('#f8f9fa')
            ax.tick_params(colors='#424242')
            ax.xaxis.label.set_color('#424242')
            ax.yaxis.label.set_color('#424242')
            ax.title.set_color('#37474f')
        
        x = np.arange(len(setups))
        width = 0.35
        
        # R@1 comparison
        bars1 = axes[0].bar(x - width/2, t2i_r1, width, label='Text→Image', color='#26a69a', alpha=0.9)
        bars2 = axes[0].bar(x + width/2, i2t_r1, width, label='Image→Text', color='#ef5350', alpha=0.9)
        axes[0].set_ylabel('Recall@1 (%)', fontsize=12)
        axes[0].set_title('Recall@1 Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(setups, fontsize=9)
        axes[0].legend(facecolor='#ffffff', edgecolor='#e0e0e0')
        axes[0].grid(True, alpha=0.4, axis='y', color='#bdbdbd')
        
        # Add value labels
        for bar in bars1:
            h = bar.get_height()
            axes[0].annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, color='#424242')
        for bar in bars2:
            h = bar.get_height()
            axes[0].annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, color='#424242')
        
        # R@5 comparison
        bars3 = axes[1].bar(x - width/2, t2i_r5, width, label='Text→Image', color='#42a5f5', alpha=0.9)
        bars4 = axes[1].bar(x + width/2, i2t_r5, width, label='Image→Text', color='#66bb6a', alpha=0.9)
        axes[1].set_ylabel('Recall@5 (%)', fontsize=12)
        axes[1].set_title('Recall@5 Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(setups, fontsize=9)
        axes[1].legend(facecolor='#ffffff', edgecolor='#e0e0e0')
        axes[1].grid(True, alpha=0.4, axis='y', color='#bdbdbd')
        
        for bar in bars3:
            h = bar.get_height()
            axes[1].annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, color='#424242')
        for bar in bars4:
            h = bar.get_height()
            axes[1].annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, color='#424242')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Training curves comparison
    histories_available = [(name, data) for name, data in all_metrics.items() 
                           if data.get("history") and data["history"].get("train_loss")]
    
    if histories_available:
        st.markdown("### 📉 Training Curves Comparison")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#ffffff')
        
        for ax in axes:
            ax.set_facecolor('#f8f9fa')
            ax.tick_params(colors='#424242')
            ax.xaxis.label.set_color('#424242')
            ax.yaxis.label.set_color('#424242')
            ax.title.set_color('#37474f')
        
        color_map = {'Full Scratch': '#d32f2f', 'Full Pretrained': '#00897b', 
                     'Pretrained ViT Only': '#1976d2', 'Pretrained Text Only': '#7cb342'}
        
        for name, data in histories_available:
            history = data["history"]
            epochs = history.get("epoch", [])
            train_loss = history.get("train_loss", [])
            val_t2i = history.get("val_t2i_r@1", [])
            color = color_map.get(name, '#5c6bc0')
            
            if epochs and train_loss:
                axes[0].plot(epochs, train_loss, '-', linewidth=2.5, label=name, color=color)
            if epochs and val_t2i:
                axes[1].plot(epochs, [v * 100 for v in val_t2i], '-', linewidth=2.5, label=name, color=color)
        
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Training Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].legend(facecolor='#ffffff', edgecolor='#e0e0e0', fontsize=9)
        axes[0].grid(True, alpha=0.4, color='#bdbdbd')
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Val T2I R@1 (%)', fontsize=12)
        axes[1].set_title('Validation Performance', fontsize=14, fontweight='bold')
        axes[1].legend(facecolor='#ffffff', edgecolor='#e0e0e0', fontsize=9)
        axes[1].grid(True, alpha=0.4, color='#bdbdbd')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Best model highlight
    if comparison_data:
        best_combined = 0
        best_model = None
        for setup_name, data in all_metrics.items():
            if data["available"] and data["best_summary"]:
                score = data["best_summary"].get("best_score", 0)
                if score > best_combined:
                    best_combined = score
                    best_model = setup_name
        
        if best_model:
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #e8f5e9, #e3f2fd); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; margin-top: 1rem;
                        border: 2px solid #81c784;">
                <h3 style="margin: 0; color: #2e7d32;">🏆 Best Model: {MODEL_CONFIGS[best_model]['icon']} {best_model}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #388e3c;">
                    Combined Score: <b>{best_combined:.1%}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="CLIP Retrieval Explorer", 
        layout="wide", 
        page_icon="🛰️",
        initial_sidebar_state="expanded",
    )
    
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    render_header()
    
    # ---- Sidebar Settings ----
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        csv_path = st.text_input("📄 CSV Path", value=DEFAULT_CSV)
        image_root = st.text_input("📁 Image Root", value=DEFAULT_IMAGE_ROOT)
        split = st.selectbox("📊 Data Split", ["test", "val", "train", "all"], index=0)
        top_k = st.slider("🔝 Top-K Results", min_value=1, max_value=20, value=5)
        batch_size = st.number_input("📦 Batch Size", min_value=8, max_value=512, value=128)
        
        # Device (default GPU)
        device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        device_str = st.selectbox(
            "🖥️ Device", 
            device_options, 
            index=0,
            help="GPU (cuda) recommended for fast inference"
        )
        
        st.markdown("---")
        
        # Model status
        st.markdown("### 📁 Model Status")
        for setup_name, config in MODEL_CONFIGS.items():
            available = check_model_available(config["dir"])
            status_class = "status-ready" if available else "status-pending"
            status_text = "Ready" if available else "Not trained"
            st.markdown(f"""
            {config['icon']} **{setup_name}**
            <span class="{status_class}">{status_text}</span>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("💡 Select a model tab to explore")
    
    # ---- Main tabs ----
    setup_tabs = st.tabs([
        f"{config['icon']} {name}" 
        for name, config in MODEL_CONFIGS.items()
    ] + ["📊 Compare All"])
    
    # Render each model tab
    for i, (setup_name, config) in enumerate(MODEL_CONFIGS.items()):
        with setup_tabs[i]:
            # Create unique key prefix for this tab
            key_prefix = setup_name.replace(" ", "_").lower()
            
            st.markdown(f"## {config['icon']} {setup_name}")
            st.markdown(f"*{config['description']}*")
            
            model_dir = config["dir"]
            
            if not check_model_available(model_dir):
                st.warning(f"⚠️ Model not found at `{model_dir}`")
                
                with st.expander("📝 Training Command", expanded=True):
                    if setup_name == "Full Scratch":
                        flags = "--text_encoder_name scratch"
                    elif setup_name == "Full Pretrained":
                        flags = "--image_backbone_pretrained --text_encoder_name BAAI/bge-base-en-v1.5 --text_backbone_pretrained --text_backbone_trainable"
                    elif setup_name == "Pretrained ViT Only":
                        flags = "--image_backbone_pretrained --text_encoder_name scratch"
                    else:
                        flags = "--text_encoder_name BAAI/bge-base-en-v1.5 --text_backbone_pretrained --text_backbone_trainable"
                    
                    st.code(f"""CUDA_VISIBLE_DEVICES=0 python train_clip_scratch.py \\
    --csv_path {DEFAULT_CSV} \\
    --image_root {DEFAULT_IMAGE_ROOT} \\
    --output_dir {config['dir'].replace('/best_model', '')} \\
    --image_encoder_name vit_b_16 {flags} \\
    --epochs 20 --batch_size 64 --amp""", language="bash")
                continue
            
            # Load model with progress
            with st.spinner(f"⏳ Loading {setup_name}..."):
                ctx = load_model_context(
                    model_dir=model_dir,
                    csv_path=csv_path,
                    image_root=image_root if image_root.strip() else None,
                    split=split,
                    batch_size=batch_size,
                    device_str=device_str,
                )
            
            if ctx is None:
                st.error("❌ Failed to load model.")
                continue
            
            # Quick stats row
            stat_cols = st.columns(4)
            with stat_cols[0]:
                t2i = ctx["metrics"].get("t2i_r@1", ctx["test_metrics"].get("t2i_r@1", 0))
                st.metric("🎯 T2I R@1", f"{t2i:.1%}")
            with stat_cols[1]:
                i2t = ctx["metrics"].get("i2t_r@1", ctx["test_metrics"].get("i2t_r@1", 0))
                st.metric("🎯 I2T R@1", f"{i2t:.1%}")
            with stat_cols[2]:
                st.metric("🖼️ Images", len(ctx["image_paths"]))
            with stat_cols[3]:
                st.metric("📝 Captions", len(ctx["captions"]))
            
            st.markdown("---")
            
            # Inner tabs with unique keys
            inner_tabs = st.tabs([
                "🔍 Interactive Retrieval",
                "📈 Training Plots",
                "📊 Detailed Metrics",
            ])
            
            with inner_tabs[0]:
                render_retrieval_interface(ctx, top_k, key_prefix)
            
            with inner_tabs[1]:
                render_training_plots(ctx, setup_name)
            
            with inner_tabs[2]:
                render_metrics_dashboard(ctx, setup_name, key_prefix)
    
    # Comparison tab
    with setup_tabs[-1]:
        render_comparison_dashboard()
    
    # Footer
    st.markdown("""
    <div class="footer">
        🛰️ CLIP for Satellite Imagery &nbsp;|&nbsp; 
        🚀 GPU Accelerated &nbsp;|&nbsp; 
        ⚡ {workers} Worker Threads &nbsp;|&nbsp;
        Built with Streamlit
    </div>
    """.format(workers=NUM_WORKERS), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
