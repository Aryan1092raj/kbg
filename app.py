"""
app.py – Streamlit demo for BoneVision Fracture Classifier.

Run:
    streamlit run app.py

If you trained on Colab, download the checkpoint first:
    - From Google Drive → My Drive/KBG_Results/checkpoints/best_model.pth
    - Place it at: checkpoints/best_model.pth  (relative to this file)
"""

import os
import sys
import io
import re
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Make sure local imports work ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import FractureClassifier, SoftVotingEnsemble, GradCAMWrapper

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BoneVision – Fracture Type Classifier",
    page_icon="🦴",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants — loaded from config.yaml so they stay in sync with training
# ─────────────────────────────────────────────────────────────────────────────

def _load_class_names(cfg_name="config.yaml"):
    """Load class names from config file; fall back to defaults if missing."""
    cfg_path = os.path.join(SCRIPT_DIR, cfg_name)
    default_classes = ["fractured", "not fractured"] if "binary" in cfg_name else [
        "Avulsion fracture", "Comminuted fracture",
        "Compression-Crush fracture", "Fracture Dislocation",
        "Greenstick fracture", "Hairline Fracture",
        "Impacted fracture", "Intra-articular fracture",
        "Longitudinal fracture", "Oblique fracture",
        "Pathological fracture", "Spiral Fracture",
    ]
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("data", {}).get("classes", default_classes)
    return default_classes


CLASSES_BINARY = _load_class_names("config.yaml")
CLASSES_MULTI = _load_class_names("config_multiclass.yaml")
NUM_CLASSES_BINARY = len(CLASSES_BINARY)
NUM_CLASSES_MULTI = len(CLASSES_MULTI)
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_inference_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def render_image(img, caption: str):
    """Render image with Streamlit API compatibility across versions."""
    try:
        # Newer Streamlit versions (recommended API)
        st.image(img, caption=caption, width="stretch")
        return
    except TypeError:
        pass

    try:
        # Older compatibility path
        st.image(img, caption=caption, use_column_width=True)
        return
    except TypeError:
        # Final fallback for very old/new API changes
        st.image(img, caption=caption)


def _extract_state_dict(ckpt: dict):
    if "model_state" in ckpt:
        return ckpt["model_state"]
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _get_cfg_ensemble_defaults(n_models: int, is_multiclass: bool = False):
    cfg_file = "config_multiclass.yaml" if is_multiclass else "config.yaml"
    cfg_path = os.path.join(SCRIPT_DIR, cfg_file)
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        arch_names = cfg["model"]["ensemble"]["models"][:n_models]
        weights = cfg["model"]["ensemble"]["weights"][:n_models]
        return arch_names, weights
    return ["vit_small_patch16_224", "efficientnet_b0", "convnext_tiny"][:n_models], [0.5, 0.25, 0.25][:n_models]


def _infer_arch_for_submodel(state_dict: dict, model_idx: int) -> str:
    prefix = f"models.{model_idx}.backbone."
    subkeys = [k for k in state_dict.keys() if k.startswith(prefix)]
    if not subkeys:
        return "vit_small_patch16_224"

    # ViT
    vit_probe = f"{prefix}cls_token"
    if vit_probe in state_dict:
        embed_dim = int(state_dict[vit_probe].shape[-1])
        if embed_dim == 768:
            return "vit_base_patch16_224"
        if embed_dim == 384:
            return "vit_small_patch16_224"

    # EfficientNet family
    eff_probe = f"{prefix}conv_stem.weight"
    if eff_probe in state_dict:
        stem_channels = int(state_dict[eff_probe].shape[0])
        if stem_channels == 40:
            return "efficientnet_b3"
        if stem_channels == 32:
            return "efficientnet_b0"

    # ConvNeXt family
    convnext_probe = f"{prefix}stem.0.weight"
    if convnext_probe in state_dict:
        stage2_blocks = []
        for k in subkeys:
            m = re.search(r"\.stages\.2\.blocks\.(\d+)\.", k)
            if m:
                stage2_blocks.append(int(m.group(1)))
        if stage2_blocks and max(stage2_blocks) >= 26:
            return "convnext_small"
        return "convnext_tiny"

    return "vit_small_patch16_224"


def _infer_single_arch_from_state_dict(state_dict: dict) -> str:
    if "backbone.cls_token" in state_dict:
        embed_dim = int(state_dict["backbone.cls_token"].shape[-1])
        if embed_dim == 768:
            return "vit_base_patch16_224"
        if embed_dim == 384:
            return "vit_small_patch16_224"
    if "backbone.conv_stem.weight" in state_dict:
        stem_channels = int(state_dict["backbone.conv_stem.weight"].shape[0])
        return "efficientnet_b3" if stem_channels == 40 else "efficientnet_b0"
    if "backbone.stem.0.weight" in state_dict:
        stage2_blocks = []
        for k in state_dict.keys():
            m = re.search(r"^backbone\.stages\.2\.blocks\.(\d+)\.", k)
            if m:
                stage2_blocks.append(int(m.group(1)))
        if stage2_blocks and max(stage2_blocks) >= 26:
            return "convnext_small"
        return "convnext_tiny"
    return "vit_small_patch16_224"


def _infer_num_classes_from_state_dict(state_dict: dict, default_num_classes: int) -> int:
    """Try to infer num_classes from the final classifier output layer in checkpoint.

    The classifier is typically a Sequential with intermediate hidden layers
    (e.g. classifier.0, classifier.1) followed by the output projection
    (e.g. classifier.2).  We must use the *highest-indexed* weight tensor,
    whose shape[0] equals num_classes — NOT the first one found, which may be
    a hidden layer with shape[0] == hidden_dim.
    """
    best_layer_idx = -1
    best_tensor = None

    for k, v in state_dict.items():
        if "classifier" not in k or "weight" not in k or v.ndim != 2:
            continue
        # Extract the numeric layer index from the key.
        # e.g. "models.0.classifier.2.weight"  →  parts after "classifier." → "2"
        # e.g. "classifier.2.weight"            →  same result
        after_cls = k[k.index("classifier") + len("classifier."):]
        parts = after_cls.split(".")
        try:
            layer_idx = int(parts[0])
        except (ValueError, IndexError):
            layer_idx = 0

        if layer_idx > best_layer_idx:
            best_layer_idx = layer_idx
            best_tensor = v

    if best_tensor is not None:
        return best_tensor.shape[0]

    # Fallback: bias vectors also encode num_classes
    for k, v in state_dict.items():
        if "classifier" in k and "bias" in k and v.ndim == 1:
            return v.shape[0]

    return default_num_classes


@st.cache_resource(show_spinner="Loading model…")
def load_model(checkpoint_path: str, is_multiclass: bool = False):
    """Load the trained model from a checkpoint."""
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = _extract_state_dict(ckpt)
    
    # Infer num_classes from checkpoint
    default_classes = NUM_CLASSES_MULTI if is_multiclass else NUM_CLASSES_BINARY
    num_classes = _infer_num_classes_from_state_dict(state_dict, default_classes)

    # Check if ensemble (has keys like "models.0.backbone…") or single model
    is_ensemble = any(k.startswith("models.") for k in state_dict.keys())

    if is_ensemble:
        # Discover how many sub-models
        model_indices = set()
        for k in state_dict.keys():
            if k.startswith("models."):
                idx = int(k.split(".")[1])
                model_indices.add(idx)
        ordered_indices = sorted(model_indices)
        n_models = len(ordered_indices)

        arch_names, weights = _get_cfg_ensemble_defaults(n_models, is_multiclass)
        inferred_arch_names = [
            _infer_arch_for_submodel(state_dict, idx) for idx in ordered_indices
        ]

        # If config does not match checkpoint tensor shapes, prioritize inferred arch.
        if arch_names != inferred_arch_names:
            arch_names = inferred_arch_names

        models = [
            FractureClassifier(name, num_classes=num_classes, pretrained=False)
            for name in arch_names
        ]
        model = SoftVotingEnsemble(models, weights=weights)
    else:
        # Single model — try config first, then infer from checkpoint structure
        arch = None
        cfg_file = "config_multiclass.yaml" if is_multiclass else "config.yaml"
        cfg_path = os.path.join(SCRIPT_DIR, cfg_file)
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            arch = cfg["model"]["architecture"]

        inferred_arch = _infer_single_arch_from_state_dict(state_dict)
        if arch is None or arch != inferred_arch:
            arch = inferred_arch

        model = FractureClassifier(arch, num_classes=num_classes, pretrained=False)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, device


def get_gradcam_target_layer(model):
    """Pick the best target layer for GradCAM based on model type."""
    if isinstance(model, SoftVotingEnsemble):
        # Use first model in ensemble
        inner = model.models[0]
    else:
        inner = model

    backbone = inner.backbone

    # Try common layer names
    for attr in ["norm", "layer4", "features", "stages"]:
        if hasattr(backbone, attr):
            layer = getattr(backbone, attr)
            if isinstance(layer, (torch.nn.Sequential, torch.nn.ModuleList)):
                return layer[-1]
            return layer

    # Fallback — last module
    children = list(backbone.children())
    return children[-1] if children else backbone


def generate_gradcam(model, image_tensor, device, class_idx):
    """Generate GradCAM heatmap. Returns heatmap as numpy array (H, W)."""
    target_layer = get_gradcam_target_layer(model)
    cam = GradCAMWrapper(model, target_layer)
    try:
        heatmap = cam(image_tensor.unsqueeze(0).to(device), class_idx=class_idx)
        if heatmap.ndim == 1:
            # ViT: reshape token sequence to 2D (exclude CLS token)
            n_tokens = heatmap.shape[0]
            if n_tokens == 197:  # 14*14 + 1 (CLS)
                heatmap = heatmap[1:].reshape(14, 14)
            else:
                side = int(np.sqrt(n_tokens))
                if side * side == n_tokens:
                    heatmap = heatmap.reshape(side, side)
                else:
                    heatmap = heatmap[1:n_tokens]
                    side = int(np.sqrt(n_tokens - 1))
                    heatmap = heatmap[:side * side].reshape(side, side)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        return heatmap
    except Exception as e:
        st.warning(f"GradCAM failed: {e}")
        return None


def overlay_heatmap(original_img, heatmap, alpha=0.5):
    """Overlay GradCAM heatmap on original image."""
    img = np.array(original_img.resize((IMG_SIZE, IMG_SIZE)))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def predict(model, image: Image.Image, device):
    """Run inference on a PIL Image, return class probs."""
    transform = build_inference_transform()
    img_np = np.array(image.convert("RGB"))
    transformed = transform(image=img_np)
    tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

    return probs, tensor.squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h1>🦴 BoneVision – Fracture Type Classifier</h1>
            <p style='font-size: 1.05rem; color: grey; margin-top: -0.6rem;'>
                Upload a bone X-ray to identify the fracture type from 12 categories using AI-powered classification with GradCAM explainability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Settings")

        # Binary Checkpoint selector
        default_ckpt_binary = os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pth")
        ckpt_path_binary = st.text_input(
            "Detection Model (Binary)",
            value=default_ckpt_binary,
            help="Path to the binary fracture detection model (.pth)",
        )

        # Multi-class Checkpoint selector
        default_ckpt_multi = os.path.join(SCRIPT_DIR, "checkpoints", "best_model_multiclass.pth")
        ckpt_path_multi = st.text_input(
            "Classification Model (Multi-class)",
            value=default_ckpt_multi,
            help="Path to the multi-class fracture typing model (.pth)",
        )

        show_gradcam = st.checkbox("Show GradCAM", value=True)
        gradcam_alpha = st.slider("GradCAM overlay opacity", 0.1, 0.9, 0.5, 0.05)
        
        st.markdown("---")
        st.markdown("**Supported Fracture Types:**")
        for i, cls in enumerate(CLASSES_MULTI):
            st.markdown(f"{i+1}. {cls}")

    # ── Model loading ──
    if not os.path.exists(ckpt_path_binary):
        st.warning(f"⚠️ Binary model not found at `{ckpt_path_binary}`.")
        st.stop()
    if not os.path.exists(ckpt_path_multi):
        st.warning(f"⚠️ Multi-class model not found at `{ckpt_path_multi}`. It might still be training.")
        st.stop()

    model_binary, device = load_model(ckpt_path_binary, is_multiclass=False)
    model_multi, _ = load_model(ckpt_path_multi, is_multiclass=True)
    st.sidebar.success(f"Models ready on **{device}**")

    # ── Upload ──
    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.subheader("📤 Upload X-Ray")
        uploaded = st.file_uploader(
            "Choose an X-ray image",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Upload a bone X-ray image for fracture type classification",
        )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        with col_upload:
            render_image(image, caption="Input X-Ray")

        # ── Two-Stage Inference ──
        with st.spinner("🔍 Stage 1: Detecting fracture…"):
            probs_binary, img_tensor = predict(model_binary, image, device)

        pred_idx_binary = int(np.argmax(probs_binary))
        pred_class_binary = CLASSES_BINARY[pred_idx_binary]
        conf_binary = float(probs_binary[pred_idx_binary]) * 100

        with col_result:
            st.subheader("🩻 Prediction Result")

            if pred_class_binary != "fractured":
                st.success(f"### ✅ **{pred_class_binary.title()}** — {conf_binary:.1f}% confidence")
                st.info("No fracture detected. Skipping fracture type classification.")
                
                if show_gradcam:
                    st.markdown("---")
                    st.subheader("🔥 GradCAM — Detection Attention")
                    heatmap = generate_gradcam(model_binary, img_tensor, device, pred_idx_binary)
                    if heatmap is not None:
                        overlay = overlay_heatmap(image, heatmap, alpha=gradcam_alpha)
                        render_image(overlay, caption="GradCAM Overlay")
            
            else:
                st.error(f"### ⚠️ **Fracture Detected** — {conf_binary:.1f}% confidence")
                
                with st.spinner("🔍 Stage 2: Classifying fracture type…"):
                    probs_multi, _ = predict(model_multi, image, device)
                
                pred_idx_multi = int(np.argmax(probs_multi))
                pred_class_multi = CLASSES_MULTI[pred_idx_multi]
                conf_multi = float(probs_multi[pred_idx_multi]) * 100

                top3_indices = np.argsort(probs_multi)[::-1][:3]

                st.markdown("---")
                st.error(f"### 🔬 Type: **{pred_class_multi}** ({conf_multi:.1f}%)")

                # Top-3 predictions
                st.markdown("**Top 3 Possible Types:**")
                for rank, idx in enumerate(top3_indices, 1):
                    cls_name = CLASSES_MULTI[idx]
                    pct = float(probs_multi[idx]) * 100
                    emoji = "🥇" if rank == 1 else ("🥈" if rank == 2 else "🥉")
                    st.progress(float(probs_multi[idx]), text=f"{emoji} {cls_name}: {pct:.1f}%")

                # GradCAM
                if show_gradcam:
                    st.markdown("---")
                    st.subheader("🔥 GradCAM — Diagnostic Attention")
                    with st.spinner("Generating GradCAM…"):
                        heatmap = generate_gradcam(model_multi, img_tensor, device, pred_idx_multi)
                    if heatmap is not None:
                        overlay = overlay_heatmap(image, heatmap, alpha=gradcam_alpha)
                        cam_col1, cam_col2 = st.columns(2)
                        with cam_col1:
                            render_image(image.resize((IMG_SIZE, IMG_SIZE)), caption="Original")
                        with cam_col2:
                            render_image(overlay, caption="Classification GradCAM Overlay")
                        st.caption("GradCAM identifies the fracture location the model used to determine the *type*.")
                    else:
                        st.info("GradCAM visualization unavailable for this model architecture.")

    else:
        with col_result:
            st.info("👈 Upload an X-ray image to get started.")


if __name__ == "__main__":
    main()