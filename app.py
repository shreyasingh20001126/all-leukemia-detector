"""
ALL Detection Web App
Streamlit application — deploy on Hugging Face Spaces

Run locally:  streamlit run app.py
"""

import os
import json
import gc
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, ResNet152, MobileNet, EfficientNetV2S
from tensorflow.keras import layers, Model
import cv2
import sys

# Add project root to path
sys.path.append(os.path.dirname(__file__))
from utils.preprocessing import preprocess_image, get_intermediate_images

# ---------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="ALL Detection — Shreya Singh",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .result-all {
        background: #fce4ec;
        border-left: 5px solid #e53935;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-normal {
        background: #e8f5e9;
        border-left: 5px solid #43a047;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-box {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .disclaimer {
        background: #fff8e1;
        border: 1px solid #fdd835;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #5d4037;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Sidebar — about the project
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This app implements the model from:

    **Acute Lymphoblastic Leukemia Identification through Blood Smear Microscopic Images: 
    Integrating Pre-Trained CNNs with Weighted Average Voting**

    *Shreya Singh et al., IEEE DASA 2024*

    [📄 View Paper](https://doi.org/10.1109/DASA63652.2024.10836445)
    """)

    st.divider()
    st.markdown("### Model performance")
    perf_data = {
        "Model": ["ResNet-152", "EfficientNetV2", "AlexNet", "ResNet-50", "VGG-16", "MobileNet", "**Ensemble**"],
        "Accuracy": ["97.04%", "95.13%", "94.64%", "94.51%", "91.89%", "87.17%", "**97.54%**"],
        "F1 Score": ["96.22%", "94.78%", "94.55%", "94.51%", "91.08%", "87.17%", "**97.01%**"],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("### Dataset")
    st.markdown("""
    **C-NMC 2019** (ISBI Challenge)  
    - 10,661 blood cell images  
    - 69% blast cells (ALL)  
    - 31% normal cells (HEM)  
    - Source: AIIMS New Delhi
    """)

    st.divider()
    st.markdown("### Preprocessing pipeline")
    st.markdown("""
    1. Black background removal (grayscale threshold)
    2. Center crop with square padding
    3. CLAHE on Y channel (YUV space)
    4. Resize to 224×224
    """)


# ---------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------
def add_head(base, name, dropout=0.5):
    base.trainable = False
    x = layers.GlobalAveragePooling2D(name=f"{name}_gap")(base.output)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    out = layers.Dense(1, activation="sigmoid", name=f"{name}_out")(x)
    return Model(base.input, out, name=name)

def build_alexnet():
    inp = tf.keras.Input((224, 224, 3))
    x = layers.Conv2D(96, 11, strides=4, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 5, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(384, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(384, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="AlexNet")

MODEL_BUILDERS = {
    "AlexNet":        build_alexnet,
    "VGG16":          lambda: add_head(VGG16(weights=None, include_top=False, input_shape=(224,224,3)), "VGG16"),
    "ResNet50":       lambda: add_head(ResNet50(weights=None, include_top=False, input_shape=(224,224,3)), "ResNet50"),
    "ResNet152":      lambda: add_head(ResNet152(weights=None, include_top=False, input_shape=(224,224,3)), "ResNet152"),
    "MobileNet":      lambda: add_head(MobileNet(weights=None, include_top=False, input_shape=(224,224,3)), "MobileNet"),
    "EfficientNetV2": lambda: add_head(EfficientNetV2S(weights=None, include_top=False, input_shape=(224,224,3)), "EfficientNetV2"),
}

# Default weights from paper (used if ensemble_weights.json not found)
DEFAULT_WEIGHTS = {
    "ResNet152": 0.9622, "EfficientNetV2": 0.9478, "AlexNet": 0.9455,
    "ResNet50": 0.9451, "VGG16": 0.9108, "MobileNet": 0.8717,
}

@st.cache_resource
def load_models_and_weights():
    """
    Checks which trained weight files are available.
    Models are NOT built or loaded here -- only one model is built,
    used, and discarded at a time during prediction, to keep peak
    memory low on free-tier hardware.
    """
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")

    weights_config_path = os.path.join(weights_dir, "ensemble_weights.json")
    if os.path.exists(weights_config_path):
        with open(weights_config_path) as f:
            raw_weights = json.load(f)
    else:
        raw_weights = DEFAULT_WEIGHTS

    total = sum(raw_weights.values())
    ensemble_weights = {k: v / total for k, v in raw_weights.items()}

    weight_paths = {}
    missing = []

    for name in MODEL_BUILDERS:
        weight_path = os.path.join(weights_dir, f"{name}.h5")
        if os.path.exists(weight_path):
            weight_paths[name] = weight_path
        else:
            missing.append(name)

    return weight_paths, ensemble_weights, missing


def predict_one_model(name, weight_path, input_tensor):
    """
    Builds a single model, loads its weights, predicts once, then
    frees it from memory. Keeps peak memory to ~1 model at a time.
    """
    model = MODEL_BUILDERS[name]()
    model.load_weights(weight_path)
    prob = float(model.predict(input_tensor, verbose=0)[0][0])
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return prob


# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown('<div class="main-title">🔬 Acute Lymphoblastic Leukemia Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Weighted Ensemble CNN · C-NMC 2019 · 97.54% Accuracy · '
    '<a href="https://doi.org/10.1109/DASA63652.2024.10836445" target="_blank">IEEE DASA 2024</a></div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Load models
# ---------------------------------------------------------------
with st.spinner("Loading models... (first load takes ~30 seconds)"):
    models, ensemble_weights, missing_models = load_models_and_weights()

if missing_models:
    st.warning(
        f"⚠️ Trained weights not found for: {', '.join(missing_models)}. "
        "Run `train_colab.py` in Google Colab first, then place the `.h5` files "
        "in the `weights/` folder. See README for instructions."
    )

if not models:
    st.error("No trained models found. Please train the models first using `train_colab.py`.")
    st.stop()

# ---------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------
st.divider()
st.subheader("Upload a blood smear image")
st.caption(
    "Upload a microscopy blood smear image (JPG or PNG). "
    "Best results with C-NMC 2019 style images (450×450 with dark background). "
    "You can download test images from the "
    "[C-NMC 2019 dataset](https://competitions.codalab.org/competitions/20395)."
)

uploaded_file = st.file_uploader(
    "Choose image",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed",
)

if uploaded_file is not None:

    # Load image
    pil_image = Image.open(uploaded_file).convert("RGB")

    # Show preprocessing pipeline
    st.subheader("Preprocessing pipeline")
    orig, cropped, enhanced, final = get_intermediate_images(pil_image)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(orig, caption="1. Original", use_container_width=True)
    with col2:
        st.image(cropped, caption="2. Center crop (black removed)", use_container_width=True)
    with col3:
        st.image(enhanced, caption="3. CLAHE applied", use_container_width=True)
    with col4:
        st.image(final, caption="4. Resized 224×224", use_container_width=True)

    # Prepare input tensor
    preprocessed = preprocess_image(pil_image)
    input_tensor = np.expand_dims(preprocessed, axis=0)

    # Run ensemble prediction
    st.subheader("Prediction")

    with st.spinner("Running ensemble prediction across all models..."):
        individual_probs = {}
        weighted_sum = 0.0

        for name, weight_path in models.items():
            prob = predict_one_model(name, weight_path, input_tensor)
            individual_probs[name] = prob
            if name in ensemble_weights:
                weighted_sum += prob * ensemble_weights[name]

    prob_all = weighted_sum
    prob_normal = 1.0 - prob_all
    is_all = prob_all > 0.5
    confidence = prob_all if is_all else prob_normal

    # Result card
    if is_all:
        st.markdown(f"""
        <div class="result-all">
            <h3 style="margin:0;color:#b71c1c">⚠️ ALL Detected — Blast Cell</h3>
            <p style="margin:0.5rem 0 0;color:#c62828">
                Ensemble confidence: <strong>{confidence:.1%}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-normal">
            <h3 style="margin:0;color:#1b5e20">✅ Normal Cell — HEM</h3>
            <p style="margin:0.5rem 0 0;color:#2e7d32">
                Ensemble confidence: <strong>{confidence:.1%}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Probability bars
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("ALL probability", f"{prob_all:.1%}")
        st.progress(float(prob_all))
    with col_b:
        st.metric("Normal probability", f"{prob_normal:.1%}")
        st.progress(float(prob_normal))

    # Individual model breakdown
    st.subheader("Individual model predictions")
    cols = st.columns(len(individual_probs))
    model_display_names = {
        "AlexNet": "AlexNet", "VGG16": "VGG-16", "ResNet50": "ResNet-50",
        "ResNet152": "ResNet-152", "MobileNet": "MobileNet", "EfficientNetV2": "EfficientNetV2",
    }
    
    for col, (name, prob) in zip(cols, individual_probs.items()):
        with col:
            label = "⚠️ ALL" if prob > 0.5 else "✅ Normal"
            st.metric(
                label=model_display_names.get(name, name),
                value=label,
                delta=f"ALL prob: {prob:.1%}",
                delta_color="inverse" if prob > 0.5 else "normal",
            )

    # Ensemble weight breakdown
    with st.expander("Show ensemble weight breakdown"):
        weight_data = []
        for name in individual_probs:
            w = ensemble_weights.get(name, 0)
            weight_data.append({
                "Model": model_display_names.get(name, name),
                "ALL Probability": f"{individual_probs[name]:.4f}",
                "Ensemble Weight": f"{w:.4f}",
                "Weighted Contribution": f"{individual_probs[name] * w:.4f}",
            })
        st.dataframe(pd.DataFrame(weight_data), hide_index=True, use_container_width=True)
        st.caption(f"Final ensemble probability (sum of weighted contributions): **{prob_all:.4f}**")

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical disclaimer:</strong> This is a research demonstration tool based on 
        the published IEEE paper. It is <strong>not validated for clinical use</strong> and must 
        not be used to make medical decisions. Always consult a qualified haematologist for 
        diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

else:
    # Placeholder when no image uploaded
    st.info(
        "👆 Upload a blood smear microscopy image above to begin. "
        "The model will classify it as ALL (blast cell) or Normal (HEM cell) "
        "using a weighted ensemble of 6 CNN architectures."
    )

    # Show example of what a good image looks like
    st.divider()
    st.markdown("### What kind of image does this expect?")
    st.markdown("""
    The model was trained on **C-NMC 2019** microscopy images:
    - Blood smear images under microscope
    - Purple-stained cell images (Giemsa stain)
    - Dark/black background with a single cell visible
    - Original size: 450×450 pixels (the app handles any size)
    
    You can get test images from the 
    [C-NMC 2019 CodaLab competition page](https://competitions.codalab.org/competitions/20395) 
    after registering for free.
    """)
