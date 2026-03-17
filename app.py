import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import base64
import os
from src import config

# ==========================================
# 1. Configuration & Model Loading
# ==========================================
@st.cache_resource
def load_model():
    if not os.path.exists(config.MODEL_SAVE_PATH):
        st.error(f"Model not found at {config.MODEL_SAVE_PATH}. Please run train.py first.")
        return None
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    return model

# ==========================================
# 2. Background Slideshow & CSS Styling  
# ==========================================
def set_background_slideshow(image_list):
    """
    Creates a looping CSS animation for the background.
    Adds 'Glassmorphism' (semi-transparent boxes) styling to specific elements.
    """
    b64_images = []
    for image_file in image_list:
        try:
            with open(image_file, "rb") as f:
                img_data = f.read()
            b64_encoded = base64.b64encode(img_data).decode()
            b64_images.append(b64_encoded)
        except FileNotFoundError:
            st.warning(f"Background image not found: {image_file}")
            return

    if len(b64_images) < 3:
        st.error("Please provide 3 images (bg1.jpg, bg2.jpg, bg3.jpg).")
        return

    style = f"""
        <style>
        /* --- Background Animation --- */
        @keyframes backgroundScroll {{
            0% {{ background-image: url(data:image/png;base64,{b64_images[0]}); }}
            33% {{ background-image: url(data:image/png;base64,{b64_images[1]}); }}
            66% {{ background-image: url(data:image/png;base64,{b64_images[2]}); }}
            100% {{ background-image: url(data:image/png;base64,{b64_images[0]}); }}
        }}

        .stApp {{
            animation-name: backgroundScroll;
            animation-duration: 20s;
            animation-iteration-count: infinite;
            animation-timing-function: ease-in-out;
            background-size: 100vw 100vh;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* --- Transparent Main Container --- */
        .block-container {{
            background-color: transparent !important;
            box-shadow: none !important;
            border: none !important;
        }}

        /* --- Global Text Styling (White with Shadow) --- */
        h1, h2, h3, h4, h5, h6, p, span, label, div {{
            color: #FFFFFF !important;
            text-shadow: 2px 2px 5px #000000;
        }}

        /* --- STYLING THE FILE UPLOADER BOX --- */
        /* This targets the specific drag-and-drop area */
        [data-testid="stFileUploader"] section {{
            background-color: rgba(0, 0, 0, 0.7) !important; /* Dark semi-transparent box */
            border-radius: 15px;
            padding: 20px;
            border: 2px dashed #4CAF50;
        }}
        
        /* Fix button text inside uploader */
        button {{
            color: #000000 !important; /* Make button text black so it's readable on white buttons */
            text-shadow: none !important; 
        }}

        /* --- STYLING THE ANALYZE BUTTON --- */
        .stButton>button {{
            color: white !important;
            background-color: #0e1117;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            font-size: 20px;
            padding: 10px 24px;
            transition: 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #4CAF50;
            border-color: white;
        }}
        
        /* --- CUSTOM CLASS FOR RESULT BOX --- */
        .result-card {{
            background-color: rgba(0, 0, 0, 0.75); /* Darker box for results */
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #4CAF50;
            margin-top: 20px;
            margin-bottom: 20px;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# ==========================================
# 3. Helper Functions
# ==========================================
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    img_reshape = img_reshape / 255.0 
    prediction = model.predict(img_reshape)
    return prediction, img_reshape

def generate_gradcam(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ==========================================
# 4. Main Application
# ==========================================
def main():
    model = load_model()
    if model is None: return

    images = ['bg1.jpg', 'bg2.jpg', 'bg3.jpg'] 
    set_background_slideshow(images)

    st.title("🩺 Skin Lesion XAI Diagnostic Tool")
    st.markdown("### AI-Powered Multi-Class Classification & Visualization")

    # --- File Uploader ---
    file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file)
        
        # Center image display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # --- Analyze Button ---
        if st.button("Analyze Lesion"):
            with st.spinner('Analyzing...'):
                prediction, img_array = import_and_predict(image, model)
                
                class_names = [
                    'Actinic keratoses (akiec)', 'Basal cell carcinoma (bcc)', 
                    'Benign keratosis (bkl)', 'Dermatofibroma (df)', 
                    'Melanoma (mel)', 'Melanocytic nevi (nv)', 
                    'Vascular lesions (vasc)'
                ]
                predicted_index = np.argmax(prediction)
                predicted_class = class_names[predicted_index]
                confidence = np.max(prediction) * 100

                # --- NEW RESULTS SECTION WITH BACKGROUND BOX ---
                risk_color = "#FF4B4B" if predicted_index in [1, 4] else "#4CAF50"
                risk_msg = "⚠️ High Risk Detected" if predicted_index in [1, 4] else "✅ Likely Benign"

                # We inject HTML here to create the background box for results
                st.markdown(f"""
                <div class="result-card">
                    <h2 style="margin-bottom: 0px;">🔍 Prediction: <span style="color: {risk_color};">{predicted_class}</span></h2>
                    <h3 style="margin-top: 5px;">📊 Confidence: {confidence:.2f}%</h3>
                    <hr style="border: 1px solid white;">
                    <h4 style="color: {risk_color};">{risk_msg}</h4>
                    
                </div>
                """, unsafe_allow_html=True)
                # ------------------------------------------------

                # --- XAI Section ---
                st.divider()
                st.subheader("💡 Explainable AI (Grad-CAM)")
                st.write("The AI focused on the **Red/Yellow** areas to make this decision.")
                
                heatmap = generate_gradcam(img_array, model, 'conv5_block3_out')
                heatmap = np.uint8(255 * heatmap)
                jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
                jet = cv2.resize(jet, (image.size[0], image.size[1]))
                original_img = np.array(image)
                overlay = jet * 0.4 + original_img
                overlay = np.clip(overlay, 0, 255).astype('uint8')

                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(jet, caption="AI Attention Map", use_container_width=True)
                with col_b:
                    st.image(overlay, caption="Overlay on Lesion", use_container_width=True)

if __name__ == "__main__":
    main()