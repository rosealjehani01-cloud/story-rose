import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import json

# ===== FILE PATHS - Edit these paths to match your file locations =====
MODEL_PATH = 'model_torchscript (1).pt'
METADATA_PATH = r"C:\Users\rosek\Desktop\deployyyyy\metadata.json"
# =====================================================================

# Page configuration
st.set_page_config(
    page_title="Skin Disease Predict",
    page_icon="🔬",
    layout="centered"
)

# Custom CSS for clean styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 30px;
        background: #2c3e50;
        border-radius: 10px;
        color: white;
        margin-bottom: 40px;
    }
    .result-card {
        padding: 40px;
        border-radius: 10px;
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        margin: 30px 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .disease-name {
        font-size: 36px;
        font-weight: bold;
        color: #1f2937;
        margin: 20px 0;
    }
    .confidence-score {
        font-size: 28px;
        color: #4b5563;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>Skin Disease Predict</h1>
    </div>
""", unsafe_allow_html=True)


# Load model and metadata
@st.cache_resource
def load_model():
    """Load the TorchScript model and metadata"""
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.jit.load(MODEL_PATH, map_location=device)
        model.eval()

        return model, metadata, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


# Image preprocessing
def preprocess_image(image, metadata):
    """Preprocess the uploaded image"""
    img_size = metadata['img_size']
    mean = metadata['normalize_mean']
    std = metadata['normalize_std']

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


# Predict function
def predict(model, image_tensor, metadata, device):
    """Make prediction on the image"""
    try:
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = metadata['class_names'][predicted_idx.item()]
        confidence_score = confidence.item() * 100

        return predicted_class, confidence_score
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None, None


# Disease information
disease_info = {
    "Acne": "Acne",
    "Aczema": "Eczema",
    "Bcc": "Basal Cell Carcinoma",
    "Melanoma": "Melanoma",
    "Normal Skin": "Normal Skin",
    "Psoriasis": "Psoriasis",
    "Sk": "Seborrheic Keratosis"
}


# Main app
def main():
    model, metadata, device = load_model()

    if model is None:
        return

    # Upload section
    st.markdown("### Upload Skin Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Display image
        st.image(image, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Auto-analyze on upload
        with st.spinner("Analyzing..."):
            img_tensor = preprocess_image(image, metadata)
            predicted_class, confidence = predict(model, img_tensor, metadata, device)

            if predicted_class:
                # Display result
                display_name = disease_info.get(predicted_class, predicted_class)

                st.markdown(f"""
                    <div class="result-card">
                        <div class="disease-name">
                            Disease: {display_name}
                        </div>
                        <div class="confidence-score">
                            Confidence: {confidence:.1f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Warning for serious conditions
                if predicted_class in ["Bcc", "Melanoma"]:
                    st.error("Warning: Results indicate a serious condition. Please consult a doctor immediately.")


if __name__ == "__main__":

    main()
