import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

st.title("Fake News Detection (Text + Image)")

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

# Load ResNet
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()  # remove final layer

# Simple Fusion Classifier
class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.fc = nn.Linear(768 + 512, 2)  # BERT + ResNet features

    def forward(self, text_feat, img_feat):
        combined = torch.cat((text_feat, img_feat), dim=1)
        return self.fc(combined)

fusion_model = FusionModel()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Inputs
text_input = st.text_area("Enter News Text")
uploaded_image = st.file_uploader("Upload News Image", type=["jpg", "png", "jpeg"])

if st.button("Check News"):
    if text_input and uploaded_image:

        # ---- TEXT FEATURES ----
        inputs = tokenizer(
            text_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            text_output = bert(**inputs)
            text_features = text_output.last_hidden_state[:, 0, :]

        # ---- IMAGE FEATURES ----
        image = Image.open(uploaded_image).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            image_features = resnet(image)

        # ---- FUSION ----
        output = fusion_model(text_features, image_features)
        prediction = torch.argmax(output, dim=1)

        if prediction.item() == 0:
            st.success("This News is Real ✅")
        else:
            st.error("This News is Fake ❌")

    else:
        st.warning("Please enter text and upload image.")