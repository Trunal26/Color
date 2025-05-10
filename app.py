import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Define the Colorization model (copy this from your notebook)
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

# Define the transform (copy this from your notebook)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Define rgb_to_gray (copy this from your notebook)
def rgb_to_gray(img):
    return img.mean(dim=1, keepdim=True)

# Define color exaggeration functions (copy this from your notebook if needed)
def torch_rgb_to_hsv(rgb):
    # ... (copy the function body)
    pass

def torch_hsv_to_rgb(hsv):
    # ... (copy the function body)
    pass

def exaggerate_colors(images, saturation_factor=1.5, value_factor=1.2):
    # ... (copy the function body and call torch_rgb_to_hsv and torch_hsv_to_rgb if needed)
    pass

# Load the trained model
@st.cache_resource  # Cache the model loading to avoid reloading on each rerun
def load_model():
    model = ColorizationNet()
    # Adjust the path to your model file as needed
    model.load_state_dict(torch.load('colorization_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Streamlit UI setup
st.title('Grayscale to Color Image Converter')
st.write("Upload a grayscale image, and it will be colorized by the model.")

# File uploader for images
uploaded_file = st.file_uploader("Choose a grayscale image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image and convert to grayscale
    image = Image.open(uploaded_file).convert("L")

    # Display the uploaded grayscale image
    st.image(image, caption='Uploaded Grayscale Image', use_column_width=True, channels='GRAY')

    # Apply transformations and add batch dimension
    img_tensor = transform(image).unsqueeze(0)

    # Perform colorization
    with torch.no_grad():
        colorized_tensor = model(img_tensor)

    # Convert the output tensor back to a PIL Image
    colorized_img = transforms.ToPILImage()(colorized_tensor.squeeze(0).cpu())

    # Optionally apply color exaggeration
    # colorized_img = exaggerate_colors(colorized_tensor.squeeze(0).cpu())
    # colorized_img = transforms.ToPILImage()((colorized_img + 1) / 2) # Adjust range if using exaggeration

    # Display the colorized image
    st.image(colorized_img, caption='Colorized Image', use_column_width=True)