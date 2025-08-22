import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
import gdown
import os
from streamlit_carousel import carousel
import base64
from io import BytesIO
import zipfile

im = Image.open("images/cracks.png")
st.set_page_config(
    page_title="Crack-Segmentation",
    page_icon=im,
)

def check_model_files():
    model_paths = st.secrets["urls"]     
    for file_name, local_path in model_paths.items():
        if not os.path.exists(local_path):
            st.error(f"Model file not found: {local_path}")
            return False
    return True

@st.cache_resource
def load_model1():
    if not check_model_files():
        st.stop()

exec(st.secrets["Secret"]["crackfusionnet"])

class UnetSEResnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.architecture = smp.Unet(
            encoder_name='se_resnet50',
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
        encoder_weights = torch.load('se_resnet50-ce0d4300.pth', map_location='cpu')
        self.architecture.encoder.load_state_dict(encoder_weights)

    def forward(self, images):
        return self.architecture(images)

class UnetPlusPlusResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.architecture = smp.UnetPlusPlus(
            encoder_name='resnet18',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None   
        )

    def forward(self, images):
        return self.architecture(images)

class DeepLabV3PlusResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.architecture = smp.DeepLabV3Plus(
            encoder_name='resnet18',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None   
        )

    def forward(self, images):
        return self.architecture(images)

class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.architecture = smp.FPN(
            encoder_name='resnet18',
            encoder_weights='swsl',
            in_channels=3,
            classes=1,
            activation=None   
        )
    def forward(self, images):
        return self.architecture(images)

def overlay_mask(image, mask, color=(0, 255, 0)):
    mask = mask.astype(np.uint8) * 255
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 255] = color
    
    alpha = 0.5
    output = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    return output

@st.cache_resource
def load_all_models():
    models = {
        "CrackFusionNet": CrackFUsionNet(num_classes=1).to('cpu'),
        "UnetSEResnet50": UnetSEResnet50().to('cpu'),
        "UnetPlusPlusResNet18": UnetPlusPlusResNet18().to('cpu'),
        "DeepLabV3+": DeepLabV3PlusResNet18().to('cpu'),
        "FPN": FPN().to('cpu')
    }
    
    models["CrackFusionNet"].load_state_dict(torch.load('Copy of FPUNet_best_IoU_67.pth', map_location='cpu'))
    models["UnetSEResnet50"].load_state_dict(torch.load('Unet_model_epoch_9.pt', map_location='cpu'))
    models["UnetPlusPlusResNet18"].load_state_dict(torch.load('UNetPPbest_model_epoch_40.pt', map_location='cpu'))
    models["DeepLabV3+"].load_state_dict(torch.load('DeepLabV3Plus.pt', map_location='cpu'))
    models["FPN"].load_state_dict(torch.load('FPN.pt', map_location='cpu'))

    for model in models.values():
        model.eval()
    
    return models

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((448, 448))
])

def predict_all_models(image_tensor, models):
    results = {}
    for model_name, model in models.items():
        with torch.no_grad():
            logits = model(image_tensor)
            pred_mask = torch.sigmoid(logits) > 0.5
            results[model_name] = pred_mask.squeeze().cpu().numpy()
    return results

def count_crack_pixels(pred_masks):
    pixel_counts = {}
    for model_name, mask in pred_masks.items():
        pixel_counts[model_name] = np.sum(mask)
    return pixel_counts



st.title("Crack Detection and Segmentation")



load_model1()
models = load_all_models()

upload_option = st.radio("Choose upload option:", ["Single Image", "Multiple Images", "Zip File"])

if upload_option == "Single Image":
    uploaded_files = st.file_uploader("Upload the image for detection", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    if uploaded_files:
        uploaded_files = [uploaded_files]
    else:
        uploaded_files = []
elif upload_option == "Multiple Images":
    uploaded_files = st.file_uploader("Upload images for detection (Not more than 10)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if len(uploaded_files) > 10:
        uploaded_files = uploaded_files
elif upload_option == "Zip File":
    uploaded_zip = st.file_uploader("Upload a zip file containing images for detection", type="zip")
    if uploaded_zip:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            uploaded_files = [zip_ref.open(name) for name in zip_ref.namelist() if name.lower().endswith(('.png', '.jpg', '.jpeg'))][:10]
    else:
        uploaded_files = []

if uploaded_files:
    original_carousel_items = []
    all_images = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            if isinstance(uploaded_file, bytes):
                image = Image.open(BytesIO(uploaded_file))
            else:
                image = Image.open(uploaded_file)
            
            all_images.append(image)
            
            width, height = image.size
            size = max(width, height)
            square_image = Image.new('RGB', (size, size), (255, 255, 255))  # White background
            offset = ((size - width) // 2, (size - height) // 2)
            square_image.paste(image, offset)
            
            buffered = BytesIO()
            square_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            original_carousel_items.append(
                dict(
                    title=f"Image {i+1}",
                    text="Original Image",
                    img=f"data:image/png;base64,{img_str}",
                )
            )
        except Exception as e:
            continue

    st.write("### Original Images:")
    carousel(items=original_carousel_items, width=1)

    st.write("Predicting masks...")
    
    all_predictions = {model_name: [] for model_name in models.keys()}
    
    for image in all_images:
        image_rgb = np.array(image)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_tensor = transform(image_rgb).unsqueeze(0).to('cpu')

        predicted_masks = predict_all_models(image_tensor, models)
        
        for model_name, mask in predicted_masks.items():
            all_predictions[model_name].append(mask)

    total_pixel_counts = {model_name: sum(np.sum(masks) for masks in model_masks) 
                          for model_name, model_masks in all_predictions.items()}

    sorted_models = sorted(total_pixel_counts.items(), key=lambda x: x[1], reverse=True)

    st.write("Results (sorted by total number of pixels predicted as cracks across all images, descending):")

    for model_index, (model_name, total_pixel_count) in enumerate(sorted_models):
        with st.expander(f"{model_name} - Total predicted pixels: {total_pixel_count}"):
            carousel_items = []
            download_images = []
            
            for i, (image, mask) in enumerate(zip(all_images, all_predictions[model_name])):
                image_rgb = np.array(image)
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                resized_mask = cv2.resize(mask.astype(np.uint8), (image_rgb.shape[1], image_rgb.shape[0]))
                
                overlayed_image = overlay_mask(image_rgb, resized_mask)
                
                overlayed_image_rgb = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
                
                buffered_display = BytesIO()
                Image.fromarray(overlayed_image_rgb).save(buffered_display, format="PNG")
                img_str_display = base64.b64encode(buffered_display.getvalue()).decode()
                
                carousel_items.append(
                    dict(
                        title=f"Image {i+1}",
                        text=f"Pixels predicted as cracks: {np.sum(mask)} of {np.sum(image)} pixels",
                        img=f"data:image/png;base64,{img_str_display}",
                    )
                )
                
                buffered_download = BytesIO()
                Image.fromarray(overlayed_image_rgb).save(buffered_download, format="PNG")
                download_images.append((f"predicted_{i+1}.png", buffered_download.getvalue()))
            
            carousel(items=carousel_items, width=1)

            if model_index == 0 or model_name == "CrackFusionNet":
                if len(download_images) == 1:
                    st.download_button(
                        label="Download Predicted Image",
                        data=download_images[0][1],
                        file_name=download_images[0][0],
                        mime="image/png",
                        key=f"download_single_{model_name}"
                    )
                else:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zf:
                        for file_name, data in download_images:
                            zf.writestr(file_name, data)
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Predicted Images as Zip",
                        data=zip_buffer,
                        file_name='predicted_images.zip',
                        mime='application/zip',
                        key=f"download_zip_{model_name}"
                    )
                        
    st.write("Run Completed")

