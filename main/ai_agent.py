import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime

# -----------------------
# Set model paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models/gemma-lora-final')
RESNET_PATH = os.path.join(BASE_DIR, 'models/resnet50.pth')
EFFICIENTNET_PATH = os.path.join(BASE_DIR, 'models/efficientnet_b0.pth')

# -----------------------
# Set up device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load models once (if needed)
# -----------------------
# ResNet and EfficientNet
resnet = models.resnet50(weights=None)
resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device))
resnet = resnet.to(device).eval()

efficientnet = models.efficientnet_b0(weights=None)
efficientnet.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device))
efficientnet = efficientnet.to(device).eval()

# Load Gemma model and tokenizer with LoRA
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

# -----------------------
# Image preprocessing
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------
# Helper functions
# -----------------------
def extract_features(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # ResNet features
        x_resnet = resnet.conv1(image_tensor)
        x_resnet = resnet.bn1(x_resnet)
        x_resnet = resnet.relu(x_resnet)
        x_resnet = resnet.maxpool(x_resnet)
        x_resnet = resnet.layer1(x_resnet)
        x_resnet = resnet.layer2(x_resnet)
        x_resnet = resnet.layer3(x_resnet)
        x_resnet = resnet.layer4(x_resnet)
        x_resnet = resnet.avgpool(x_resnet)
        resnet_feat = x_resnet.view(x_resnet.size(0), -1)

        # EfficientNet features
        x_eff = efficientnet.features(image_tensor)
        x_eff = efficientnet.avgpool(x_eff)
        efficient_feat = torch.flatten(x_eff, 1)

        combined = torch.cat([resnet_feat, efficient_feat], dim=1)
        return combined.squeeze().cpu().numpy()


def encode_image_features(features, num_features=10):
    return ", ".join([f"{val:.3f}" for val in features[:num_features]])

def format_gemma_prompt(findings, indications, resnet_feat, eff_feat):
    instruction = "Generate an impression for a medical radiograph based on the findings, indications, and image features."
    input_text = (
        f"Findings: {findings}\n"
        f"Indications: {indications}\n"
        f"ResNet features: {encode_image_features(resnet_feat)}\n"
        f"EfficientNet features: {encode_image_features(eff_feat)}"
    )
    return f"<start_of_turn>user\n{instruction}\n\n{input_text}<end_of_turn>\n<start_of_turn>model\n"

def generate_caption_from_xray(image_path):
    # Step 1: Open image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)

    # Step 2: Extract features
    features = extract_features(image_tensor)
    resnet_feat = features[:2048]
    eff_feat = features[2048:2048+1280]

    # Step 3: Static text fields (or replace with optional form input later)
    findings = "No acute abnormality"
    indications = "Routine checkup"

    # Step 4: Generate caption
    prompt = format_gemma_prompt(findings, indications, resnet_feat, eff_feat)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    caption = decoded.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()

    # Step 5: Save dummy Grad-CAM (replace later)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    gradcam_path = f"gradcam_{timestamp}.png"
    gradcam_full_path = os.path.join("media", gradcam_path)
    image.save(gradcam_full_path)

    return {
        "caption": caption,
        "gradcam_path": gradcam_path
    }
