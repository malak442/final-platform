from efficientnet_pytorch import EfficientNet
import torch

# Load the pre-trained EfficientNet B0 model
model = EfficientNet.from_pretrained('efficientnet-b0')

# Save the model state dict to a specific location
efficientnet_path = './models/efficientnet_b0.pth'
torch.save(model.state_dict(), efficientnet_path)

print(f'EfficientNet B0 model saved at {efficientnet_path}')
