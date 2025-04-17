import torch
from torchvision import models

# Load the pre-trained ResNet50 model
resnet50 = models.resnet50(pretrained=True)

# Save the model state dict to a specific location
resnet50_path = './models/resnet50.pth'
torch.save(resnet50.state_dict(), resnet50_path)

print(f'ResNet50 model saved at {resnet50_path}')
