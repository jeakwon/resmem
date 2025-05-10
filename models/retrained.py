from models.ResNet import resnet18, resnet50
from huggingface_hub import hf_hub_download
import torch

__all__ = [
    "ResNet18CIFAR99",
    "ResNet50CIFAR99",
]

def ResNet18CIFAR99():
    model = resnet18(num_classes=100)
    file_path = hf_hub_download(repo_id="onlytojay/engram", filename=f"retrain/resnet18/cifar100/0-0/model_best.pth.tar")
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    return model.cuda()  # Added .cuda() to return the model on GPU


def ResNet50CIFAR99():
    model = resnet50(num_classes=100)
    file_path = hf_hub_download(repo_id="onlytojay/engram", filename=f"retrain/resnet50/cifar100/0-0/model_best.pth.tar")
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    return model.cuda()  # Added .cuda() to return the model on GPU