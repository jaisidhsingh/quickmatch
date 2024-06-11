import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .backbones import get_model
from torchvision import transforms
from tqdm import tqdm
from types import SimpleNamespace
from PIL import Image


def inference_transforms(img):
    img = Image.open(img).convert("RGB")
    img = img.resize((112, 112))
    img = np.array(img)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img


class InferenceDataset2(Dataset):
    def __init__(self, input_folder, transforms=None):
        self.input_folder = input_folder
        self.transforms = transforms

    def __len__(self):
        return len(self.input_folder)

    def __getitem__(self, idx):
        img_name = os.listdir(self.input_folder)[idx]
        img_path = os.path.join(self.input_folder, img_name)

        img_path = self.input_folder[idx]

        img = Image.open(img_path).convert('RGB')
        img = img.resize((112, 112))
        if self.transforms is not None:
            img = self.transforms(img_path)
        return img


class InferenceDataset(Dataset):
    def __init__(self, input_folder, transforms=None):
        self.input_folder = input_folder
        self.transforms = transforms

    def __len__(self):
        return len(os.listdir(self.input_folder))

    def __getitem__(self, idx):
        img_name = os.listdir(self.input_folder)[idx]
        img_path = os.path.join(self.input_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((112, 112))
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def infer(backbone, inference_loader, output_path, cfg):
    feats = []
    with torch.no_grad():
        for img in tqdm(inference_loader):
            img = img.float().to(cfg.device)
            feat = backbone(img)
            feats.append(feat.cpu())
        feats = torch.cat(feats, dim=0)
        print(f"Made face matcher embeddings with shape: {feats.shape}")
        torch.save(feats, output_path)
        print(f"Embeddings saved at {output_path}")


def main(input_dir, output_path, args):
    cfg = SimpleNamespace(**{})
    cfg.device = args.device
    cfg.weights = os.path.join(args.ckpt_folder, "arcface_backbone.pth")
    cfg.network = "r50"

    backbone = get_model(cfg.network, fp16=False)
    backbone.load_state_dict(torch.load(cfg.weights, map_location=cfg.device))
    backbone.eval()
    backbone.to(cfg.device)

    inference_transforms = transforms.Compose([
        transforms.PILToTensor(),
    ])

    inference_dataset = InferenceDataset(
        input_folder=input_dir, transforms=inference_transforms)
    
    print(f"Started processing {len(inference_dataset)} images.")
    
    inference_loader = DataLoader(inference_dataset, batch_size=32)
    infer(backbone, inference_loader, output_path, cfg)
    
    print("Done")
