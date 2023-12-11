import argparse
import pickle
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from backbones import get_model
from torchvision import transforms
import time
from cfg import cfg
from tqdm import tqdm
from types import SimpleNamespace
from PIL import Image
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-dir",
    type=str,
    required=True
)
parser.add_argument(
    "--output-path",
    type=str,
    required=True
)
parser.add_argument(
    "--name",
    type=str,
    required=True
)
args = parser.parse_args()


def inference_transforms(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def infer(backbone, inference_loader, output_path):
    feats = []
    with torch.no_grad():
        for img in tqdm(inference_loader):
            img = img.float().cuda()
            feat = backbone(img)
            feats.append(feat.cpu())
        feats = torch.cat(feats, dim=0)
        # print(feats.shape)
        torch.save(feats, output_path)


def main(input_dir, output_path):
    cfg = SimpleNamespace(**{})
    cfg.device = "cuda"
    cfg.weights = f"{args.name}/backbone.pth"
    cfg.network = "r50"

    backbone = get_model(cfg.network, fp16=False)
    backbone.load_state_dict(torch.load(cfg.weights, map_location=cfg.device))
    backbone.eval()
    backbone.cuda()

    inference_transforms = transforms.Compose([
        transforms.PILToTensor(),
    ])

    inference_dataset = InferenceDataset(
        input_folder=input_dir, transforms=inference_transforms)
    print(len(inference_dataset))
    inference_loader = DataLoader(inference_dataset, batch_size=32)
    infer(backbone, inference_loader, output_path)


main(args.input_dir, args.output_path)
