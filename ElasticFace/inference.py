#from turtle import back
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pickle
import numpy as np
import random
import argparse
import sys
import warnings
from PIL import Image
from config.config import config as cfg
from tqdm import tqdm
from backbones.iresnet import iresnet100
warnings.simplefilter('ignore')
#from torchsummary import summary
#import torchsummary
from time import time
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



class InferenceDataset(Dataset):
    def __init__(self, input_folder, transforms=None):
        self.input_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
        self.transforms = transforms

    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, idx):
        img_path = self.input_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((112,112))
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def infer(backbone, inference_loader, output_path):
    feats = []
    with torch.no_grad():
        for img in tqdm(inference_loader):
            img = img.float().cuda()
            features = backbone(img)
            feats.append(features.cpu())
            del img
            del features
        
        feats = torch.cat(feats, dim=0)
        torch.save(feats, output_path)

def main(image_dir, output_path):
    dev = torch.device('cuda')
    backbone = iresnet100(num_features=cfg.embedding_size).to(dev)
    model_path = f'{args.name}/models/295672backbone.pth'
    state_dict = torch.load(model_path)
    backbone.load_state_dict(state_dict)
    backbone = backbone.to(dev)
    backbone.eval()

    inference_transforms = transforms.Compose([
        transforms.PILToTensor(),
    ])

    inference_dataset = InferenceDataset(input_folder=image_dir, transforms=inference_transforms)
    inference_loader = DataLoader(inference_dataset, batch_size=32)
    infer(backbone, inference_loader, output_path)



main(args.input_dir, args.output_path)
