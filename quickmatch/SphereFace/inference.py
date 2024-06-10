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
from tqdm import tqdm
warnings.simplefilter('ignore')
#from torchsummary import summary
#import torchsummary
from time import time
from net_sphere import sphere20a
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
        self.input_folder = input_folder
        self.transforms = transforms

    def __len__(self):
        return len(os.listdir(self.input_folder))
    
    def __getitem__(self, idx):
        img_name = os.listdir(self.input_folder)[idx]
        img_path = os.path.join(self.input_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((112, 96))
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def infer(backbone, inference_loader, output_path):
    feats = []
    with torch.no_grad():
        for img in tqdm(inference_loader):
            img = img.float().cuda()
            img = (img-127.5) / 128.0
            features = backbone(img)
            feats.append(features.cpu())
            del img
            del features
        
        feats = torch.cat(feats, dim=0)
        torch.save(feats, output_path)

def main(input_dir, output_path):
    dev = torch.device('cuda')
    backbone = sphere20a(feature=True)
    model_path = f'{args.name}/model/sphere20a_20171020.pth'
    state_dict = torch.load(model_path)
    backbone.load_state_dict(state_dict)
    backbone = backbone.to(dev)
    backbone.eval()

    inference_transforms = transforms.Compose([
        transforms.PILToTensor(),
    ])

    inference_dataset = InferenceDataset(input_folder=input_dir, transforms=inference_transforms)
    inference_loader = DataLoader(inference_dataset, batch_size=32)
    infer(backbone, inference_loader, output_path)


main(args.input_dir, args.output_path)
