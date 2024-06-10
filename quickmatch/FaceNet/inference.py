import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import pickle
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
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

def inference_transforms(mtcnn, img):
    img_cr = mtcnn(img)
    return img_cr

class InferenceDataset(Dataset):
    def __init__(self, input_folder, mtcnn, transforms=None):
        self.input_folder = input_folder
        self.transforms = transforms
        self.mtcnn = mtcnn

    def __len__(self):
        return len(os.listdir(self.input_folder))
    
    def __getitem__(self, idx):
        img_name = os.listdir(self.input_folder)[idx]
        img_path = os.path.join(self.input_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        #img = img.resize((112,112))
        if self.transforms is not None:
            tmp = img
            img = self.transforms(self.mtcnn, img)
            if img == None:
                img = tmp.resize((256, 256))
                img = transforms.Compose([transforms.PILToTensor()])(img)
        return img

def inference(resnet, inference_loader, output_path):
    features = []
    with torch.no_grad():
        for img in tqdm(inference_loader):
            tmp = img   
            img = torch.cat(img, axis=0).view((len(tmp), 3, 256, 256))
            img = img.cuda()    
            ie = resnet(img) 
            features.append(ie.cpu())
    
    features = torch.cat(features, dim=0)
    torch.save(features, output_path)

def main(input_dir, output_path):
    mtcnn = MTCNN(image_size=256)
    resnet = InceptionResnetV1(pretrained='vggface2').cuda().eval()

    inference_dataset = InferenceDataset(input_folder=input_dir, mtcnn=mtcnn, transforms=inference_transforms)
    inference_loader = DataLoader(inference_dataset, batch_size=32, collate_fn=lambda x: x)
    inference(resnet, inference_loader, output_path)


main(args.input_dir, args.output_path)

