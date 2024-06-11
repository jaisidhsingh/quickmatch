import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.simplefilter('ignore')
import argparse


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

def inference(resnet, inference_loader, output_path, args):
    features = []
    with torch.no_grad():
        for img in tqdm(inference_loader):
            tmp = img   
            img = torch.cat(img, axis=0).view((len(tmp), 3, 256, 256))
            img = img.to(args.device)    
            ie = resnet(img) 
            features.append(ie.cpu())
    
    feats = torch.cat(features, dim=0)
    print(f"Made face matcher embeddings with shape: {feats.shape}")
    torch.save(features, output_path)
    print(f"Embeddings saved at {output_path}")

def main(input_dir, output_path, args):
    mtcnn = MTCNN(image_size=256)
    resnet = InceptionResnetV1(pretrained='vggface2').to(args.device).eval()

    inference_dataset = InferenceDataset(input_folder=input_dir, mtcnn=mtcnn, transforms=inference_transforms)
    
    print(f"Started processing {len(inference_dataset)} images.")
    
    inference_loader = DataLoader(inference_dataset, batch_size=32, collate_fn=lambda x: x)
    inference(resnet, inference_loader, output_path, args)
    
    print("Done")
