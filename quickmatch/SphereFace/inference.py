import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import warnings
from PIL import Image
from tqdm import tqdm
warnings.simplefilter('ignore')
from .net_sphere import sphere20a


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

def infer(backbone, inference_loader, output_path, args):
    feats = []
    with torch.no_grad():
        for img in tqdm(inference_loader):
            img = img.float().to(args.device)
            img = (img-127.5) / 128.0
            features = backbone(img)
            feats.append(features.cpu())
            del img
            del features
        
        feats = torch.cat(feats, dim=0)
        print(f"Made face matcher embeddings with shape: {feats.shape}")
        torch.save(feats, output_path)
        print(f"Embeddings saved at {output_path}")

def main(input_dir, output_path, args):
    dev = torch.device(args.device)
    backbone = sphere20a(feature=True)
    model_path = os.path.join(args.ckpt_folder, "sphereface_backbone.pth")
    state_dict = torch.load(model_path)
    backbone.load_state_dict(state_dict)
    backbone = backbone.to(dev)
    backbone.eval()

    inference_transforms = transforms.Compose([
        transforms.PILToTensor(),
    ])

    inference_dataset = InferenceDataset(input_folder=input_dir, transforms=inference_transforms)
    
    print(f"Started processing {len(inference_dataset)} images.")
    
    inference_loader = DataLoader(inference_dataset, batch_size=32)
    infer(backbone, inference_loader, output_path, args)
    
    print("Done")
