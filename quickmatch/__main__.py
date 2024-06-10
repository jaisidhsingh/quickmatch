import os
import torch
import argparse
import subprocess
from onedrivedownloader import download
from sys import platform
from .ArcFace import inference as arcface_inference
from .ElasticFace import inference as elasticface_inference
from .SphereFace import inference as sphereface_inference
from .FaceNet import inference as facenet_inference


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matcher", type=str, required=True)
    parser.add_argument("--input-folder", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--ckpt-folder", type=str, default="quickmatch_pretrained_models")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    return args


def download_checkpoint(args):
	if not torch.cuda.is_available():
		print("CUDA not found, using CPU instead.")
		args.device = "cpu"

	if platform == "linux" or platform == "linux2":   
		ckpt_save_folder = os.path.join(os.getenv("HOME"), ".cache", "pip", args.ckpt_folder)

	if platform == "win32":
		ckpt_save_folder = os.path.join(os.getenv("LOCALAPPDATA"), "pip", "Cache", args.ckpt_folder)

	if platform == "darwin":
		ckpt_save_folder = os.path.join(os.getenv("HOME"), "Library", "Caches", "pip", args.ckpt_folder)

	os.makedirs(ckpt_save_folder, exist_ok=True)

	model_save_folder = os.path.abspath(ckpt_save_folder)
	args.ckpt_folder = model_save_folder
	main_folder = os.path.abspath(".")

	os.chdir(model_save_folder)

	urls_map = {
		"ArcFace": "https://iitjacin-my.sharepoint.com/:u:/g/personal/singh_118_iitj_ac_in/EUx4gU4fYBtPtCjjsRpOz7IBTR_GSQHVUpGYBDGPBB5Ajg?e=CbLFat",
		"ElasticFace": "https://iitjacin-my.sharepoint.com/:u:/g/personal/singh_118_iitj_ac_in/EU2N--g9Om1ElQiukvtFokwB94uH4YRG2GOAdStswK2u6A?e=TfaOGL",
		"SphereFace": "https://iitjacin-my.sharepoint.com/:u:/g/personal/singh_118_iitj_ac_in/Ecv-gkrQggJEleJuUxqUWFsBPQQ24O7W0bZnA6-WlXKaZg?e=HFArcM",
	}
	all_good = True

	if args.matcher != "FaceNet":
		url = urls_map[args.matcher]

		if not os.path.exists(f"{args.matcher.lower()}_backbone.pth"):
			try:
				download(
					url, 
					filename=f"{args.matcher.lower()}_backbone.pth", 
					unzip=False, 
					unzip_path= None, 
					force_download=False, 
					force_unzip=False, 
					clean=False
				)
				all_good = True
			except Exception as err:
				all_good = False
				print(err)

		else:
			al_good = True

	else:
		print("Model already downloaded and cached. Loading from cache directly.")

	os.chdir(main_folder)
	return all_good


def run(args):
	go_ahead = download_checkpoint(args)

	if go_ahead:
		# subprocess.call(f"python quickmatch/{args.matcher}/inference.py --name={args.matcher} --input-dir={args.input_folder} --output-path={args.output_path}")
		inference.main(args.input_folder, args.output_path, args)
	else:
		print("Error in downloading model checkpoint. Embedding generation cannot run.")


if __name__ == "__main__":
    args = setup_args()
    run(args)
