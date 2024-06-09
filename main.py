import os
import torch
import argparse
import subprocess
from onedrivedownloader import download


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matcher", type=str, default=None)
    parser.add_argument("--input-folder", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--ckpt-folder", type=str, default="./pretrained_models")
    
    args = parser.parse_args()
    return args


def download_checkpoint(args):
	os.makedirs(args.ckpt_folder)

	model_save_folder = os.path.abspath(args.ckpt_folder)
	main_folder = os.path.abspath(".")
 
	os.chwd(model_save_folder)

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
			except:
				all_good = False
		else:
			all_good = True
  
	os.chwd(main_folder)
	return all_good


def run(args):
	go_ahead = download_checkpoint(args)

	if go_ahead:
		subprocess.call(f"python3 {args.matcher}/inference.py --name={args.matcher} --input-dir={args.input_folder} --output-path={args.output_path}")
	else:
		print("Error in downloading model checkpoint.")


if __name__ == "__main__":
    args = setup_args()
    run(args)
