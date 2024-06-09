# Ez-Face-Match

A library for conveniently using SOTA face-recognition networks. 

## Provided Face-Matchers

> ElasticFace

> ArcFace

> SphereFace

> FaceNet

## Installation

The following Python libraries are required for running inference on the face-matchers and can be installed by using `pip install <library-name>`.

```plaintext
numpy
torch
tqdm
facenet-pytorch
onedrivedownloader
```

## Usage

This library functions as a commandline tool which takes in a directory of images and the face-matcher you want to use create face-matcher embeddings that are stored in a `.pt` file. This `.pt` contains a stack of PyTorch tensors corresponding to all images.

For example, if you want to use "ArcFace" on a folder of images called `my_face_shots`, run the command by specifying these inputs and the output path of the file.

```bash
python3 -m ez-face-match --matcher=ArcFace --input-folder=my_face_shots --output-path=./matcher_embeddings.pt
```

The `main.py` file automatically checks if `"cuda"` is enabled or not. Note, however, using a purely cpu runtime for the inference of these networks may take significantly longer. Additionally, upon first time use, the script will create a folder named `pretrained_models` where the model weights will be downloaded and loaded automatically.
