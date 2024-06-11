# QUICKMATCH

A simply command-line tool for conveniently using SOTA face-recognition networks. 

## Provided Face-Matchers

1. <a href="https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Boutros_ElasticFace_Elastic_Margin_Loss_for_Deep_Face_Recognition_CVPRW_2022_paper.pdf">ElasticFace</a>
2. <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf">ArcFace</a>
3. <a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf">SphereFace</a>
4. <a href="https://arxiv.org/pdf/1503.03832">FaceNet</a>

## Installation

This package is released on <a href="https://pypi.org">PyPi</a> and can be found <a href="https://pypi.org/project/quickmatch/0.1.0/">here</a>. Installation can be simply done by `pip install quickmatch`. 

### Dependancies

The following Python libraries are required for running inference on the face-matchers and can be installed by using `pip install <library-name>`. The package will automatically install them at the versions that were used to test it.

```plaintext
easydict
torch
tqdm
facenet-pytorch
onedrivedownloader
```

## Usage

This library functions as a commandline tool which takes in a directory of images and the face-matcher you want to use create face-matcher embeddings that are stored in a `.pt` file. This `.pt` contains a stack of PyTorch tensors corresponding to all images. Note that the shape of the PyTorch tensor stack will be `[N, D]` where `N` is the number of images you provided in the input directory and `D` is the dimension that the matchers embed to (`D`$=512$ for all matchers).

For example, if you want to use "ArcFace" on a folder of images called `my_face_shots`, run the command by specifying these inputs and the output path of the file.

```bash
python3 -m ez-face-match --matcher=ArcFace --input-folder=my_face_shots --output-path=./matcher_embeddings.pt
```

The `main.py` file automatically checks if `"cuda"` is enabled or not (<a href="https://pytorch.org/">PyTorch</a> must be installed and compiled with CUDA). Note, however, using a purely CPU runtime for the inference of these networks may take significantly longer. Additionally, upon first time use, the script will create a folder named `quickmatch_pretrained_models` at your default `pip` cache location. Here, the model weights will be downloaded and loaded automatically.
