#!/bin/bash

matcherName="ArcFace"
inputDir="/home/username/some_face_image_collection"

# this library puts the face embeddings in ".pt" files typically used in torch 
outputPath="face_embeddings.pt"

python3 $matcher/inference.py --name=$matcher --input-dir=$inputDir --output-path=$outputPath
