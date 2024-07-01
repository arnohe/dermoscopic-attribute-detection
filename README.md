# Detection of dermoscopic attributes
Author: Arno Heirman

## Project information

This repository contains the code related to my [Master's thesis](https://lib.ugent.be/catalog/rug01:003150464) and can be used to reproduce my results.
This project is based on the data from the [ISIC 2018](https://challenge.isic-archive.com/landing/2018/46/) competition.
The project makes use of my [Weights & Biases project](https://wandb.ai/arno/lesion-attributes) to store runs and models.

The goal is the detection of certain structures in dermoscopic images, with the aim of aiding in the diagnosis of melanoma.
This is achieved through machine learning using a UNet-architecture with a compact encoder (EfficientNetV2) to segment the structures.
The main challenges are related to problems with the labelled dataset, including heavy data imbalance.
To address this two families of loss functions and an oversampling technique are evaluated.
To improve interpretability, heatmaps of the model output are produced.

The following image is an example of the model output for the five different structures from left to right.
The middle row shows a heatmap of the raw model output.
The top and bottom row show a comparison of the produced segmentation masks to the ground truth.
<p align="center">
<img src="https://github.com/arnohe/dermoscopic-attribute-detection/blob/main/thesis_BFL_output.png?raw=true" width="800">
</p>



## Instructions

### Initial setup
First install the python packages
```
pip install -r requirements.txt 
```
Download the dataset
```
python main.py download
```
Preprocess the images to a fixed size
```
python main.py preprocess --size 512
```
Use --size to set the width

### Container setup

To ensure compatibility the project can be run using the NVIDIA container image of TensorFlow

First install [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)

Next build the container with the Dockerfile
```
docker build -t dermo-attributes .
```
  
Run the docker container interactively
```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -ti -e WANDB_API_KEY=$YOUR_KEY dermo-attributes
```

### Model training

Configure the wandb project in dermo_attributes/config.py\
Train a model (Add --help to list all parameters)
```
python main.py train
```
Run a gridsearch sweep for loss parameters and oversampling method (Add --help to list all parameters)
```
python main.py sweep
```
Process validation results of the gridsearch\
Table summary of best parameters is printed\
Barplot and heatplots are saved to data/results\
Use --metric to change the metric
```
python main.py validation
```
Calculate final ISIC test scores for models given their wandb index\
Also saves an image output with example validation segmentations to data/results
```
python main.py test --idx model_id
```
