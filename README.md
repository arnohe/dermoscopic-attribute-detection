# dermoscopic-attribute-detection
Author: Arno Heirman

Master's thesis: https://lib.ugent.be/catalog/rug01:003150464

# Instructions

## Setting up the dataset

Download the dataset
```
python main.py download
```
Preprocess the images to a fixed size (Add `--help` for further options)
```
python main.py preprocess
```

## To use the docker setup

First install [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)

Build the docker container
```
docker build -t dermo-attributes .
```

Run the docker container interactively
```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -ti dermo-attributes
```
