# dermoscopic-attribute-detection
Author: Arno Heirman

Master's thesis: https://lib.ugent.be/catalog/rug01:003150464

To ensure compatiblity the project can be run using the NVIDIA container image of TensorFlow\
Alternativly project can be run directly (see requirements.txt for compatibility)

# Instructions

## Run directly

Download the dataset
```
python main.py download
```
Preprocess the images to a fixed size (Add `--help` for further options)
```
python main.py preprocess
```
Configure the wandb project in dermo_attributes/config.py

Train a model (Add `--help` to list the parameters)
```
python main.py train
```
Run a gridsearch sweep to test the parameters (Add `--help` to list the parameters)
```
python main.py sweep
```

## Container setup

First install [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)

Next set up the datasets in `/data`
```
python main.py download
python main.py preprocess
```

Build the container with the Dockerfile
```
docker build -t dermo-attributes .
```
  
Run the docker container interactively
```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -ti dermo-attributes
```
Now you can run main.py from the `dermoscopic-attribute-detection` directory
