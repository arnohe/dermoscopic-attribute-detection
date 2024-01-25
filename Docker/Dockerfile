FROM nvcr.io/nvidia/tensorflow:23.08-tf2-py3
ENV TZ=Europe/Brussels
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y python3-opencv
RUN pip install --upgrade pip
RUN pip install wandb scikit-image imgaug tensorboard-plugin-profile tqdm
RUN git config --global --add safe.directory /tmp