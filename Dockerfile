FROM nvcr.io/nvidia/tensorflow:23.12-tf2-py3
ENV TZ=Europe/Brussels
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y python3-opencv
RUN pip install --upgrade pip
COPY . ./
RUN pip install -r requirements.txt