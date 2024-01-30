FROM nvcr.io/nvidia/tensorflow:24.01-tf2-py3
ENV TZ=Europe/Brussels
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y python3-opencv
RUN git clone https://github.com/arnohe/dermoscopic-attribute-detection.git
RUN pip install --upgrade pip
RUN pip install -r dermoscopic-attribute-detection/requirements.txt
ADD /data/ISIC dermoscopic-attribute-detection/data/ISIC
ADD /data/processed dermoscopic-attribute-detection/data/processed