FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# install dependencies
#RUN pip install functorch 
# '' may actually no longer be needed from torch 1.13 on
RUN pip install cupy-cuda112
RUN conda install -c conda-forge cupy  
#RUN conda install pytorch torchvision cudatoolkit=11.7 -c pytorch
RUN pip install opencv-python 
# RUN conda install -c conda-forge opencv
RUN pip install scipy rasterio natsort matplotlib scikit-image tqdm pandas
RUN pip install Pillow dominate visdom tensorboard
RUN pip install kornia torchgeometry torchmetrics torchnet segmentation-models-pytorch
RUN pip install s2cloudless 
# see: https://github.com/sentinel-hub/sentinel2-cloud-detector/issues/17
RUN pip install numpy==1.21.6

RUN apt-get -y update
RUN apt-get -y install git
RUN pip install -U 'git+https://github.com/facebookresearch/fvcore'

# just in case some last-minute changes are needed
RUN apt-get install nano

# bake repository into dockerfile
RUN mkdir -p ./data
RUN mkdir -p ./model
RUN mkdir -p ./util

ADD data ./data
ADD model ./model
ADD util ./util
ADD . ./

WORKDIR /workspace/model