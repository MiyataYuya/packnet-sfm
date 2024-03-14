FROM nvcr.io/nvidia/pytorch:23.09-py3
# FROM ubuntu:20.04
# FROM horovod/horovod:latest

RUN export PYTHONPATH="${PYTHONPATH}:/workspaces/packnet-sfm/"
RUN apt update && apt upgrade -y
RUN pip install yacs termcolor wandb matplotlib opencv-python-headless

# install apex
# RUN git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# install horovod (for distributed training)
# RUN HOROVOD_WITHOUT_PYTORCH=1 pip install horovod[pytorch]
