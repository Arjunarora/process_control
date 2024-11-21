# Stage 1: Base image
# See https://www.tensorflow.org/install/source#gpu for compatibility tensorflow <-> CUDA
# FROM nvidia/cuda:11.7.0-base-ubuntu20.04 AS crystalml_base
FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04 AS crystalml_base

# Stop tzdata from hanging the build process...
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Stop install proecesses from hanging build with user inputs
ENV DEBIAN_FRONTEND noninteractive

# Update
RUN apt update
RUN apt upgrade -y

# For GPU Support
# See https://docs.nvidia.com/deploy/cuda-compatibility/ for compatibility CUDA <-> Nvidia drivers
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute
RUN #apt-get install -y cuda-drivers-470 cuda-libraries-11-4 libcudnn8=8.2.2.26-1+cuda11.4
RUN apt install -y cuda-drivers-470

# Install python3.10 (has to be installed from deadsnakes repo)
RUN apt install -y --no-install-recommends software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt install -y --no-install-recommends python3.10-dev python3.10-venv python3-pip git
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.10 /usr/bin/python3

# Stage 2: Image with git repo
FROM crystalml_base AS crystalml

# Mount persistent volume at /data; Must contain the cloned repo of target branch of crystalML
VOLUME ["/data"]
WORKDIR /data

# Change PYTHONPATH so modules can be loaded properly
ENV PYTHONPATH=/data

CMD /bin/bash ./entrypoint.sh



