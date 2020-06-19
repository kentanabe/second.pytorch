# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvidia/l4t-ml:r32.4.2-py3
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        fish \
        libsparsehash-dev \
        make \
        llvm-9 \
        libboost-all-dev \
        libgeos-dev \
        && \
# ==================================================================
# python
# ------------------------------------------------------------------
    LLVM_CONFIG=/usr/lib/llvm-9/bin/llvm-config
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3 \
        python3-dev \
        python3-pip
        && \
    $PIP_INSTALL \
        -U pip \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        matplotlib \
        Cython \
        && \
    $PIP_INSTALL \
        shapely \
        scikit-image==0.16.2 \
        numba \
        sparsehash \
        fire \
        tensorboardX \
        && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    $PIP_INSTALL \
        shapely \
        scikit-image==0.16.2 \
        numba \
        sparsehash \
        fire \
        tensorboardX \
        pybind11 \
        protobuf \
        pillow \
        && \

# WORKDIR /root
# RUN git clone https://github.com/traveller59/second.pytorch.git --depth 10
# RUN git clone https://github.com/traveller59/SparseConvNet.git --depth 10
# RUN cd ./SparseConvNet && python setup.py install && cd .. && rm -rf SparseConvNet
# ENV NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
# ENV NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
# ENV NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
# ENV PYTHONPATH=/root/second.pytorch
# 
# VOLUME ["/root/data"]
# VOLUME ["/root/model"]
# WORKDIR /root/second.pytorch/second

ENTRYPOINT ["fish"]
