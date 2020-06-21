# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvcr.io/nvidia/l4t-ml:r32.4.2-py3

# ==================================================================
# tools
# ------------------------------------------------------------------
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        wget \
	curl \
        git \
	bash \
        vim \
        libsparsehash-dev \
        make \
        llvm-9-dev \
        libboost-all-dev \
        libgeos-dev \
        && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
# ==================================================================
# python
# ------------------------------------------------------------------
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3 \
        python3-dev \
        python3-pip \
	libfreetype6-dev \
	libpng-dev \
        && \
    $PIP_INSTALL \
        -U pip \
	&& \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        matplotlib \
	Cython \
        && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# ==================================================================
# VS Code
# ------------------------------------------------------------------
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libnotify4 \
        libsecret-1-0 \
        libxss1 \
        && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
RUN curl -L https://github.com/toolboc/vscode/releases/download/1.32.3/code-oss_1.32.3-arm64.deb \
        -o code-oss_1.32.3-arm64.deb && \
    dpkg -i code-oss_1.32.3-arm64.deb
	
# ==================================================================
# PointPillars
# ------------------------------------------------------------------

RUN PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
    LLVM_CONFIG=/usr/lib/llvm-9/bin/llvm-config \
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
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

WORKDIR /root
RUN git clone https://github.com/traveller59/second.pytorch.git --depth 10
RUN git clone https://github.com/traveller59/SparseConvNet.git --depth 10 && \
    pushd ./SparseConvNet && \
    python setup.py install && \
    popd && \
    rm -rf SparseConvNet
ENV NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
ENV NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
ENV NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
ENV PYTHONPATH=/root/second.pytorch
 
VOLUME ["/root/data"]
VOLUME ["/root/model"]
WORKDIR /root/second.pytorch/second

ENTRYPOINT ["bash"]
