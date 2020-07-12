# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvcr.io/nvidia/l4t-pytorch:r32.4.2-pth1.5-py3


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
	time \
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
# PIP for PointPillars
# ------------------------------------------------------------------

RUN PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
    LLVM_CONFIG=/usr/lib/llvm-9/bin/llvm-config \
    $PIP_INSTALL \
        fire \
	numba \
	pillow==7.1.2 \
	protobuf \
        pybind11 \
        scikit-image==0.16.2 \
        shapely \
	sparsehash \
        tensorboardX \
        && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

ENTRYPOINT ["bash"]
