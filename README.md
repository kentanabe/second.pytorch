# PointPillars

Welcome to PointPillars.

This repo demonstrates how to reproduce the results from
[_PointPillars: Fast Encoders for Object Detection from Point Clouds_](https://arxiv.org/abs/1812.05784) (to be published at CVPR 2019) on the
[KITTI dataset](http://www.cvlibs.net/datasets/kitti/) by making the minimum required changes from the preexisting
open source codebase [SECOND](https://github.com/traveller59/second.pytorch). 

This is not an official nuTonomy codebase, but it can be used to match the published PointPillars results.

**WARNING: This code is not being actively maintained. This code can be used to reproduce the results in the first version of the paper, https://arxiv.org/abs/1812.05784v1. For an actively maintained repository that can also reproduce PointPillars results on nuScenes, we recommend using [SECOND](https://github.com/traveller59/second.pytorch). We are not the owners of the repository, but we have worked with the author and endorse his code.**

![Example Results](https://raw.githubusercontent.com/nutonomy/second.pytorch/master/images/pointpillars_kitti_results.png)


## Getting Started

This is a fork of [SECOND for KITTI object detection](https://github.com/traveller59/second.pytorch) and the relevant
subset of the original README is reproduced here.

### Docker Environments

If you do not waste time on pointpillars envs, please pull my docker virtual environments :

#### Jetson Xavier NX

```bash
docker pull tanabeken/public:pointpillars.ubuntu18.04.arm64.cuda10_2.jetson-xaviernx.20200715
```

#### AMD64(x86_64) PC with NVIDIA GPU

T.B.D.

### Code Support

ONLY supports python 3.7, pytorch 1.5.0. Code has only been tested on Ubuntu 18.04.

### Install

#### 1. Clone code

```bash
git clone https://github.com/nutonomy/second.pytorch.git
```

#### 2. Install Python packages

It is recommend to use the Anaconda package manager.

First, install packages via apt.
```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
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
	time
sudo apt-get install -y --no-install-recommends \
	software-properties-common
sudo apt-get install -y --no-install-recommends \
	software-properties-common
        python3 \
        python3-dev \
        python3-pip \
	libfreetype6-dev \
	libpng-dev
```

Then use pip for python packages.
```bash
python3 -m pip --no-cache-dir install --upgrade --user \
	-U pip
python3 -m pip --no-cache-dir install --upgrade --user \
	setuptools
python3 -m pip --no-cache-dir install --upgrade --user \
	numpy \
        scipy \
        matplotlib \
	Cython
LLVM_CONFIG=/usr/lib/llvm-9/bin/llvm-config \
	python3 -m pip --no-cache-dir install --upgrade --user \
	        fire \
		numba==0.43.1 \
		pillow==7.1.2 \
		protobuf \
	        pybind11 \
	        scikit-image==0.16.2 \
	        shapely \
		sparsehash \
	        tensorboardX
```

Finally, install SparseConvNet. This is not required for PointPillars, but the general SECOND code base expects this
to be correctly configured. 
```bash
git clone https://github.com/kentanabe/SparseConvNet.git
cd ./SparseConvNet
git pull
git checkout a0b9bbbbee335eeb98aba55377b8835bfde1fb26
python3 setup.py install
cd ..
rm -rf SparseConvNet
```

#### 4. PYTHONPATH

Add second.pytorch/ to your PYTHONPATH.
```bash
export PYTHONPATH=${HOME}/second.pytorch:${PYTHONPATH}
```


### Prepare dataset

#### 1. Dataset preparation

Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```

Note: PointPillar's protos use ```KITTI_DATASET_ROOT=/data/sets/kitti_second/```.

```bash
export KITTI_DATASET_ROOT=/data/sets/kitti_second/
```

#### 2. Create kitti infos:

```bash
python3 create_data.py create_kitti_info_file --data_path=${KITTI_DATASET_ROOT}
```

#### 3. Create reduced point cloud:

```bash
python3 create_data.py create_reduced_point_cloud --data_path=${KITTI_DATASET_ROOT}
```

#### 4. Create groundtruth-database infos:

```bash
python3 create_data.py create_groundtruth_database --data_path=${KITTI_DATASET_ROOT}
```

#### 5. Modify config file

The config file needs to be edited to point to the above datasets:

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
```


### Train

```bash
cd ~/second.pytorch/second
python3 -W ignore::UserWarning ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/path/to/model_dir
```

* If you want to train a new model, make sure "/path/to/model_dir" doesn't exist.
* If "/path/to/model_dir" does exist, training will be resumed from the last checkpoint.
* Training only supports a single GPU. 
* Training uses a batchsize=2 which should fit in memory on most standard GPUs.
* On a single 1080Ti, training xyres_16 requires approximately 20 hours for 160 epochs.


### Evaluate


```bash
cd ~/second.pytorch/second/
python3 -W ignore::UserWarning pytorch/train.py evaluate --config_path= configs/pointpillars/car/xyres_16.proto --model_dir=/path/to/model_dir
```

* Detection result will saved in model_dir/eval_results/step_xxx.
* By default, results are stored as a result.pkl file. To save as official KITTI label format use --pickle_result=False.

### Build Docker image

#### 1. Set nvidia docker runtime as default to use cuda tools from docker build.

```
*** daemon.json~        2018-03-07 13:06:38.000000000 +0900
--- daemon.json 2020-06-28 14:55:33.341543078 +0900
***************
*** 1,4 ****
--- 1,5 ----
  {
+     "default-runtime": "nvidia",
      "runtimes": {
          "nvidia": {
              "path": "nvidia-container-runtime",
```

```bash
sudo service docker restart
```

#### 2. Build docker image

##### Jetson Xavier NX

```bash
docker build -t <TAG NAME> docker/pointpillars.18.04.arm64.cuda10_2/
```
##### AMD64(x86_64) PC with NVIDIA GPU

```bash
docker build -t <TAG NAME> docker/pointpillars.18.04.amd64.cuda10_2/
```

#### 3. Add swap for Jetson Xavier NX

```bash
sudo dd if=/dev/zero of=/swap bs=1G count=8
sudo chmod 600 /swap
sudo mkswap /swap
sudo swapon /swap
```

##### Modify fstab

```bash
sudo sh -c "echo '/swap none swap sw 0 0' >> /etc/fstab"
```

##### Disable swappiness

```bash
sudo sh -c "echo 'vm.swappiness=0' >> /etc/sysctl.conf"
sudo sysctl -p
```

#### 4. Run docker image

```bash
docker run --gpus all --network host \
    -it -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v <KITTI_DATASET_ROOT>:/data/sets/kitti_second/ \
    -v ${HOME}/pointpillars/model:/root/model \
    <TAG NAME>
```
