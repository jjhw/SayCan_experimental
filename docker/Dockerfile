FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04

# Timezone Configuration
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y    apt-utils \
                                    lsb-release \
                                    mesa-utils \
                                    gnupg2 \
                                    net-tools \
                                    iputils-ping \
                                    build-essential \
                                    wget \
                                    unzip \
                                    curl \
                                    git \
                                    mc \
                                    vim \
                                    nano \
                                    ffmpeg \
                                    x11-apps

## GUI tunnel from docker
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxau6 libxau6:i386 \
        libxdmcp6 libxdmcp6:i386 \
        libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 \
        libx11-6 libx11-6:i386 && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.0"


RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglvnd0 libglvnd0:i386 \
        libgl1 libgl1:i386 \
        libglx0 libglx0:i386 \
        libegl1 libegl1:i386 \
        libgles2 libgles2:i386 && \
    rm -rf /var/lib/apt/lists/*
# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt update && apt install -y --allow-change-held-packages libcudnn8

RUN apt update &&  apt install -y make \
                                libssl-dev \
                                zlib1g-dev \
                                libbz2-dev \
                                libreadline-dev \
                                libsqlite3-dev \
                                llvm \
                                libncurses5-dev \
                                libncursesw5-dev \
                                xz-utils \
                                tk-dev \
                                libffi-dev \
                                liblzma-dev \
                                libgdbm-dev \
                                libnss3-dev \
                                libedit-dev \
                                libc6-dev \
                                cmake \
                                unzip \
                                autoconf \
                                autogen \
                                automake \
                                libtool \
                                mlocate

RUN apt update && apt install -y python3-pip python3-dev && pip3 install --upgrade setuptools pip
RUN apt update && apt install -y python3-tk
ENV DEBIAN_FRONTEND=noninteractive

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update && apt install -y ros-noetic-desktop-full
RUN apt update && apt install -y python3-rosdep \
                                python3-rosinstall \
                                python3-rosinstall-generator \
                                python3-wstool \
                                build-essential \
                                ros-noetic-moveit 

RUN pip3 install pybullet \
                gym \
                tqdm \
                git+https://github.com/openai/CLIP.git
RUN pip3 install -U --no-cache-dir gdown --pre
RUN pip3 install tflite-model-maker --no-deps
RUN pip3 install ftfy \
                regex \
                fvcore \
                moviepy \
                flax==0.5.1 \
                openai \
                easydict \
                jax==0.4.8 \
                pycocotools
RUN pip3 install tensorflow==2.12.0 tensorboard
RUN pip3 install opencv-contrib-python
WORKDIR /
RUN pip3 install imageio==2.4.1 imageio-ffmpeg
RUN wget https://github.com/imageio/imageio-binaries/raw/master/ffmpeg/ffmpeg-linux64-v3.3.1 && \
    mkdir /root/.imageio && \
    mv /ffmpeg-linux64-v3.3.1 /root/.imageio/ffmpeg
# RUN apt update && apt install -y --allow-downgrades libgl1-mesa-dri=22.0.1-1ubuntu2 \
#     libglx-mesa0=22.0.1-1ubuntu2 \
#     libgl1-mesa-glx=22.0.1-1ubuntu2 \
#     libglapi-mesa=22.0.1-1ubuntu2 \
#     libgbm1=22.0.1-1ubuntu2 \
#     libegl-mesa0=22.0.1-1ubuntu2 \
#     libgl1-mesa-dri=22.0.1-1ubuntu2 \
#     libglx-mesa0=22.0.1-1ubuntu2 \
WORKDIR /
RUN apt update && apt install -y ros-noetic-realsense2-camera software-properties-common
RUN apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
RUN /usr/bin/apt-add-repository -y "https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
RUN apt update && apt install -y librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg && \
    pip3 install pyrealsense2
RUN apt update && apt install -y libboost-system1.67.0 libboost-thread1.67.0 libmongoclient0 ros-noetic-warehouse-ros-mongo \
                                libnlopt-dev libnlopt0 ros-noetic-trac-ik-lib \
                                ros-noetic-ur-client-library \
                                ros-noetic-cartesian-control-msgs ros-noetic-cartesian-interface ros-noetic-pass-through-controllers ros-noetic-speed-scaling-interface \
                                ros-noetic-joint-trajectory-action ros-noetic-joint-trajectory-action-tools \
                                ros-noetic-ur-msgs ros-noetic-trac-ik-kinematics-plugin ros-noetic-effort-controllers ros-noetic-speed-scaling-state-controller \
                                ros-noetic-urdf-parser-plugin ros-noetic-urdf-tutorial ros-noetic-scaled-joint-trajectory-controller \
                                ros-noetic-cartesian-trajectory-controller ros-noetic-cartesian-trajectory-interpolation \
                                ros-noetic-industrial-msgs ros-noetic-industrial-robot-status-controller ros-noetic-force-torque-sensor-controller socat ros-noetic-twist-controller \
                                ros-noetic-industrial-robot-status-interface ros-noetic-industrial-robot-simulator ros-noetic-velocity-controllers

ENV PYTHONPATH "${PYTHONPATH}:/workspace/ros_ws/src/alpaca_connector/"
RUN /bin/bash -c 'echo "source /opt/ros/noetic/setup.bash; source /workspace/ros_ws/devel/setup.bash --extend"  >> ~/.bashrc'
WORKDIR /workspace
ENTRYPOINT /bin/bash -c 'source /opt/ros/noetic/setup.bash && cd /workspace/ros_ws && catkin_make && source /workspace/ros_ws/devel/setup.bash --extend && cd .. && /bin/bash'