#!/bin/bash

xhost +local:docker || true

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
docker run -ti --rm \
      --gpus all \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e "DISPLAY" \
      -e "QT_X11_NO_MITSHM=1" \
      -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
      -e XAUTHORITY \
      -e OPENAI_KEY="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" \
      -v $ROOT_DIR:/workspace \
      -v $ROOT_DIR/cache:/root/.cache \
      --net=host \
      --privileged \
      --name rt1_exp rt1-img \
      bash