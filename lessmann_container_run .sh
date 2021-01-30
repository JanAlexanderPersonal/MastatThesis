#!/bin/bash
# Run the torch container
docker run --gpus all --net=host --ipc=host -m 24g -e DISPLAY=$DISPLAY -v ${PWD}/code:/root/space/ -v ${PWD}/Data/Zenodo/:/root/space/data -v /tmp/.X11-unix:/tmp/.X11-unix --env="QT_X11_NO_MITSHM=1" -it jal:torch_lessmann bash