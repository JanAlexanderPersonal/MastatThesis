#!/bin/bash
# Run the torch container
docker run --rm --gpus all --net=host --ipc=host -m 24g -e DISPLAY=$DISPLAY \
    -v ${PWD}/DataPreparation:/root/space/code/DataPreparation \
    -v ${PWD}/weakSupervision:/root/space/code/weakSupervision \
    -v ${PWD}/utils:/root/space/code/utils \
    -v /media/jan/DataStorage/ProjectData/spine_volumes/:/root/space/data/ \
    -v /media/jan/DataStorage/ProjectData/temp/:/root/space/temp/ \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --env="QT_X11_NO_MITSHM=1" -it jal:torch_thesis bash
