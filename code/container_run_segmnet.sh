#!/bin/bash

PATH_CODE=/root/space/code
PATH_DATA=/root/space/data
PATH_OP=/root/space/output

# Run the torch container
docker run --rm --gpus all --net=host --ipc=host -m 24g -e DISPLAY=$DISPLAY \
    -v ${PWD}/DataPreparation:$PATH_CODE/DataPreparation \
    -v ${PWD}/weakSupervision:$PATH_CODE/weakSupervision \
    -v ${PWD}/utils:$PATH_CODE/utils \
    -v /media/jan/DataStorage/ProjectData/spine_volumes/:$PATH_DATA \
    -v /media/jan/DataStorage/ProjectData/temp/:$PATH_OP \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --memory-swap 1 \
    --env="QT_X11_NO_MITSHM=1" -it jal:torch_thesis bash
