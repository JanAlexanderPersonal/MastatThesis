#!/bin/bash
# Run the torch container
docker run --rm --gpus all --net=host --ipc=host -m 24g -e DISPLAY=$DISPLAY \
    -v ${PWD}:/root/space/code \
    -v /media/jan/DataStorage/ProjectData/spine_volumes/:/root/space/data/ \
    -v /media/jan/DataStorage/ProjectData/temp/:/root/space/temp/ \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --env="QT_X11_NO_MITSHM=1" -it jal:torch_thesis bash
