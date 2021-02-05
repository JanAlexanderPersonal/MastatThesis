#!/bin/bash
# Run the torch container
docker run --rm --gpus all --net=host --ipc=host -m 24g -e DISPLAY=$DISPLAY -v ${PWD}/code/U_Net_Lessmann:/root/space/U_Net_Lessmann -v /media/jan/DataStorage1/ProjectData/spine_volumes/xVertSeg/Data1:/root/space/data/ -v /tmp/.X11-unix:/tmp/.X11-unix --env="QT_X11_NO_MITSHM=1" -it jal:torch_lessmann bash
