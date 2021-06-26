#!/bin/bash
# run the latest pdf container

# python PlotNeuralNets/unet.py 
python PlotNeuralNets/vgg16_upscore.py 
python PlotNeuralNets/resnet.py

docker run --rm -it -v ${PWD}:/home/thesis/ jal:pdflatex

