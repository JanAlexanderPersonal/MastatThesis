#!/bin/bash
# run the latest pdf container

docker run --rm -it -v ${PWD}:/home/thesis/ jal:pdflatex

