#!/bin/bash

read -p "Do you want to build the basic torch image? (y/n)" REPLY
echo $REPLY   # (optional) move to a new line
if [ $REPLY = "y" ]
then
    echo "Start to build jal:torch"
    echo "---"
    docker build --pull --rm -f "docker_pytorch" -t jal:torch "." 
    echo "--"
    echo "Done"
fi

 


read -p "Do you want to build the thesis torch image? (y/n)" REPLY
echo $REPLY # (optional) move to a new line
if [ $REPLY = "y" ]
then
    echo "Start to build jal:torch_wisenet"
    echo "---"
    docker build --pull --rm -f "Dockerfile_pytorch_thesis" -t jal:torch_thesis "." 
    echo "--"
    echo "Done"
fi

