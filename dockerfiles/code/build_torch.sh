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

 
read -p "Do you want to build the Lessmann torch image? (y/n)" REPLY
echo $REPLY # (optional) move to a new line
if [ $REPLY = "y" ]
then
    echo "Start to build jal:torch_lessmann"
    echo "---"
    docker build --pull --rm -f "docker_pytorch_lessmann" -t jal:torch_lessmann "." 
    echo "--"
    echo "Done"
fi

