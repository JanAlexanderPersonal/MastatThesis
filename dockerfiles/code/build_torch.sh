#!/bin/bash
echo "Start to build jal:pdflatex"
echo "---"
docker build --pull --rm -f "docker_pytorch" -t jal:torch "." 
echo "--"
echo "Done"
