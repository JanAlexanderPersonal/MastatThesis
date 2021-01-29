#!/bin/bash
echo "Start to build jal:pdflatex"
echo "---"
docker build --pull --rm -f "latex_docker" -t jal:pdflatex "." 
echo "--"
echo "Done"
