#!/bin/bash
# Build a new image for reiss
echo "clean up docker"
docker system prune -a
echo "Start to build thesis:pdflatex"
echo "---"
docker build --pull --rm -f "Dockerfile" -t jal_pdflatex:pdflatex "." 
echo "--"
echo "Done"
