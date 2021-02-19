#!/bin/bash
# clean up idle docker images & containers
echo "clean up docker"
docker container stop $(docker container ls -aq)
docker system prune -a