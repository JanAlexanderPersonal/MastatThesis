#!/bin/bash
# clean up idle docker images & containers
echo "clean up docker"
docker system prune -a