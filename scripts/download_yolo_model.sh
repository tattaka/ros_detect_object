#!/bin/sh
export FILE_ID=0B4kMaWAXZNSWcUJCVW1aOHV0MkU
export FILE_NAME=yolo.weights
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
export CODE="$(awk '/warning/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}

