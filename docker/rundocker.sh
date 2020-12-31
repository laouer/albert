#!/bin/bash
docker build . -t albert-fr
docker run -it --privileged -v $PWD:/mnt albert-fr bash