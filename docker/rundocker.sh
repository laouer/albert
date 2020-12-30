#!/bin/bash
docker build . -t albert-fr
docker run -it --rm -v $PWD:/mnt albert-fr bash