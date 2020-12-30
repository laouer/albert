#!/bin/bash
docker build . -t albert-fr
docker run -it --rm -v $PWD:/media albert-fr bash