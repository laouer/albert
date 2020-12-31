FROM tensorflow/tensorflow:1.15.4

RUN apt-get update -y && apt-get install -y git wget cmake bc gawk curl  lsb-release nano vim

# ENV export GCSFUSE_REPO=
RUN echo "deb http://packages.cloud.google.com/apt gcsfuse-`lsb_release -c -s` main" > /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update -y && apt-get install -y gcsfuse

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN git clone https://github.com/google/sentencepiece.git && \ 
        cd sentencepiece && \ 
        mkdir build && \ 
        cd build && \ 
        cmake .. && \ 
        make -j $(nproc) && \ 
        make install && \ 
        ldconfig -v && \
        cd ../.. && rm -rf sentencepiece

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-319.0.0-linux-x86_64.tar.gz
RUN tar -xvf google-cloud-sdk-319.0.0-linux-x86_64.tar.gz 
RUN bash ./google-cloud-sdk/install.sh -q --path-update true && rm google-cloud-sdk-319.0.0-linux-x86_64.tar.gz

WORKDIR /mnt
CMD ["/bin/bash"]
