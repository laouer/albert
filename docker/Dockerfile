FROM tensorflow/tensorflow:1.15.4
RUN apt-get update -y && apt-get install -y git wget cmake
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

CMD ['/bin/bash']
