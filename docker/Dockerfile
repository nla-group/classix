# This is just demo for docker image build for classix, be free to modify it.
# Author: Stefan Güttel and Xinye Chen

FROM xnla/ubuntu:py
LABEL description="CLASSIX: Fast and explainable clustering based on sorting"

COPY requirements.txt /root
WORKDIR /root
RUN pip install Cython \
    && pip install -r requirements.txt 
EXPOSE 8888/tcp
ENV SHELL /bin/bash