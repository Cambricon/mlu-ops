FROM ubuntu:20.04

LABEL org.opencontainers.image.authors="www.cambricon.com"

ENV NEUWARE_HOME="/usr/local/neuware/"

WORKDIR /tmp

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install --no-install-recommends -y apt-utils build-essential lcov html2text wget sshpass \
    	git cmake python3 python pkg-config libprotobuf-dev libxml2-dev protobuf-compiler vim libreadline-dev clang-format \
        debhelper texinfo lsb-release patchutils diffstat python-dev swig python-six openssh-client openssh-server ca-certificates sudo unzip zsh python3-pip less && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


ARG CNTOOLKIT_VERSION="3.7.2-1"
ENV CNTOOLKIT_VERSION ${CNTOOLKIT_VERSION}
# NOTE: you could replace with your own cntoolkit package.
ADD https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu20.04/cntoolkit_${CNTOOLKIT_VERSION}.ubuntu20.04_amd64.deb /tmp/

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y /tmp/cntoolkit_${CNTOOLKIT_VERSION}.ubuntu20.04_amd64.deb && \
    apt-get update && \
    apt-get install --no-install-recommends -y cncc cnas cnbin cnrt cndrv cndev cnrtc cngdb cnperf && \
    echo "/usr/local/neuware/lib64" > /etc/ld.so.conf.d/neuware.conf && ldconfig && \
    rm -rf /tmp/*.deb && apt-get autoclean && apt-get clean && rm -rf /var/lib/apt/lists/*


ARG CNNL_VERSION="1.21.1-1"
ENV CNNL_VERSION ${CNNL_VERSION}
# NOTE: you could replace with your own cntoolkit package.
ADD https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu20.04/cnnl_${CNNL_VERSION}.ubuntu20.04_amd64.deb /tmp/

RUN set -ex && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y /tmp/cnnl_${CNNL_VERSION}.ubuntu20.04_amd64.deb && \
    ldconfig && rm -rf /tmp/*.deb && \
    apt-get autoclean && apt-get clean

RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar -zxvf eigen-3.4.0.tar.gz && \
    cd eigen-3.4.0 && mkdir build && cd build && \
    cmake ../ && \
    make install && \
    cd ../.. && \
    rm -r eigen-3.4.0 eigen-3.4.0.tar.gz

RUN pip3 install cpplint==1.6.0

ENV PATH="${NEUWARE_HOME}/bin":${PATH}
ENV LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64":${LD_LIBRARY_PATH}
CMD /bin/bash
