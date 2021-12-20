FROM ubuntu:16.04

LABEL org.opencontainers.image.authors="cnnl@cambricon.com"

ENV NEUWARE_HOME="/usr/local/neuware/"

WORKDIR /tmp

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install --no-install-recommends -y apt-transport-https apt-utils build-essential \
    	git cmake python3 python pkg-config libprotobuf-dev libxml2-dev protobuf-compiler && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


ARG CNTOOLKIT_VERSION="2.5.2-1"
# NOTE: the url below is only for internal usage
ADD http://daily.software.cambricon.com/release/cntoolkit/Linux/x86_64/Ubuntu/16.04/${CNTOOLKIT_VERSION}/cntoolkit_${CNTOOLKIT_VERSION}.ubuntu16.04_amd64.deb /tmp/
# You could replace with your own cntoolkit package

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y /tmp/cntoolkit_${CNTOOLKIT_VERSION}.ubuntu16.04_amd64.deb && \
    apt-get update && \
    apt-get install --no-install-recommends -y cncc cnas cnbin cnrt cndrv && \
    echo "/usr/local/neuware/lib64" > /etc/ld.so.conf.d/neuware.conf && ldconfig && \
    rm -rf /tmp/*.deb && apt-get autoclean && apt-get clean && rm -rf /var/lib/apt/lists/*

