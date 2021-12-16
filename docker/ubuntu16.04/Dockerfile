FROM ubuntu:16.04

ARG CNTOOLKIT_VERSION="2.5.2-1"
ENV maintainer="cnnl@cambricon.com"
ENV NEUWARE_HOME="/usr/local/neuware/"

WORKDIR /tmp

RUN apt update && apt install -y wget

# NOTE: the url below is only for internal usage
RUN wget http://daily.software.cambricon.com/release/cntoolkit/Linux/x86_64/Ubuntu/16.04/${CNTOOLKIT_VERSION}/cntoolkit_${CNTOOLKIT_VERSION}.ubuntu16.04_amd64.deb
# You could use COPY with your own cntoolkit package

RUN apt install -y git apt-transport-https && \
    dpkg -i cntoolkit_${CNTOOLKIT_VERSION}.ubuntu16.04_amd64.deb && \
    apt update && \
    apt install --no-install-recommends -y cncc cnas cnbin cnrt cndrv build-essential cmake python3 python pkg-config libprotobuf-dev libxml2-dev protobuf-compiler && \
    echo "/usr/local/neuware/lib64" > /etc/ld.so.conf.d/neuware.conf && ldconfig && \
    rm -rf /tmp/*.deb && apt-get autoclean && apt-get clean && rm -rf /var/lib/apt/lists/*

