FROM ubuntu:18.04

LABEL org.opencontainers.image.authors="www.cambricon.com"

ENV NEUWARE_HOME="/usr/local/neuware/"

WORKDIR /tmp

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install --no-install-recommends -y apt-transport-https apt-utils build-essential lcov html2text\
    	git cmake python3 python pkg-config libprotobuf-dev libxml2-dev protobuf-compiler vim libreadline-dev clang-format&& \
    apt-get clean && rm -rf /var/lib/apt/lists/*


ARG CNTOOLKIT_VERSION="3.3.0-1"
# NOTE: the url below is only for internal usage, you could replace with your own cntoolkit package.
ADD http://daily.software.cambricon.com/release/cntoolkit/Linux/x86_64/Ubuntu/18.04/${CNTOOLKIT_VERSION}/cntoolkit_${CNTOOLKIT_VERSION}.ubuntu18.04_amd64.deb /tmp/

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y /tmp/cntoolkit_${CNTOOLKIT_VERSION}.ubuntu18.04_amd64.deb && \
    apt-get update && \
    apt-get install --no-install-recommends -y cncc cnas cnbin cnrt cndrv cndev cnrtc cngdb cnperf && \
    echo "/usr/local/neuware/lib64" > /etc/ld.so.conf.d/neuware.conf && ldconfig && \
    rm -rf /tmp/*.deb && apt-get autoclean && apt-get clean && rm -rf /var/lib/apt/lists/*


# Install BANGPy depended-upon software.
# Install llvm-6.0.
RUN apt-get update && \
    apt-get install -y llvm-6.0-dev

# Install python3.8.0.
# NOTE: the url below is only for internal usage, you could replace with your own python-3.8.0 package.
ADD http://daily.software.cambricon.com/download/python/3.8.0/Python-3.8.0.tgz /tmp/
ADD http://daily.software.cambricon.com/download/python/3.8.0/_sqlite3.cpython-38-x86_64-linux-gnu.so /tmp/
ADD http://daily.software.cambricon.com/download/openssl/openssl-1.1.1p.tar.gz /tmp/
RUN tar -xzvf /tmp/openssl-1.1.1p.tar.gz -C /tmp/ && \
    cd /tmp/openssl-1.1.1p && \
    ./config --prefix=/usr/local/openssl && \
    make -j24 && make install && \
    ln -s /usr/local/openssl/bin/openssl /usr/bin/openssl && \
    ln -s /usr/local/openssl/include/openssl /usr/include/openssl && \
    echo "/usr/local/openssl/lib" >> /etc/ld.so.conf && \
    ldconfig && \
    rm /tmp/openssl-1.1.1p* -rf
RUN tar -xvf /tmp/Python-3.8.0.tgz -C /tmp/ && \
    cd /tmp/Python-3.8.0 && \
    ./configure --with-openssl=/usr/local/openssl --enable-shared --enable-optimizations && \
    make -j24 && \
    make install && \
    cp /usr/local/lib/libpython3.8.so.1.0 /usr/lib/ && \
    cd - && \
    rm /tmp/Python-3.8.0* -rf

RUN apt-get update && \
    apt-get install -y ca-certificates

# Install python requirements.
RUN pip3.8 install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --upgrade pip && \
    pip3.8 install                     \
    appdirs==1.4.4                     \
    astroid==2.6.5                     \
    backcall==0.2.0                    \
    black==19.10b0                     \
    click==8.0.1                       \
    cpplint==1.5.5                     \
    cycler==0.10.0                     \
    cloudpickle                        \
    decorator==4.4.2                   \
    flake8==3.9.2                      \
    flatbuffers==1.12                  \
    graphviz==0.14.1                   \
    iniconfig==1.0.1                   \
    ipython==7.18.1                    \
    ipython-genutils==0.2.0            \
    isort==5.5.0                       \
    jedi==0.17.2                       \
    kiwisolver==1.2.0                  \
    lazy-object-proxy==1.5.1           \
    matplotlib==3.2.1                  \
    mccabe==0.6.1                      \
    more-itertools==8.5.0              \
    numpy==1.21.2                      \
    onnxruntime==1.4.0                 \
    packaging==21.0                    \
    parso==0.7.1                       \
    pathspec==0.8.0                    \
    pexpect==4.8.0                     \
    pickleshare==0.7.5                 \
    pip==19.2.3                        \
    plint==0.1                         \
    pluggy==0.13.1                     \
    prompt-toolkit==3.0.7              \
    protobuf==3.13.0                   \
    psutil==5.7.0                      \
    ptyprocess==0.6.0                  \
    py==1.9.0                          \
    pycodestyle==2.7.0                 \
    pyflakes==2.3.1                    \
    Pygments==2.9.0                    \
    pylint==2.9.5                      \
    pyparsing==2.4.7                   \
    pytest==6.0.2                      \
    python-dateutil==2.8.1             \
    regex==2021.7.6                    \
    scipy==1.4.1                       \
    setuptools==41.2.0                 \
    six==1.15.0                        \
    toml==0.10.1                       \
    traitlets==5.0.4                   \
    typed-ast==1.4.1                   \
    wcwidth==0.2.5                     \
    wrapt==1.12.1                      \
    xgboost==0.7.post3                 \
    prototxt_parser==1.0               \
    -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && \
    cp /tmp/_sqlite3.cpython-38-x86_64-linux-gnu.so /usr/local/lib/python3.8/site-packages/ && \
    rm /tmp/_sqlite3.cpython-38-x86_64-linux-gnu.so

# Install BANGPy.
ARG BANGPY_VERSION="2.2.0"
ARG RELEASE_VERSION="2.2.0-1"
# NOTE: the url below is only for internal usage,you could replace with your own BANGPy package.
ADD http://daily.software.cambricon.com/daily/bangpy/wheel/2.2.0/20221212-185904-6cd05e4/bangpy-2.2.0-py3-none-any.whl /tmp/
RUN pip3.8 install /tmp/bangpy-${BANGPY_VERSION}-py3-none-any.whl && \
    rm /tmp/bangpy-${BANGPY_VERSION}-py3-none-any.whl
