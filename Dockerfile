FROM ubuntu:focal

# Docker image for CHiLL
#
# Note:
#   This image simply download the latest versions of
#   ROSE, IEGenLib and CHiLL.
#   We don't do any version pinning, etc.
#
# Build using:
#   docker build -t 'chill:latest' .
#
# Usage:
#   # Print help
#   docker run --rm chill:latest
#
#   # Directly call `chill` with `fuse_distribute.script.py` that exists in `$(pwd)`.
#   docker run --rm -v $(pwd):/opt/project chill:latest chill fuse_distribute.script.py
#
#   # Run in interactive mode
#   docker run --rm -it -v $(pwd):/opt/project chill:latest /bin/bash
#     # In the container
#     chill fuse_distribute.script.py

RUN apt-get update && apt-get upgrade -y

# Install ROSE from their PPA (i.e. their repository) and
# also some other dependencies.
RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:rosecompiler/rose-stable && \
    apt-get update && \
    apt-get install -y git \
                        make \
                        g++ \
                        gcc \
                        python2 \
                        python2-dev \
                        python3-pip \
                        rose \
                        libgmp-dev \
                        texinfo && \
    apt-get autoremove
RUN pip install cmake

# ROSE from their PPA installs to this directory:
ENV ROSEHOME="/usr/rose"

# Install IEGenLib
RUN git clone https://github.com/CompOpt4Apps/IEGenLib.git /opt/IEGenLib && \
    cd /opt/IEGenLib && \
    ./configure && \
    make -j$(nproc) && \
    make install
ENV IEGENHOME="/opt/IEGenLib/iegen"

# Install ISL
# We use the version from IEGenLib/lib/isl.  IEGenLib does not
# install isl's headers, so we simply reconfigure the directory
# and then compile and install isl.
RUN cd /opt/IEGenLib/lib/isl && \
    ./configure && \
    make -j$(nproc) && \
    make install

ENV LD_LIBRARY_PATH="/lib:/usr/lib:/usr/local/lib"

# Install CHiLL
# We use the latest version from GitHub and build it in debug mode.
# There seem to be issues with the Release mode.
# We suppress the noisy `-Wreturn-type` warning.  This needs to be fixed, though.
RUN git clone https://github.com/CtopCsUtahEdu/chill.git /opt/chill && \
    cd /opt/chill && \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_FLAGS="-Wno-return-type" \
        && \
    cmake --build build --config=Debug && \
    cmake --build build --config=Debug --target=install && \
    cd /opt && \
    rm -rf chill

VOLUME [ "/opt/project" ]

WORKDIR /opt/project

CMD [ "chill", "--help" ]
