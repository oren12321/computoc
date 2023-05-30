FROM amd64/ubuntu:22.10 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        git \
        make \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y --no-install-recommends \
        build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/
RUN apt-get update && apt-get install -y wget \
 && rm -rf /var/lib/apt/lists/* \
 && wget -q -O cmake-linux.sh https://github.com/Kitware/CMake/releases/download/v3.22.3/cmake-3.22.3-Linux-x86_64.sh \
 && sh cmake-linux.sh -- --skip-license --prefix=/usr/local \
 && rm -rf /tmp/*

WORKDIR /tmp/
RUN git clone -b release-1.11.0 https://github.com/google/googletest.git \
 && cd googletest \
 && mkdir build \
 && cd build \
 && cmake .. \
 && make -j$(nproc) \
 && make install \
 && rm -rf /tmp/*

WORKDIR /tmp/
RUN git clone -b v1.6.1 https://github.com/google/benchmark.git \
 && cd benchmark \
 && cmake -E make_directory "build" \
 && cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -S . -B "build" \
 && cmake --build "build" --config Release \
 && cmake -E chdir "build" ctest --build-config Release \
 && cmake --build "build" --config Release --target install \
 && rm -rf /tmp/*

WORKDIR /tmp/
COPY . /tmp/
RUN cmake . -DIN_DOCKER=TRUE -DCMAKE_BUILD_TYPE=Release \
 && make -j$(nproc) \
 && ./tests/erroc/erroc_test \
 && ./tests/enumoc/enumoc_test \
 && ./tests/memoc/memoc_test \
 && ./tests/computoc/computoc_test \
 && ./benchmark/memoc/memoc_benchmark \
 && rm -rf /tmp/*

