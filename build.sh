#!/usr/bin/env bash

if [[ ! -d "tritonserver" ]]; then
  mkdir -p tritonserver/include
  cd tritonserver/include
  curl -O https://raw.githubusercontent.com/triton-inference-server/core/main/include/triton/core/tritonserver.h
  cd ../../
fi

if [[ ! -d "pybind11" ]]; then
  git clone -b v2.10.4 https://github.com/pybind/pybind11
fi

rm -rf build
mkdir build && cd build && cmake .. && make
cd ..
