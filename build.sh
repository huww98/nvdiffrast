#!/bin/bash

docker run --name nvdiffrast_builder -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64 /io/build_wheel.sh