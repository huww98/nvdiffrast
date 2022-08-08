#!/bin/bash
set -e

PYTHON_VERSION=3.7
TORCH_VERSION=1.9.0+cu102
CUDA_VERSION=10.2
GCC_VERSION=8
export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX"

yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
YUM_CUDA_VERSION=${CUDA_VERSION/./-}
yum install -y \
    *cusparse-dev*-$YUM_CUDA_VERSION \
    *cublas-dev*-$YUM_CUDA_VERSION \
    *cusolver-dev*-$YUM_CUDA_VERSION \
    cuda-cudart-dev*-$YUM_CUDA_VERSION \
    cuda-compiler-$YUM_CUDA_VERSION \
    ninja-build \
    devtoolset-$GCC_VERSION-gcc-c++ \
    mesa-libEGL-devel
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}

source /opt/rh/devtoolset-8/enable
which c++

PYBIN=(/opt/python/cp${PYTHON_VERSION/./}-*/bin)
"${PYBIN}/pip" install torch==$TORCH_VERSION -f https://download.pytorch.org/whl/torch_stable.html

rm /tmp/wheelhouse/*.whl
export MAX_JOBS=4
"${PYBIN}/pip" wheel /io --verbose --no-deps -w /tmp/wheelhouse/

if [[ "$(auditwheel repair --help)" != *"--exclude"* ]]; then
    echo "patching auditwheel repair"
    patch -d /opt/_internal/pipx/venvs/auditwheel/lib/python3.9/site-packages/auditwheel -p3 < /io/auditwheel.patch
fi

# Bundle external shared libraries into the wheels
for whl in /tmp/wheelhouse/*.whl; do
    auditwheel repair "$whl" \
        --strip \
        --exclude libtorch_cuda.so \
        --exclude libtorch_cpu.so \
        --exclude libtorch_python.so \
        --exclude libtorch.so \
        --exclude libc10_cuda.so \
        --exclude libc10.so \
        --exclude libcudart.so.${CUDA_VERSION} \
        -w /io/wheelhouse/
done
