#!/bin/bash
set -eux

echo ">>> Start of post install script <<<"

git config --global --add safe.directory "*"

DISPLAY_EXPORT="export DISPLAY=:0"

conda activate go-slam
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install evo --upgrade --no-binary evo
cd go-slam
python setup.py install

# pip install torch-scatter PyMCubes -> now inside of environment.yaml

echo ">>> End of post install script <<<"
