#!/bin/bash

# Load the LCG environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

# Activate the created virtual environment
source xzz2l2nu_env/bin/activate

export PYTHONPATH=$(pwd):$(pwd)/scripts:$(pwd)/plotting:$PYTHONPATH

# Allow unlimited stack size for large jobs
ulimit -s unlimited
