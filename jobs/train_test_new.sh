#!/bin/bash
ulimit -s unlimited
set -e

# Navigate to the project directory
cd /afs/cern.ch/work/r/rasharma/h2l2nu/ML/ML_ScalarSearch

# Activate pre-configured environment
. setup.sh

# Run the training
name='test_new'
python scripts/train_multiclass_DNN.py --inputPath /eos/user/a/avijay/HZZ_mergedrootfiles/ --output_dir /eos/user/r/rasharma/HZZ2l2nu/  --num_events 1000 --job_name test_new --json ./data/input_variables.json

echo "Training Done"

# Copy the output to eos
# echo "Copying the output to eos"
# cp -r HHWWBBDNN_binary_${name}_BalanceYields /eos/user/r/rasharma/HZZ2l2nu/
# echo "Output copied to eos"
# ls /eos/user/r/rasharma/HZZ2l2nu/
# echo "All Done"
