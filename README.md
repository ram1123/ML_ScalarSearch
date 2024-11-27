# Parametrized Multi-Class DNN

## Setup Instructions (First time)

1. Clone the repository:

```bash
git clone -b dev_multiclass git@github.com:ram1123/ML_ScalarSearch.git
cd ML_ScalarSearch
```

2. Load the required software environment:
   ```bash
   . /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
   python -m venv xzz2l2nu_env
   source xzz2l2nu_env/bin/activate
   pip install -r requirements.txt
   chmod +x setup.sh
    ```

## Setup and Run Instructions

```bash
. setup.sh
# command
python scripts/train_multiclass_DNN.py --inputPath </path/to/data> --output_dir </path/to/output> --num_events 1000 --job_name DNN_test_newFW
# Example command
python scripts/train_multiclass_DNN.py --inputPath /eos/user/a/avijay/HZZ_mergedrootfiles/ --output_dir /eos/user/r/rasharma/HZZ2l2nu/  --num_events 1000 --job_name DNN_test_newFW
python scripts/train_multiclass_pNN.py --inputPath /eos/user/a/avijay/HZZ_mergedrootfiles/ --output_dir /eos/user/r/rasharma/HZZ2l2nu/  --num_events 1000 --job_name DNN_test_newFW_pnn
```

4. Condor submission:

```bash
# Generate proxy
voms-proxy-init --voms cms --valid 168:00
cp /tmp/x509up_uXXXXX ~/x509up_uXXXXX
export X509_USER_PROXY=~/x509up_uXXXXX

# Submit the job
python condor/prepare_condor_jobs.py --job_name "test_new" --max_events 1000 --job_flavour "tomorrow"
cd condor/jobs/
condor_submit train_test_new.jdl
```

Example condor scripts are available in the jobs directory:
- [train_test_new.sh](jobs/train_test_new.sh)
- [train_test_new.jdl](jobs/train_test_new.jdl)
