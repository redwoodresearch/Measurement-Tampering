#!/bin/bash

DATA_DIR=~/code_elk_setting/extended
CODE_DIR=~/unity/elk/func_correct
RRFS_DIR=~/rrfs/code_elk_setting/extended
LOCAL_BACKUP_FOLDER=~/code_elk_setting/extended/backup_v1
RRFS_BACKUP_FOLDER=~/rrfs/code_elk_setting/extended/backup_v1

PROBLEM_GENERATION_METHOD=extended
ISSUE_GENERATION_METHOD=extended

DO_INSTALL=0
DO_BACKUP_TO_RRFS=0
DO_EXTRACTION=0
DO_API=0
DO_TRAIN_GENERATION=0
DO_TRAIN=0
DO_COPY_TO_RRFS=0

# make dir if not exist
mkdir -p $DATA_DIR
mkdir -p $RRFS_DIR
mkdir -p $LOCAL_BACKUP_FOLDER
mkdir -p $RRFS_BACKUP_FOLDER

files=("cache_filan_datum.json" "raw_functions_v3.jsonl" "non_overlapping_val_data.jsonl" "overlapping_val_data.jsonl" "train_data.jsonl" "loaded_problem_extra_v2.jsonl" "raw_functions_v3_with_generated_answers.jsonl" "responses_code_for_issues.jsonl" "extended_functions_solved.jsonl" "mbpp_easy_closure.jsonl")

# installation for docker

if [ $DO_INSTALL -eq 1 ]; then
    (
        set -e
        ARCH=$(uname -m)
        URL=https://storage.googleapis.com/gvisor/releases/release/20230102/${ARCH}
        wget ${URL}/runsc ${URL}/runsc.sha512 \
            ${URL}/containerd-shim-runsc-v1 ${URL}/containerd-shim-runsc-v1.sha512
        sha512sum -c runsc.sha512 \
            -c containerd-shim-runsc-v1.sha512
        rm -f *.sha512
        chmod a+rx runsc containerd-shim-runsc-v1
        sudo mv runsc containerd-shim-runsc-v1 /usr/local/bin
    )
    sudo /usr/local/bin/runsc install
    sudo systemctl reload docker
fi

# setup config by writing to config.json
cd $CODE_DIR
echo "{\"data_dir\": \"$DATA_DIR\", \"rrfs_dir\": \"$RRFS_DIR\", \"issue_generation_method\": \"$ISSUE_GENERATION_METHOD\"}" > config.json

# backup caches locally and on rrfs
cd $DATA_DIR
for file in "${files[@]}"; do
    cp $file $LOCAL_BACKUP_FOLDER/$file
    if [ $DO_BACKUP_TO_RRFS -eq 1 ]; then
        echo "copying $file to $RRFS_BACKUP_FOLDER/$file"
        cp $file $RRFS_BACKUP_FOLDER/$file
    fi
done
# generation function data
cd $CODE_DIR
if [ $DO_EXTRACTION -eq 1 ]; then
    python raw_datasets_to_jsonl.py # 20 minutes, -> raw_functions_v3.jsonl
fi
# generation model data
if [ $DO_API -eq 1 ]; then
    if [ $PROBLEM_GENERATION_METHOD == "extended" ]; then
        python generate_more_programs.py # >1 hour -> extended_functions.jsonl (mbpp_easy_closure.jsonl)
        python generate_solutions.py --which_functions extended_functions --model gpt-4 --temperature 0 --add_as_solution --no_asserts # 45 minutes
        # -> extended_functions_with_solutions.jsonl
        python compute_tests_results.py
        # -> OVERWRITE raw_functions_v3.jsonl
    fi

    python generate_issues.py # 1h minutes -> responses_code_for_issues.jsonl
    python generate_solutions.py --nb_solutions 3 # 45 minutes -> raw_functions_v3_with_generated_answers.jsonl
fi
# generate training data (requires docker and gvisor)
if [ $DO_TRAIN_GENERATION -eq 1 ]; then
    python setup_loaded_problem_extra.py # 10 minutes -> loaded_problem_extra_v2.jsonl
    python compute_all_correctness.py # 40 minutes -> cache_filan_datum.json
    python loaded_problem_to_dataset_example.py # 5 minutes -> train_data.jsonl, overlapping_val_data.jsonl, non_overlapping_val_data.jsonl
fi
# training
# if [ $DO_TRAIN -eq 1 ]; then
#     # ... python train.py
# fi
# save to rrfs
if [ $DO_COPY_TO_RRFS -eq 1 ]; then
    cd $DATA_DIR
    for file in "${files[@]}"; do
        echo "copying $file to $RRFS_DIR/$file"
        cp $file $RRFS_DIR/$file
    done
fi
cd $CODE_DIR