#!/bin/bash
#
# The first (optional) argument is the base URL of the LLM server, the second (still optional) is the huggingface repo name. The third argument is the number of parallel jobs to run

base_url=${1:-http://localhost:8000/v1}
model=${2:-meta-llama/Meta-Llama-3.1-8B-Instruct}
N=${3:-5}

# Run on all datasets in the ./datasets/ directory
datasets=`find ./datasets/ -maxdepth 1 -type f -name "*.arff" -exec basename {} \; | sed 's/.arff//g'`

function prepend() {
    while read line;
    do
        echo "[${1}] ${line}"
    done
}

function run_baseline_blr() {
    dataset=$1
    mkdir -p results/baseline/${dataset}
    mkdir -p logs/baseline/${dataset}

    echo "Started running Bayesian Logistic Regression on ${dataset}..."

    python evaluate.py \
        --data-path datasets/${dataset}.arff \
        --samples 4 8 16 32 64 128 256 512 \
        --eval-method holdout \
        --model blr \
        2> logs/baseline/${dataset}/blr.log \
        | tee results/baseline/${dataset}/blr.csv | prepend "${dataset}"

    echo "Finished running Bayesian Logistic Regression on ${dataset}..."
}

function run_llm_blr() {
    mkdir -p results/${model}/${dataset}
    mkdir -p logs/${model}/${dataset}
    mkdir -p prior_cache/${model}

    echo "Started running Bayesian Logisitic Regression with LLM (${model}) Prior on ${dataset}..."

    python evaluate.py \
        --base-url ${base_url} \
        --llm ${model} \
        --data-path datasets/${dataset}.arff \
        --prior-cache prior_cache/${model}/${dataset}.pkl \
        --samples 0 4 8 16 32 64 128 256 512 \
        --prior-samples 128 \
        --eval-method holdout \
        --model blr \
        2> logs/${model}/${dataset}/blr.log \
        | tee results/${model}/${dataset}/blr.csv | prepend "${dataset}"

    echo "Finished running Bayesian Logisitic Regression with LLM (${model}) Prior on ${dataset}..."
}

function run_on_dataset() {
    dataset=$1

    run_llm_blr ${dataset}
    run_baseline_blr ${dataset}
}

for dataset in ${datasets[@]}
do
    run_on_dataset ${dataset} &

    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        wait
    fi
done

wait
