#!/bin/bash

expDir=$1
model=${2:-meta-llama/Meta-Llama-3.1-8B-Instruct}

# Run on all datasets in the ./datasets/ directory
datasets=`find ./datasets/ -maxdepth 1 -type f -name "*.arff" -exec basename {} \; | sed 's/.arff//g'`

mkdir -p figure-data/sample-efficiency

for dataset in $datasets; do
    tsv-summarize --header -d , --group-by num_train --mean roc_auc --stdev roc_auc $expDir/results/baseline/${dataset}/blr.csv > figure-data/sample-efficiency/${dataset}-blr.csv
    tsv-summarize --header -d , --group-by num_train --mean roc_auc --stdev roc_auc $expDir/results/baseline/${dataset}/bnn.csv > figure-data/sample-efficiency/${dataset}-bnn.csv
    tsv-summarize --header -d , --group-by num_train --mean roc_auc --stdev roc_auc $expDir/results/${model}/${dataset}/blr.csv > figure-data/sample-efficiency/${dataset}-blr-llm.csv
    tsv-summarize --header -d , --group-by num_train --mean roc_auc --stdev roc_auc $expDir/results/${model}/${dataset}/bnn.csv > figure-data/sample-efficiency/${dataset}-bnn-llm.csv
done