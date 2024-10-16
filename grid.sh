#!/bin/bash

# Check if doc_pool is provided as a command-line argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <doc_pool>"
    exit 1
fi

# Use the provided doc_pool
doc_pool=2

# Arrays for other environment variables
VIDORE_QUERY_POOL=(0.01 0.02 0.03)
VIDORE_N_ANN=(40 80)
n_maxsim=$1

# Loop through all combinations
for query_pool in "${VIDORE_QUERY_POOL[@]}"; do
    for n_ann in "${VIDORE_N_ANN[@]}"; do
            echo "VIDORE_DOC_POOL=$doc_pool, VIDORE_QUERY_POOL=$query_pool, VIDORE_N_ANN=$n_ann, VIDORE_N_MAXSIM=$n_maxsim"
            VIDORE_DOC_POOL=$doc_pool VIDORE_QUERY_POOL=$query_pool VIDORE_N_ANN=$n_ann VIDORE_N_MAXSIM=$n_maxsim \
            vidore-benchmark evaluate-retriever --model-name colbert_live --collection-name "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d" --split test
            echo "----------------------------------------"
    done
done
