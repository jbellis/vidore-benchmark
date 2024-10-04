#!/bin/bash

# Arrays for each environment variable
VIDORE_DOC_POOL=(1 2 3 4 5)

# Loop through all combinations
for doc_pool in "${VIDORE_DOC_POOL[@]}"; do
    echo "VIDORE_DOC_POOL=$doc_pool, VIDORE_N_ANN=$n_ann, VIDORE_N_MAXSIM=$n_maxsim"
    VIDORE_DOC_POOL=$doc_pool \
    vidore-benchmark evaluate-retriever --model-name colbert_live --collection-name "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d" --split test
    echo "----------------------------------------"
done
