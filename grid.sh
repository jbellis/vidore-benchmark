#!/bin/bash

# Arrays for each environment variable
VIDORE_DOC_POOL=(1 2 3 4 5)
VIDORE_QUERY_POOL=(0.0 0.01 0.015 0.02 0.03)
VIDORE_N_ANN=(80 160 240 320)
VIDORE_N_MAXSIM=(10 15 20 25)

# Loop through all combinations
for doc_pool in "${VIDORE_DOC_POOL[@]}"; do
  for query_pool in "${VIDORE_QUERY_POOL[@]}"; do
      for n_ann in "${VIDORE_N_ANN[@]}"; do
          for n_maxsim in "${VIDORE_N_MAXSIM[@]}"; do
              echo "VIDORE_DOC_POOL=$doc_pool, VIDORE_QUERY_POOL=$query_pool, VIDORE_N_ANN=$n_ann, VIDORE_N_MAXSIM=$n_maxsim"
              VIDORE_DOC_POOL=$doc_pool VIDORE_QUERY_POOL=$query_pool VIDORE_N_ANN=$n_ann VIDORE_N_MAXSIM=$n_maxsim \
              vidore-benchmark evaluate-retriever --model-name colbert_live --collection-name "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d" --split test
  #            vidore-benchmark evaluate-retriever --model-name colbert_live --dataset-name vidore/docvqa_test_subsampled --split test
              echo "----------------------------------------"
          done
      done
  done
done
