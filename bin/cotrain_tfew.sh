export TOKENIZERS_PARALLELISM=false

for seed in 0 1 32 42 1024
do
    for dataset in rte cb boolq
    do
        python -m sr