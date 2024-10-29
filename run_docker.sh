#!/bin/bash
WANDB_API_KEY=$(cat ./setup/docker/wandb_key)
git pull

script_and_args="${@:2}"
gpu=$1
echo "Launching container data_collection_$gpu on GPU $gpu"
docker run \
    --env CUDA_VISIBLE_DEVICES=$gpu \
    --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e WANDB_AGENT_MAX_INITIAL_FAILURES=1000 \
    -e WANDB_AGENT_DISABLE_FLAPPING=true \
    -v $(pwd):/home/duser/rl \
    --name data_collection_$gpu \
    --user $(id -u) \
    -d \
    -t data_collection\
    /bin/bash -c "$script_and_args"
