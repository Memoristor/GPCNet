#!/bin/bash

source source.sh

CONFIG=$1
SEED=$2
PRUNE_GPUS=$3
PRUNE_ARGS=$4
TRAIN_GPUS=$5   
TRAIN_ARGS=$6
TEST_GPUS=$7
TEST_ARGS=$8

# Get the number of visible GPUS
NUM_PRUNE_GPUS=$(echo "$PRUNE_GPUS" | awk -F',' '{print NF}')
NUM_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | awk -F',' '{print NF}')
NUM_TEST_GPUS=$(echo "$TEST_GPUS" | awk -F',' '{print NF}')

# Get random master port
MASTER_PORT=$(alloc_port 28000 38000)

# Run script
CUDA_VISIBLE_DEVICES="${PRUNE_GPUS}" torchrun \
    --nproc_per_node ${NUM_PRUNE_GPUS} \
    --nnodes 1 \
    --master_port ${MASTER_PORT} \
    main.py --wandb_project GPCNet \
    --yaml_config ${CONFIG} \
    --proc_phase prune \
    --random_seed ${SEED} ${PRUNE_ARGS} 

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun \
    --nproc_per_node ${NUM_TRAIN_GPUS} \
    --nnodes 1 \
    --master_port ${MASTER_PORT} \
    main.py --wandb_project GPCNet \
    --yaml_config ${CONFIG} \
    --proc_phase train \
    --random_seed ${SEED} ${TRAIN_ARGS} 
    
CUDA_VISIBLE_DEVICES="${TEST_GPUS}" torchrun \
    --nproc_per_node ${NUM_TEST_GPUS} \
    --nnodes 1 \
    --master_port ${MASTER_PORT} \
    main.py --wandb_project GPCNet \
    --yaml_config ${CONFIG} \
    --proc_phase test \
    --random_seed ${SEED} ${TEST_ARGS} 