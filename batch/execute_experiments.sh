#!/bin/bash

# This script is used to execute experiments on a SLURM cluster.

BLOCK_DIM_X=2    BLOCK_DIM_Y=2  sbatch ./batch/profile.batch && sleep 1
BLOCK_DIM_X=4    BLOCK_DIM_Y=4  sbatch ./batch/profile.batch && sleep 1
BLOCK_DIM_X=8    BLOCK_DIM_Y=8  sbatch ./batch/profile.batch && sleep 1
BLOCK_DIM_X=16   BLOCK_DIM_Y=16 sbatch ./batch/profile.batch && sleep 1
BLOCK_DIM_X=32   BLOCK_DIM_Y=32 sbatch ./batch/profile.batch && sleep 1

# BLOCK_DIM_X=4    BLOCK_DIM_Y=1  sbatch ./batch/profile.batch && sleep 1
# BLOCK_DIM_X=8    BLOCK_DIM_Y=1  sbatch ./batch/profile.batch && sleep 1
# BLOCK_DIM_X=16   BLOCK_DIM_Y=1  sbatch ./batch/profile.batch && sleep 1
# BLOCK_DIM_X=32   BLOCK_DIM_Y=1  sbatch ./batch/profile.batch && sleep 1
# BLOCK_DIM_X=64   BLOCK_DIM_Y=1  sbatch ./batch/profile.batch && sleep 1
# BLOCK_DIM_X=128  BLOCK_DIM_Y=1  sbatch ./batch/profile.batch && sleep 1
# BLOCK_DIM_X=256  BLOCK_DIM_Y=1  sbatch ./batch/profile.batch && sleep 1
# BLOCK_DIM_X=512  BLOCK_DIM_Y=1  sbatch ./batch/profile.batch && sleep 1
# BLOCK_DIM_X=1024 BLOCK_DIM_Y=1  sbatch ./batch/profile.batch && sleep 1