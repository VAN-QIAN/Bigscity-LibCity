#!/bin/bash
#SBATCH --partition=gpu_short  
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=4        # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:1             # Using 1 GPU card
#SBATCH --mem=40GB               # Request 50GB memory
#SBATCH --time=0-12:00:00        # Time limit day-hrs:min:sec
#SBATCH --output=../TGCNCoarse_%j.log   # Standard output
#SBATCH --error=../TGCNCoarse_%j.err    # Standard error log #38191 #24834  --train False --exp_id 38191
singularity exec --nv ../Torch1-7.sif python3 ./run_model.py --task traffic_state_pred --model TGCN --dataset PEMS_BAY --n1 0.8 --n2 0.2
