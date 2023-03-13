#!/bin/bash
#SBATCH --partition=gpu_1d2g   
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=4        # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:2             # Using 1 GPU card
#SBATCH --mem=40GB               # Request 50GB memory
#SBATCH --time=0-10:00:00        # Time limit day-hrs:min:sec
#SBATCH --output=../TGCNC-LA_%j.log   # Standard output
#SBATCH --error=../TGCNC-LA_%j.err    # Standard error log
singularity exec --nv ../Torch1-7.sif python3 ./run_model.py --task traffic_state_pred --model TGCN --dataset METR_LA
