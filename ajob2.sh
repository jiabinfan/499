#!/bin/bash
#sBATCH --account=def-lilimou
#SBATCH --gres=gpu:p100l:4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=6
python transformerseq2seqGPU2.py

