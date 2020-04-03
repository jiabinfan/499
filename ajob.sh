#!/bin/bash
#sBATCH --account=def-lilimou
#SBATCH --gres=gpu:p100l:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
python transformerseq2seqGPU.py
