#!/bin/bash
#sBATCH --account=def-lilimou
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
python transformerseq2seq.py
