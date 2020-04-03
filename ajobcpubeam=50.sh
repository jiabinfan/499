#!/bin/bash
#sBATCH --account=def-lilimou
#SBATCH --mem=16G
#SBATCH --time=108:00:00
python transformerseq2seqbeam.py
