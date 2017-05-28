#!/bin/bash

INDATA='LFP_8c_artif_bipolar_BA_responsive'

for sid in $(seq 1 1 10)
do
    srun --partition=long,phi,main -c 2 --mem=2000 -t 36:00:00 /storage/software/MATLAB_R2013b/bin/matlab -nojvm -nodisplay -nosplash -r "indata='$INDATA'; sid=$sid; extract_spectrum; exit" &
    sleep 2
done