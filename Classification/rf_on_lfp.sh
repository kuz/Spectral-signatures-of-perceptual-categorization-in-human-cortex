#!/bin/bash

INDATA='LFP_8c_artif_bipolar_BA_responsive'

for sid in $(seq 0 1 97)
do
    srun --partition=long,phi,main -c 4 --mem=2000 -t 24:00:00 python rf_on_lfp.py -f $INDATA -s $sid &
    sleep 2
done