#!/bin/bash

INDATA='ft_4hz150_LFP_8c_artif_bipolar_BA_responsive'

for sid in $(seq 0 1 97)
do
    srun --partition=main -c 10 --mem=20000 -t 48:00:00 python rf_on_bbgamma.py -f $INDATA -s $sid &
    sleep 2
done