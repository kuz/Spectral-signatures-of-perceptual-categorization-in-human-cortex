#!/bin/bash

INDATA='ft_4hz150_LFP_8c_artif_bipolar_BA_responsive'

# max 97
for sid in $(seq 0 1 9)
do
    srun --partition=long,phi,main -c 16 --mem=20000 -t 48:00:00 python rf_on_ft.py -f $INDATA -s $sid &
    sleep 2
done