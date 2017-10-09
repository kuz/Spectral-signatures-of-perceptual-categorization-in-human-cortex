#!/bin/bash
#module load python-2.7.11
#source /gpfs/hpchome/a72073/ve/py27/bin/activate

for i in $(seq 1 8)
do
    let i=i-1
    srun --partition=long,phi,main -c 16 --mem=20000 -t 24:00:00 python ft_feature_importance.py --cid $i &
    sleep 1
done
