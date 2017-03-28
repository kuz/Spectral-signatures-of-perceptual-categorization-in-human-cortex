#!/bin/bash

INDATA='LFP_8c_artif_bipolar_BA_responsive'

declare -a wins=("[0 200]" "[50 250]" "[100 300]" "[150 350]" "[200 400]" "[250 450]" "[300 500]")
declare -a freqs=("[5 9]" "[10 14]" "[15 19]" "[20 24]" "[25 29]" "[30 34]" "[35 39]" "[40 44]" "[45 49]"
                  "[50 54]" "[55 59]" "[60 64]" "[65 69]" "[70 74]" "[75 79]" "[80 84]" "[85 89]" "[90 94]" "[95 99]"
                  "[100 104]" "[105 109]" "[110 114]" "[115 119]" "[120 124]" "[125 129]" "[130 134]" "[135 139]"
                  "[140 144]" "[145 149]")
declare -a cycls=("3" "4" "5" "5" "5" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6" "6")
lastfid=28
pool=8
counter=0

for win in "${wins[@]}"
do
    for fid in $(seq 0 1 $lastfid)
    do
        ((counter+=1))
        while [ $(jobs -p | wc -l) -ge $pool ]; do
            sleep 60
        done
        matlab -nojvm -nodisplay -nosplash -r "indata = '$INDATA'; freqlimits = ${freqs[$fid]}; ncycles = ${cycls[$fid]}; window = $win; extract_meanband; exit" &
    done
done