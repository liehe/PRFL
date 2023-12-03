#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 PYTHONPATH="../../" bash .sh
# ps | grep -ie python | awk '{print $1}' | xargs kill -9 


function byzantine_gaussian {
    COMMON="--lr 0.1 --identifier byzantine_gaussian -n 200 --K 4 --epochs 1 --batch-size 32 --max-batch-size-per-epoch 30 --noniid 0 --momentum 0.9  --data relabel "

    python launcher.py ${COMMON} --b 50  --byz-kind "Gaussian" --algorithm fc-grad-10-quantile0.2-1
    python launcher.py ${COMMON} --b 50  --byz-kind "Gaussian" --algorithm global
}

function byzantine_bitflipping {
    COMMON="--lr 0.1 --identifier byzantine_bitflipping -n 200 --K 4 --epochs 100 --batch-size 32 --max-batch-size-per-epoch 30 --noniid 0 --momentum 0.9  --data relabel "

    python launcher.py ${COMMON} --b 50  --byz-kind "BF" --algorithm fc-grad-10-quantile0.2-1 & 
    pids[$!]=$!
    python launcher.py ${COMMON} --b 50  --byz-kind "BF" --algorithm global&
    pids[$!]=$!

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}


PS3='Please enter your choice: '
options=("byzantine_gaussian" "byzantine_bitflipping" "debug" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "Quit")
            break
            ;;
        
        "byzantine_bitflipping")
            byzantine_bitflipping
            ;;
        
        "byzantine_gaussian")
            byzantine_gaussian
            ;;

        *) 
            echo "invalid option $REPLY"
            ;;
    esac
done
