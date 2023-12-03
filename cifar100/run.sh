#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 PYTHONPATH="../../" bash .sh
# ps | grep -ie python | awk '{print $1}' | xargs kill -9 


function cifar100_exp {
    COMMON="--lr 0.1 --identifier cifar100_exp -n 100 --K 10 --K_gen 10 --epochs 100 --batch-size 32 --max-batch-size-per-epoch 99999 --noniid 0 --momentum 0.9"
    export CUDA_VISIBLE_DEVICES=0
    python3 launcher.py ${COMMON} --algorithm global --use-cuda  --model-type 'D' &
    pids[$!]=$!
    python3 launcher.py ${COMMON} --algorithm local --use-cuda  --model-type 'D' &
    pids[$!]=$!
    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}

function cifar100_fc {
    # Find vgg size that does not fit cifar 100 well.
    COMMON="--lr 0.1 --identifier cifar100_exp -n 100 --K 10 --K_gen 10 --epochs 100 --batch-size 32 --max-batch-size-per-epoch 99999 --noniid 0 --momentum 0.9"
    export CUDA_VISIBLE_DEVICES=1
    python3 launcher.py ${COMMON} --algorithm fc-grad-10-quantile0.1-5-20 --use-cuda  --model-type 'D'
    python3 launcher.py ${COMMON} --algorithm fc-grad-10-quantile0.1-2-20 --use-cuda  --model-type 'D'
}

PS3='Please enter your choice: '
options=("cifar100_exp" "cifar100_fc" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "cifar100_exp")
            cifar100_exp
            ;;
        
        "cifar100_fc")
            cifar100_fc
            ;;


        "Quit")
            break
            ;;

        *) 
            echo "invalid option $REPLY"
            ;;
    esac
done
