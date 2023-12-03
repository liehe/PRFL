#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 PYTHONPATH="../../" bash .sh
# ps | grep -ie python | awk '{print $1}' | xargs kill -9 

function rotation {
    # ---------------------------------------------------------------------------- #
    #                 Compare cluster algorithms on rotation data.                 #
    # ---------------------------------------------------------------------------- #
    # NOTE: 
    #     small neural network
    #     default validation batch size
    export CUDA_VISIBLE_DEVICES=0
    COMMON="--lr 0.1 --identifier rotation -n 20 --K 4 --epochs 30 --batch-size 32 --max-batch-size-per-epoch 999999 --noniid 0 --momentum 0.9 --data rotation"

    python3 launcher.py ${COMMON} --use-cuda --algorithm ifca-grad &
    pids[$!]=$!

    python3 launcher.py ${COMMON} --use-cuda --algorithm global  &
    pids[$!]=$!

    python3 launcher.py ${COMMON} --use-cuda --algorithm "local"  &
    pids[$!]=$!

    python3 launcher.py ${COMMON} --use-cuda --algorithm ditto-1 &
    pids[$!]=$!

    python3 launcher.py ${COMMON} --use-cuda --algorithm groundtruth &
    pids[$!]=$!


    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}

function relabel {
    # ---------------------------------------------------------------------------- #
    #                  Compare cluster algorithms on relabel data.                 #
    # ---------------------------------------------------------------------------- #
    # NOTE: 
    #     small neural network
    #     default validation batch size
    export CUDA_VISIBLE_DEVICES=1
    COMMON="--lr 0.1 --identifier relabel -n 20 --K 4 --epochs 60 --batch-size 32 --max-batch-size-per-epoch 9999999 --noniid 0 --momentum 0.9"

    python3 launcher.py ${COMMON} --use-cuda --data relabel --algorithm ifca-grad &
    pids[$!]=$!

    python3 launcher.py ${COMMON} --use-cuda --data relabel --algorithm global  &
    pids[$!]=$!

    python3 launcher.py ${COMMON} --use-cuda --data relabel --algorithm "local"  &
    pids[$!]=$!

    python3 launcher.py ${COMMON} --use-cuda --data relabel --algorithm ditto-0.1 &
    pids[$!]=$!

    python3 launcher.py ${COMMON} --use-cuda --data relabel --algorithm groundtruth &
    pids[$!]=$!

    # python3 launcher.py ${COMMON} --use-cuda --data relabel --algorithm fc-grad-3-cd &
    # pids[$!]=$!

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}


function knnper_rotation_addon {
    # ---------------------------------------------------------------------------- #
    #                  Compare cluster algorithms on relabel data.                 #
    # ---------------------------------------------------------------------------- #
    # NOTE: 
    #     small neural network
    #     default validation batch size
    export CUDA_VISIBLE_DEVICES=3
    COMMON="--lr 0.1 --identifier relabel -n 20 --K 4 --epochs 60 --batch-size 32 --max-batch-size-per-epoch 9999999 --noniid 0 --momentum 0.9"

    python3 launcher.py ${COMMON} --data rotation --algorithm knnper-0.2-10-gaussian --knnper-phase FedAvgVal --use-cuda 
    python3 launcher.py ${COMMON} --data rotation --algorithm knnper-0.2-10-gaussian --knnper-phase kNNPerVal --use-cuda 
    python3 launcher.py ${COMMON} --data rotation --algorithm knnper-0.2-10-gaussian --knnper-phase FedAvgTrain --use-cuda 
    python3 launcher.py ${COMMON} --data rotation --algorithm knnper-0.2-10-gaussian --knnper-phase kNNPerTest --use-cuda 
}

function knnper_relabel_addon {
    # ---------------------------------------------------------------------------- #
    #                  Compare cluster algorithms on relabel data.                 #
    # ---------------------------------------------------------------------------- #
    # NOTE: 
    #     small neural network
    #     default validation batch size
    export CUDA_VISIBLE_DEVICES=3
    COMMON="--lr 0.1 --identifier relabel -n 20 --K 4 --epochs 60 --batch-size 32 --max-batch-size-per-epoch 9999999 --noniid 0 --momentum 0.9"

    python3 launcher.py ${COMMON} --data relabel --algorithm knnper-0.2-10-gaussian --knnper-phase FedAvgVal --use-cuda 
    python3 launcher.py ${COMMON} --data relabel --algorithm knnper-0.2-10-gaussian --knnper-phase kNNPerVal --use-cuda 
    python3 launcher.py ${COMMON} --data relabel --algorithm knnper-0.2-10-gaussian --knnper-phase FedAvgTrain --use-cuda 
    python3 launcher.py ${COMMON} --data relabel --algorithm knnper-0.2-10-gaussian --knnper-phase kNNPerTest --use-cuda 
}

function relabel_fc {
    # ---------------------------------------------------------------------------- #
    #                    fc is slower, but still try to run all.                   #
    # ---------------------------------------------------------------------------- #
    # export CUDA_VISIBLE_DEVICES=4
    # COMMON="--lr 0.1 --identifier rotation -n 20 --K 4 --epochs 30 --batch-size 32 --max-batch-size-per-epoch 999999 --noniid 0 --momentum 0.9 --data rotation"
    # python3 launcher.py ${COMMON} --algorithm fc-grad-3-quantile0.25-2 --use-cuda 
    export CUDA_VISIBLE_DEVICES=0
    COMMON="--lr 0.1 --identifier relabel -n 20 --K 4 --epochs 60 --batch-size 32 --max-batch-size-per-epoch 999999 --noniid 0 --momentum 0.9"
    python3 launcher.py ${COMMON} --data relabel --algorithm fc-grad-3-quantile0.2-1-10 --use-cuda 
    python3 launcher.py ${COMMON} --data relabel --algorithm fc-grad-3-quantile0.25-1-10 --use-cuda 
    python3 launcher.py ${COMMON} --data relabel --algorithm fc-grad-10-quantile0.2-2-10 --use-cuda 
    python3 launcher.py ${COMMON} --data relabel --algorithm fc-grad-5-quantile0.1-1-1 --use-cuda 
    python3 launcher.py ${COMMON} --data relabel --algorithm fc-grad-10-quantile0.1-2-10 --use-cuda 
    python3 launcher.py ${COMMON} --data relabel --algorithm fc-grad-10-quantile0.05-2-10 --use-cuda 
    python3 launcher.py ${COMMON} --data relabel --algorithm fc-grad-10-quantile0.025-2-10 --use-cuda 
    python3 launcher.py ${COMMON} --data relabel --algorithm fc-grad-10-quantile0.2-2-20 --use-cuda 
    python3 launcher.py ${COMMON} --data relabel --algorithm fc-grad-10-quantile0.2-2-40 --use-cuda 
    python3 launcher.py ${COMMON} --data relabel --algorithm fc-grad-10-quantile0.2-2-5 --use-cuda 

}


PS3='Please enter your choice: '
options=("Quit" "rotation" "relabel")
select opt in "${options[@]}"
do
    case $opt in
        "Quit")
            break
            ;;
    

        "rotation")
            rotation
            knnper_rotation_addon
            ;;
                
        "relabel")
            relabel
            knnper_relabel_addon
            relabel_fc
            ;;

        *) 
            echo "invalid option $REPLY"
            ;;
    esac
done
