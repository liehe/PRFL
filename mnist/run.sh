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
    COMMON="--lr 0.1 --identifier rotation -n 300 --K 4 --epochs 30 --batch-size 32 --max-batch-size-per-epoch 30 --noniid 0 --momentum 0.9 --data rotation"

    for algorithm in ifca-grad global "local" ditto-1 groundtruth; do
        python launcher.py ${COMMON} --algorithm $algorithm &
        pids[$!]=$!
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}

function knnper_rotation_addon {
    # Add knn-personalization experiment for rotation data.
    COMMON="--lr 0.1 -n 300 --K 4 --epochs 30 --batch-size 32 --max-batch-size-per-epoch 30 --noniid 0 --momentum 0.9"

    python launcher.py ${COMMON} --identifier relabel --data relabel --algorithm knnper-0.2-10-gaussian --knnper-phase FedAvgVal
    python launcher.py ${COMMON} --identifier relabel --data relabel --algorithm knnper-0.2-10-gaussian --knnper-phase kNNPerVal
    python launcher.py ${COMMON} --identifier relabel --data relabel --algorithm knnper-0.2-10-gaussian --knnper-phase FedAvgTrain
    python launcher.py ${COMMON} --identifier relabel --data relabel --algorithm knnper-0.2-10-gaussian --knnper-phase kNNPerTest
}

function relabel {
    # ---------------------------------------------------------------------------- #
    #                  Compare cluster algorithms on relabel data.                 #
    # ---------------------------------------------------------------------------- #
    # NOTE: 
    #     small neural network
    #     default validation batch size
    COMMON="--lr 0.1 --identifier relabel -n 300 --K 4 --epochs 30 --batch-size 32 --max-batch-size-per-epoch 30 --noniid 0 --momentum 0.9"

    python launcher.py ${COMMON} --data relabel --algorithm ifca-grad &
    pids[$!]=$!

    python launcher.py ${COMMON} --data relabel --algorithm global  &
    pids[$!]=$!

    python launcher.py ${COMMON} --data relabel --algorithm "local"  &
    pids[$!]=$!

    python launcher.py ${COMMON} --data relabel --algorithm ditto-1 &
    pids[$!]=$!

    python launcher.py ${COMMON} --data relabel --algorithm groundtruth &
    pids[$!]=$!

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}


function knnper_relabel_addon {
    # Add knn-personalization experiment for relabel data.
    COMMON="--lr 0.1 -n 300 --K 4 --epochs 30 --batch-size 32 --max-batch-size-per-epoch 30 --noniid 0 --momentum 0.9"

    python launcher.py ${COMMON} --identifier rotation --data rotation --algorithm knnper-0.2-10-gaussian --knnper-phase FedAvgVal
    python launcher.py ${COMMON} --identifier rotation --data rotation --algorithm knnper-0.2-10-gaussian --knnper-phase kNNPerVal
    python launcher.py ${COMMON} --identifier rotation --data rotation --algorithm knnper-0.2-10-gaussian --knnper-phase FedAvgTrain
    python launcher.py ${COMMON} --identifier rotation --data rotation --algorithm knnper-0.2-10-gaussian --knnper-phase kNNPerTest
}

function relabel_fc_all {
    # ---------------------------------------------------------------------------- #
    #                    fc is slower, but still try to run all.                   #
    # ---------------------------------------------------------------------------- #
    COMMON="--lr 0.1 --identifier relabel -n 300 --K 4 --epochs 50 --batch-size 1000 --max-batch-size-per-epoch 30 --noniid 0 --momentum 0.9"

    python launcher.py ${COMMON} --data relabel --algorithm fc-grad-3-quantile0.2-15
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
            # Private label task
            relabel
            knnper_relabel_addon
            relabel_fc_all
            ;;
        
        "relabel_fc_all")
            relabel_fc_all
            ;;
        
        "relabel_gt")
            relabel_gt
            ;;

        *) 
            echo "invalid option $REPLY"
            ;;
    esac
done
