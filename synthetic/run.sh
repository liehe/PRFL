#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 PYTHONPATH="../../" bash .sh
# ps | grep -ie python | awk '{print $1}' | xargs kill -9 

function synthetic_oracle {
    # Task: Compare aggregators on iid noniid topology 
    COMMON="--E 1000 --dataset D3 --initial-cluster-centers oracle --initial-assignment oracle"

    python synthetic.py ${COMMON} --solver DummyPersonalized --eta=0.01 --identifier "synthetic_oracle2"  &
    pids[$!]=$!
    python synthetic.py ${COMMON} --solver IFCA_Model --eta=0.002 --identifier "synthetic_oracle2"  &
    pids[$!]=$!
    python synthetic.py ${COMMON} --solver IFCA --eta=0.009 --identifier "synthetic_oracle2"  &
    pids[$!]=$!
    python synthetic.py ${COMMON} --solver Thresholding --eta=0.02 --identifier "synthetic_oracle2"  &
    pids[$!]=$!

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}

function verify_assumption {
    COMMON="--E 100 --dataset DX4,75,9,10 --initial-cluster-centers oracle --initial-assignment oracle"
    python synthetic.py ${COMMON} --solver GT --eta=0.0001 --identifier "verify_assumption" 
}

function mnist {
    # Task: Compare aggregators on iid noniid topology 
    COMMON="--lr 0.1 --debug -n 300 --K 4 --epochs 10 --momentum 0.9 --batch-size 32 --max-batch-size-per-epoch 50 --noniid 0 --data rotation"

    python synthetic.py ${COMMON} --algorithm "ifca-grad"  &
    pids[$!]=$!

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
}

PS3='Please enter your choice: '
options=("Quit" "synthetic_oracle" "mnist" "verify_assumption" "D3" "DX4,16,9,10" "DX16,4,9,10"
    "improved_efficiency")
select opt in "${options[@]}"
do
    case $opt in
        "Quit")
            break
            ;;

        "synthetic_oracle")
            synthetic_oracle
            ;;
        
        "mnist")
            mnist
            ;;

        "verify_assumption")
            verify_assumption
            ;;
                
        "D3")
            COMMON="--E 800 --dataset D3 --initial-cluster-centers oracle --initial-assignment oracle"

            python synthetic.py ${COMMON} --solver "Centralized" --eta=0.1 --identifier "D3"
            ;;

        "DX4,16,9,10")
            COMMON="--E 800 --dataset DX4,16,9,10 --initial-cluster-centers oracle --initial-assignment oracle"

            python synthetic.py ${COMMON} --solver "Thresholding=Q0.25" --beta 0.9 --eta=0.1 --identifier "D3"
            # python synthetic.py ${COMMON} --solver "Centralized" --eta=0.1 --identifier "D3"
            ;;

        "DX16,4,9,10")
            COMMON="--E 800 --dataset DX16,4,9,10 --initial-cluster-centers oracle --initial-assignment oracle"
            python synthetic.py ${COMMON} --solver "Centralized" --eta=0.001 --identifier "D3"
            ;;

        *) 
            echo "invalid option $REPLY"
            ;;
    esac
done


