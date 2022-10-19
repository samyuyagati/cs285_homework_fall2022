#!/bin/bash

NTU=100
NGSPTU=1

python cs285/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name "q5_${NTU}_${NGSPTU}" -ntu $NTU -ngsptu $NGSPTU

python cs285/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name "q5_${NTU}_${NGSPTU}_hc" -ntu $NTU -ngsptu $NGSPTU
