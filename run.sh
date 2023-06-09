#! /bin/bash
python main_fed.py --epochs 500 --num_users 500 --local_bs 10 --frac 0.05 --local_ep 5 --lr 0.01 --mu 0.001 --a 0.1 --beta 0.8 --model 'cnn' --dataset 'cifar' --gpu 1 --seed 42 --select 'uniform'


