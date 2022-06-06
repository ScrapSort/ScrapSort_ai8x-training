#!/bin/sh
./trainss.py --epochs 200  --compress sorting_schedule.yaml --model simplesortingnetbnworksc --dataset sortingss --qat-policy qat_policy_sorting.yaml --param-hist --enable-tensorboard --batch-size 128 --regression --optimizer Adam --lr 0.0005  --device MAX78000 "$@"
