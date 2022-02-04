#!/bin/sh
./trainbb.py --epochs 20  --compress sortingbb_schedule.yaml --model simplesortingnetbnbb --dataset sortingbb --qat-policy qat_policy_sortingbb.yaml --param-hist --enable-tensorboard --batch-size 64 --regression --optimizer Adam --lr 0.004 --device MAX78000 "$@"
