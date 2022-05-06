#!/bin/sh
./train.py --epochs 100  --compress sorting_schedule.yaml --model simplesortingnetbn --dataset sorting --qat-policy qat_policy_sorting.yaml --confusion --param-hist --pr-curves --embedding --enable-tensorboard --batch-size 128 --optimizer Adam --lr 0.002    --device MAX78000 "$@"
