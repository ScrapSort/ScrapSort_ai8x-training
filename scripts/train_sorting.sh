#!/bin/sh
./train.py --epochs 50 --deterministic  --compress sorting_schedule.yaml --model simplesortingnetbn --dataset sorting --qat-policy qat_policy_sorting.yaml --confusion --param-hist --pr-curves --embedding --enable-tensorboard --batch-size 64 --optimizer Adam --lr 0.001 --device MAX78000 "$@"
