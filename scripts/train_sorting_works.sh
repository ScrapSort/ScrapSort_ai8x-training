#!/bin/sh
./train.py --epochs 40  --compress sorting_schedule.yaml --model simplesortingnetbnworks --dataset sorting --qat-policy qat_policy_sorting.yaml --confusion --param-hist --pr-curves --embedding --enable-tensorboard --batch-size 128 --optimizer Adam --lr 0.00025 --device MAX78000 "$@"
