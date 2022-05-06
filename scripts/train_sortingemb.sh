#!/bin/sh
./trainemb.py --epochs 50  --compress sorting_schedule.yaml --model simplesortingnetemb --dataset sorting --qat-policy qat_policy_sorting.yaml --param-hist --enable-tensorboard --batch-size 128 --regression --optimizer Adam --lr 0.0001    --device MAX78000 "$@"
