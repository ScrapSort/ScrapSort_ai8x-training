#!/bin/sh
./train.py --model simplesortingnetbnbb --dataset sortingbb --evaluate --exp-load-weights-from /home/geffen/Documents/ScrapSort/src/ai8x-training/logs/2022.01.25-171044/best.pth.tar --device MAX78000 "$@"