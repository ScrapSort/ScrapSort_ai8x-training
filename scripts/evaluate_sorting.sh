#!/bin/sh
./train.py --model simplesortingnetbn --dataset sorting --confusion --evaluate --exp-load-weights-from /home/geffen/Documents/ScrapSort/src/ai8x-synthesis/trained/simplesort8_qat-q.pth.tar -8 --device MAX78000 "$@"