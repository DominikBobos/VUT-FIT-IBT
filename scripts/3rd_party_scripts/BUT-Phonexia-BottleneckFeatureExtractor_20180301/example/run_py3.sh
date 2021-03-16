#!/bin/bash 

NNETWAV2BN=FisherMono
NNETBN2POST=FisherMono
AUDIOFILE=example.wav
BNFILE=example_py3.fea
VADFILE=example.lab.gz
POSTFILE=example_py3.h5

#extract BN features and save them to the file example.fea
MKL_NUM_THREADS=1 python3 ../audio2bottleneck.py $NNETWAV2BN $AUDIOFILE $BNFILE $VADFILE
#display the first frame and a header of the BN feature file
HList -h -e 0 $BNFILE

#using extracted BN calculate phoneme states posterior probabilities and save them to the file example.h5

MKL_NUM_THREADS=1 python3 ../bottleneck2posterior.py $NNETBN2POST $BNFILE $POSTFILE


