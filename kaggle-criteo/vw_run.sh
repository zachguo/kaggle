#! /bin/bash

export DATAPATH=./data/
export OUTPATH=./output/vw/

echo -e "\nTraining VW...\n"
vw -b 30 --loss_function logistic --nn 25 -d $DATAPATH/train_with_bgctr.vw -P 1e6 --holdout_off -f $OUTPATH/model_$1
echo -e "\nPredicting...\n"
vw -b 30 -t -i $OUTPATH/model_$1 -d $DATAPATH/test_with_bgctr.vw -p $OUTPATH/predictions_$1.txt
echo -e "\nGenerating submission file...\n"
python vw2kaggle.py $1