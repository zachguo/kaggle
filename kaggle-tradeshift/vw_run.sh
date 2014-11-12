#! /bin/bash

export DATAPATH=./data/
export VWPATH=./vw/
export SUBPATH=./submission/

echo -e "\nTraining VW...\n"
vw -b 30 --loss_function logistic --nn 25 -q qq -d $DATAPATH/train.vw -f $VWPATH/model_$1
echo -e "\nPredicting...\n"
vw -b 30 -t -i $VWPATH/model_$1 -d $DATAPATH/test.vw -p $VWPATH/predictions_$1.txt
echo -e "\nGenerating submission file...\n"
python blend.py --vwvar $SUBPATH/submission_b23_q4.csv $VWPATH/predictions_$1.txt 33