#! /bin/bash

export DATAPATH=./data/
export OUTPATH=./output/cv_ensemble/vw

echo -e "\nPredicting...\n"
vw -b 30 --loss_function logistic -t -i output/vw/model_$1 -d $DATAPATH/train_with_bgctr.vw -p $OUTPATH/predictions_$1.txt
echo -e "\nGenerating submission file...\n"
python vw2kaggle_ensemble_cv.py $1