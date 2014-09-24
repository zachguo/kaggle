#! /bin/bash

export DATAPATH=./data
touch $DATAPATH/predictions.txt
for i in {0..5}; do
	echo $i; 
	python csv2vw.py $DATAPATH/train_$i.csv $DATAPATH/train_$i.vw
	python csv2vw.py $DATAPATH/test_$i.csv $DATAPATH/test_$i.vw
	vw -b 29 --nn 50 -c --passes 20 $DATAPATH/train_$i.vw -P 1e6
	vw -b 29 --nn 50 -c --passes 20 -d $DATAPATH/train_$i.vw -P 1e6 --holdout_off -f $DATAPATH/model_$i
	vw -t -i $DATAPATH/model_$i -d $DATAPATH/test_$i.vw -p $DATAPATH/predictions_$i.txt
	cat $DATAPATH/predictions_$i.txt >> $DATAPATH/predictions.txt
done