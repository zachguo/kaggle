#! /bin/bash

touch predictions.txt

for i in {0..52}; do # there're 52 subcategories
	echo $i; 
	python tsv2vw.py train_$i.tsv train_$i.vw
	python tsv2vw.py test_$i.tsv test_$i.vw
	vw -b 29 --loss_function logistic -c --passes 20 train_$i.vw -P 1e6
	vw -b 29 --loss_function logistic -c --passes 20 -d train_$i.vw -P 1e6 --holdout_off -f model_$i
	vw -t -i model_$i -d test_$i.vw -p predictions_$i.txt
	cat predictions_$i.txt >> predictions.txt
done