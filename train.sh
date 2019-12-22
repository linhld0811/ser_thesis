#!/bin/bash
# code by linhld
# created 8/2019
if test $# -ne 5;then
	echo
        echo "Usage: ./train.sh --stage --train_folder --test_folder --path_to_save_models --num_class"
	echo "Args:"
	echo "stage: 0 - train model "
	echo "       1 - train model with k-fold."
	echo "NOTE: if stage = 1: test_folder is any characters"
	echo
        exit 1
fi
stage=$1
if [ $stage -eq 0 ]; then
	echo "Training model.........................................................."

	python scr/train.py --train_folder=$2 \
		--save_dir=$4 \
		--num_class=$5 \
		--epochs=100 \
		--valid_folder=$3 \
		--test_folder=$3
	echo "Done!"
fi
if [ $stage -eq 1 ]; then
	echo "Training model with k-fold ............................................"
	python scr/train_kfold.py --data_folder=$2 \
		--num_class=$5 \
		--save_dir=$4

	echo "Done!"
fi
