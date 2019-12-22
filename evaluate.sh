#!/bin/bash
# code by linhld
# created 8/2019
if test $# -ne 3;then
	echo
        echo "Usage: ./evaluate.sh path_to_npy save_model path_to_save_result"
	echo
        exit 1
fi

echo "Evaluating model.........................................................."
python scr/evaluate.py --test_folder=$1 \
        --save_dir=$2 \
        --result=result

