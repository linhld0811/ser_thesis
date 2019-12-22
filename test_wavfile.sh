#!/bin/bash
# code by linhld
# created 8/2019
if test $# -ne 4;then
	echo 
        echo "Usage: ./test_wavfile.sh path_wav save_model duration samplerate"
	echo
        exit 1
fi

echo "Predict emotion from speech signal file.............................................."
python scr/test_wavfile.py --path_wav=$1 \
	--saved_dirs=$2 \
	--duration=$3 \
	--sr=$4


