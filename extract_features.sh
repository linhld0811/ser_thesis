#!/bin/bash
# code by linhld
# created 8/2019
if test $# -ne 4;then
	echo
        echo "Usage: ./process.sh scr_folder dst_folder duration samplerate"
	echo
        exit 1
fi

echo "Processing data.........................................................."
python scr/extract_features.py --scr_folder=$1 \
	--dst_folder=$2 \
	--duration=$3 \
	--sr=$4
echo "Done!"
