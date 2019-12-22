#!/bin/bash 
# code by linhld
# created on 8/2019

#folder to cut
folder=$1
#folder to save cutted file
if test $# -ne 2;then
	echo "--------------------------------------------------------------------------"
        echo "Usage ./cut.sh path_to_input path_to_output"
	echo ""
        exit 1
fi

new_folder=$2
stage=1
time_cut=3
time_shift=1.5
#time_overlap=$(echo "$time_cut-$time_shift"|bc -l|awk '{printf "%.2f",$0}')
time_overlap=1.5
if [ $stage -le 0 ]; then
	echo "Step1: Cutting file wav to segment with length" $time_cut "s.............."
	for i in `ls ${folder}/*/*.wav`; do
        	t=0
	    	number=0
		path=$(echo $i|awk -F/ '{print$NF}')
	        name=$( basename --suffix=.wav $path)
		emotion=$(echo $i|awk -F/ '{print$(NF-1)}')
       		length=$(soxi -D $i)
	        #n=$(echo "scale=1; $length-1"|bc -l| awk '{printf "%.2f",$0}')
		[ ! -d ${new_folder}/${emotion} ] && mkdir -p ${new_folder}/${emotion}
        	while (( $(echo "$length>$t" |bc -l) )); do
                	number=$(echo $((number+1)))
	                sox $i ${new_folder}/${emotion}/${name}_${number}.wav trim $t ${time_cut}
       		        t=$(echo "$t+$time_shift"|bc -l|awk '{printf "%.2f",$0}')
	        done
        #rm $i
	done
fi
#### remove audio has lenght that less than 1s
emotion_list="ang hap neu sad"
#emotion_list="angry neutral"
#emotion_list="angry happy neutral calm disgust surprised fearful sad"
echo "Step 2: remove file with length less than " $time_overlap "s....."
for e in ${emotion_list[*]}; do
	if [ $stage -le 1 ]; then
		for i in `ls ${new_folder}/${e}`;do
			length=$(soxi -D ${new_folder}/${e}/$i)
			if ((  $(echo "$length<$time_overlap"|bc -l) )); then
				rm ${new_folder}/${e}/$i
			fi
		done
	fi
done
