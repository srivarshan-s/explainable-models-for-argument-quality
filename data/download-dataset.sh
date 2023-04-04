#!/usr/bin/env bash

if [ -f arg_quality_rank_30k.csv ]
then
    rm arg_quality_rank_30k.csv
fi
wget "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_arg_quality_rank_30k.zip"
unzip *.zip
rm *.zip
rm readme.txt
