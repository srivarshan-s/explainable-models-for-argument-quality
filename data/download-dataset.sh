#!/usr/bin/env bash

rm ./data/arg_quality_rank_30k.csv
wget "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_arg_quality_rank_30k.zip"
unzip *.zip
rm *.zip
rm readme.txt
mv *.csv ./data/.
