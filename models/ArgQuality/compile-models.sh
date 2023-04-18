#!/usr/bin/env bash

for bertmodel in bert_*/
do
    cd $bertmodel
    bertmodel_name="${bertmodel:0:-1}.pt"
    echo "Compressing ${bertmodel} to ${bertmodel_name}"
    zip -rq $bertmodel_name $bertmodel
    cd ..
done

