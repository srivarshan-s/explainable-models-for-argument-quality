#!/usr/bin/env bash

for bertmodel in bert-*/
do
    bertmodel_name="${bertmodel:0:-1}.pt"
    echo "Compressing ${bertmodel} to ${bertmodel_name}"
    zip -rq $bertmodel_name $bertmodel
done

