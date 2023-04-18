#!/usr/bin/env bash

for bertmodel in bert_*/
do
    cd $bertmodel
    bertmodel_name="${bertmodel:0:-1}.pt"
    echo "Compressing ${bertmodel} to ${bertmodel_name}"
    zip -rq $bertmodel_name $bertmodel
    cd ..
done

for svm_model in *svm*/
do
    cd $svm_model
    svm_model_name="${svm_model:0:-1}"
    echo "Combining splits in ${svm_model}"
    cat x* > $svm_model_name
    cd ..
done
