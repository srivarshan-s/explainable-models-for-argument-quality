#!/usr/bin/env bash

echo "Compressing bert-vanilla"
zip -rq bert-vanilla.pt bert-vanilla/
echo "Compressing bert-finetune"
zip -rq bert-finetune.pt bert-finetune/
