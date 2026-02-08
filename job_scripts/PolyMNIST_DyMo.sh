#!/bin/bash

cd ..

CKPT=False
imputer_name='CMVAE'  
transformer_CKPT=/home/siyi/project/mm/result/Dynamic_project/PM40/DynamicTransformer_singleCLS_EU_whole_none_PolyMNIST_DynamicTransformer_singleCLS_0702_174823/downstream/checkpoint_best_acc.ckpt #${YOUR_DYNAMIC_TRANSFORMER_CHECKPOINT_PATH}  # TODO: Change the path to your downloaded transformer checkpoint
imputer_CKPT=/home/siyi/project/mm/result/Dynamic_project/PM51/reproduce_cmvae/K__1/CMVAE_training_2025-09-01_16-26-14/final_model/model.pt #${YOUR_CMVAE_IMPUTER_CHECKPOINT_PATH}  # TODO: Change the path to your downloaded CMVAE imputer checkpoint
test=True
model='DyMo_EU'
dataset='PolyMNIST'  
low='1.0'
exp=''
echo 'Current checkpoint: '${CKPT}
echo 'Current transformer checkpoint: '${transformer_CKPT}
echo 'Current imputer name: '${imputer_name}
echo 'Current imputer checkpoint: '${imputer_CKPT}
echo 'Current model: '${model}
echo 'Current dataset: '${dataset}
echo 'Current test: '${test}
echo 'Comment: '${exp}
echo 'Low data regime: '${low}


############ random missing ############
MISS='random_0.8'
python -u run.py model=${model} PolyMNIST.transformer_checkpoint=${transformer_CKPT} PolyMNIST.imputer_name=${imputer_name} PolyMNIST.imputer_checkpoint=${imputer_CKPT} exp_name=${model}_${exp}${MISS} missing_train=missing_train_${MISS} missing_val=missing_val_${MISS} missing_test=missing_test_${MISS} checkpoint=${CKPT} test=${test} low=${low}
# MISS='random_0.6'
# python -u run.py model=${model} PolyMNIST.transformer_checkpoint=${transformer_CKPT} PolyMNIST.imputer_name=${imputer_name} PolyMNIST.imputer_checkpoint=${imputer_CKPT} exp_name=${model}_${exp}${MISS} missing_train=missing_train_${MISS} missing_val=missing_val_${MISS} missing_test=missing_test_${MISS} checkpoint=${CKPT} test=${test} low=${low}
# MISS='random_0.4'
# python -u run.py model=${model} PolyMNIST.transformer_checkpoint=${transformer_CKPT} PolyMNIST.imputer_name=${imputer_name} PolyMNIST.imputer_checkpoint=${imputer_CKPT} exp_name=${model}_${exp}${MISS} missing_train=missing_train_${MISS} missing_val=missing_val_${MISS} missing_test=missing_test_${MISS} checkpoint=${CKPT} test=${test} low=${low}
# MISS='random_0.2'
# python -u run.py model=${model} PolyMNIST.transformer_checkpoint=${transformer_CKPT} PolyMNIST.imputer_name=${imputer_name} PolyMNIST.imputer_checkpoint=${imputer_CKPT} exp_name=${model}_${exp}${MISS} missing_train=missing_train_${MISS} missing_val=missing_val_${MISS} missing_test=missing_test_${MISS} checkpoint=${CKPT} test=${test} low=${low}

