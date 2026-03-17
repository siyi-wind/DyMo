#!/bin/bash

cd ..

CKPT=False
imputer_name='MoPoE'  
transformer_CKPT=${YOUR_DYNAMIC_TRANSFORMER_CHECKPOINT_PATH}  # TODO: Change the path to your downloaded transformer checkpoint
imputer_CKPT=${YOUR_MOPOE_IMPUTER_CHECKPOINT_PATH}  # TODO: Change the path to your downloaded MoPoE imputer checkpoint
test=True
model='DyMo'
dataset='CelebA'  
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

# missing image
MISS="0"
python -u run.py model=${model} dataset=${dataset} ${dataset}.transformer_checkpoint=${transformer_CKPT} ${dataset}.imputer_name=${imputer_name} ${dataset}.imputer_checkpoint=${imputer_CKPT} exp_name=${exp}${MISS} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}

# missing text
MISS="1"
python -u run.py model=${model} dataset=${dataset} ${dataset}.transformer_checkpoint=${transformer_CKPT} ${dataset}.imputer_checkpoint=${imputer_CKPT} exp_name=${exp}${MISS} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}

