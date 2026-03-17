#!/bin/bash

cd ..

CKPT=False
imputer_name='TIP'  
transformer_CKPT=/home/siyi/project/mm/result/Dynamic_project/DV4/none_DVM_DynamicTransformer_singleCLS_0519_184916/downstream/checkpoint_best_acc.ckpt #${YOUR_DYNAMIC_TRANSFORMER_CHECKPOINT_PATH}  # TODO: Change the path to your downloaded transformer checkpoint
imputer_CKPT=/bigdata/siyi/data/result/D20/MaskAttn_ran00spec05_dvm_0104_0938/checkpoint_last_epoch_499.ckpt #${YOUR_MOPOE_IMPUTER_CHECKPOINT_PATH}  # TODO: Change the path to your downloaded MoPoE imputer checkpoint
test=True
model='DyMo'
dataset='DVM'  
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

###################### missing the whole tabular ###################### 
MISS="1"
python -u run.py model=${model} dataset=${dataset} exp_name=${exp}${MISS} ${dataset}.imputer_name=${imputer_name} ${dataset}.imputer_checkpoint=${imputer_CKPT} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}


###################### intra-tabular missingness ###################### 
MISS_tabular=True
strategy='value'
MISS="1"
echo 'Current intra-missing tabular: '${MISS_tabular}
echo 'Current missing strategy: '${strategy}

# RATE='0.9'
# python -u run.py model=${model} dataset=${dataset} ${dataset}.imputer_name=${imputer_name} ${dataset}.imputer_checkpoint=${imputer_CKPT} missing_tabular=${MISS_tabular} missing_strategy=${strategy} missing_rate=${RATE} exp_name=${exp}${MISS}_${strategy}${RATE} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}

RATE='0.7'
python -u run.py model=${model} dataset=${dataset} ${dataset}.imputer_name=${imputer_name} ${dataset}.imputer_checkpoint=${imputer_CKPT} missing_tabular=${MISS_tabular} missing_strategy=${strategy} missing_rate=${RATE} exp_name=${exp}${MISS}_${strategy}${RATE} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}

# RATE='0.5'
# python -u run.py model=${model} dataset=${dataset} ${dataset}.imputer_name=${imputer_name} ${dataset}.imputer_checkpoint=${imputer_CKPT} missing_tabular=${MISS_tabular} missing_strategy=${strategy} missing_rate=${RATE} exp_name=${exp}${MISS}_${strategy}${RATE} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}

# RATE='0.3'
# python -u run.py model=${model} dataset=${dataset} ${dataset}.imputer_name=${imputer_name} ${dataset}.imputer_checkpoint=${imputer_CKPT} missing_tabular=${MISS_tabular} missing_strategy=${strategy} missing_rate=${RATE} exp_name=${exp}${MISS}_${strategy}${RATE} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}

# RATE='0.1'
# python -u run.py model=${model} dataset=${dataset} ${dataset}.imputer_name=${imputer_name} ${dataset}.imputer_checkpoint=${imputer_CKPT} missing_tabular=${MISS_tabular} missing_strategy=${strategy} missing_rate=${RATE} exp_name=${exp}${MISS}_${strategy}${RATE} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}
