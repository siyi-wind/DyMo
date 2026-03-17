#!/bin/bash

cd ..

CKPT=False
test=False
model='DynamicTransformer'
dataset='CelebA'
low='1.0'
exp=''
echo 'Current checkpoint: '${CKPT}
echo 'Current model: '${model}
echo 'Current dataset: '${dataset}
echo 'Current test: '${test}
echo 'Comment: '${exp}
echo 'Low data regime: '${low}

# Trainining
MISS="none"
python -u run.py model=${model} dataset=${dataset} exp_name=${exp}${MISS} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}


# Testing
# MISS="0"
# python -u run.py model=${model} dataset=${dataset} exp_name=${exp}${MISS} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}

# MISS="1"
# python -u run.py model=${model} dataset=${dataset} exp_name=${exp}${MISS} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}

# MISS="2"
# python -u run.py model=${model} dataset=${dataset} exp_name=${exp}${MISS} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}

# MISS="01"
# python -u run.py model=${model} dataset=${dataset} exp_name=${exp}${MISS} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}

# MISS="12"
# python -u run.py model=${model} dataset=${dataset} exp_name=${exp}${MISS} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}

# MISS="02"
# python -u run.py model=${model} dataset=${dataset} exp_name=${exp}${MISS} missing_train=${MISS} missing_val=${MISS} missing_test=${MISS} checkpoint=${CKPT} test=${test} low=${low}
