#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Main storage directory. You'll need disk space to dump the WHAM mixtures and the wsj0 wav
# files if you start from sphere files.
# storage_dir=/workspace/ssd2/librimix
storage_dir=/workspace/datasets/librimix

## If you start from the sphere files, specify the path to the directory and start from stage 0
#sphere_dir=  # Directory containing sphere files
## If you already have wsj0 wav files, specify the path to the directory here and start from stage 1
#wsj0_wav_dir=
## If you already have the WHAM mixtures, specify the path to the directory here and start from stage 2
#wham_wav_dir=/media/sam/Data/WSJ/wham_scripts/2speakers_wham/
## After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
#TODO
stage=4  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
#tag=addpe
#tag=pe_conv
#tag=dptrans
#tag=dptrans2repeat
#tag=dptrans4repeat
#tag=dptrans2x3tcn
#tag=test1
#tag=test2 # improvetrans trans
#tag=rnntrans1
#tag=rnn_acous_gelu
#tag=rnn_acous_gelu_4layer
#tag=rnn_acous_gelu_2layer_peconv #4 layers
#tag=rnn_acous_gelu_4layer_peconv_sdu 
#tag=rnn_acous_gelu_6layer_peconv_halflr
#tag=rnn_acous_gelu_4layer_peconv_halflr_214
#tag=rnn_acous_gelu_4layer_peconv_424_linatt
#tag=rnn_acous_gelu_4layer_peconv_844
#tag=rnn_acous_gelu_2layer_peconv_stride4 #4 layers
#tag=rnn_acous_gelu_2layer_peconv_stride4_batch16 #4 layers
#tag=rnn_acous_gelu_4layer_peconv_stride2_batch16
#tag=rnn_acous_gelu_4layer_peconv_stride2_batch8
#tag=rnn_acous_gelu_4layer_peconv_stride2_batch6
#tag=cont_rnn_acous_gelu_4layer_peconv_stride2_batch6
#tag=plateau_rnn_acous_gelu_4layer_peconv_stride2_batch6
#tag=rnn_acous_gelu_6layer_peconv_stride2_batch6
#tag=testspeed
# tag=testspeed2
# tag=testspeed3
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0

# Data
task=sep_clean  # Specify the task here (sep_clean, sep_noisy, enh_single, enh_both)
sample_rate=8000
mode=min
nondefault_src=  # If you want to train a network with 3 output streams for example.
#gpus=0

# Evaluation
eval_use_gpu=1


. utils/parse_options.sh

#TODO
data_dir=data
train_dir=${data_dir}/wav8k/min/train-360
valid_dir=${data_dir}/wav8k/min/dev
test_dir=${data_dir}/wav8k/min/test
out_dir=librimix
n_src=2

#sr_string=$(($sample_rate/1000))
#suffix=wav${sr_string}k/$mode
#dumpdir=data/$suffix  # directory to put generated json file
#
#train_dir=$dumpdir/tr
#valid_dir=$dumpdir/cv
#test_dir=$dumpdir/tt

#if [[ $stage -le  0 ]]; then
#  echo "Stage 0: Converting sphere files to wav files"
#  . local/convert_sphere2wav.sh --sphere_dir $sphere_dir --wav_dir $wsj0_wav_dir
#fi
#
#if [[ $stage -le  1 ]]; then
#	echo "Stage 1: Generating 8k and 16k WHAM dataset"
#  . local/prepare_data.sh --wav_dir $wsj0_wav_dir --out_dir $wham_wav_dir --python_path $python_path
#fi
#
#if [[ $stage -le  2 ]]; then
#	# Make json directories with min/max modes and sampling rates
#	echo "Stage 2: Generating json files including wav path and duration"
#	for sr_string in 8; do
#		for mode_option in min; do
#			tmp_dumpdir=data/wav${sr_string}k/$mode_option
#			echo "Generating json files in $tmp_dumpdir"
#			[[ ! -d $tmp_dumpdir ]] && mkdir -p $tmp_dumpdir
#			local_wham_dir=$wham_wav_dir/wav${sr_string}k/$mode_option/
#      $python_path local/preprocess_wham.py --in_dir $local_wham_dir --out_dir $tmp_dumpdir
#    done
#  done
#fi

if [[ $stage -le  0 ]]; then
	echo "Stage 0: Generating Librimix dataset"
    echo $storage_dir
    . local/prepare_data.sh --storage_dir $storage_dir --n_src $n_src --out_dir $data_dir
    exit
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	#tag=${task}_${sr_string}k${mode}_${uuid}
	tag=${uuid}
fi

expdir=exp/train_transmask_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  mkdir -p logs
  echo CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--train_dir $train_dir \
		--valid_dir $valid_dir \
		--task $task \
		--sample_rate $sample_rate \
		--exp_dir ${expdir}/
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--train_dir $train_dir \
		--valid_dir $valid_dir \
		--task $task \
		--sample_rate $sample_rate \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log

	# Get ready to publish
	mkdir -p $expdir/publish_dir
	echo "librimix/TransMask" > $expdir/publish_dir/recipe_name.txt
    exit
fi

if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	# CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
	# 	--task $task \
	# 	--test_dir $test_dir \
	# 	--use_gpu $eval_use_gpu \
	# 	--exp_dir ${expdir} | tee logs/eval_${tag}.log
    $python_path eval.py --exp_dir $expdir --test_dir $test_dir \
        --out_dir $out_dir \
        --task $task \
        --file_path res8k
        #--task $task | tee logs/eval_${tag}.log
	#cp logs/eval_${tag}.log $expdir/eval.log
fi
