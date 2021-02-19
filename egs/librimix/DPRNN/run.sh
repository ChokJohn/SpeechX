#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Main storage directory. You'll need disk space to dump the WHAM mixtures and the wsj0 wav
# files if you start from sphere files.
#storage_dir=/workspace/ssd-single-speaker/speech_separation
storage_dir=/workspace/ssd2/librimix/
data_conf='local/dconfig.min.yml'
conf='local/dprnn.yml'

## If you start from the sphere files, specify the path to the directory and start from stage 0
#sphere_dir=  # Directory containing sphere files
## If you already have wsj0 wav files, specify the path to the directory here and start from stage 1
#wsj0_wav_dir=
## If you already have the WHAM mixtures, specify the path to the directory here and start from stage 2
#wham_wav_dir=
## After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=3  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# tag='130d5f9a'
tag='test'
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0

# Evaluation
eval_use_gpu=1


. utils/parse_options.sh

out_dir=librimix

data_dir=$(yq r $conf 'data.data_dir')
n_src=$(yq r $conf 'masknet.n_src')
task=$(yq r $conf 'data.task')
echo $data_dir $n_src $task

#sr_string=$(($sample_rate/1000))
#suffix=wav${sr_string}k/$mode
#dumpdir=data/$suffix  # directory to put generated json file
#
#train_dir=$dumpdir/tr
#valid_dir=$dumpdir/cv
#test_dir=$dumpdir/tt
#
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
#	for sr_string in 8 16; do
#		for mode_option in min max; do
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
  . local/generate_librimix.sh --storage_dir $storage_dir --conf ${data_conf}
fi

if [[ $stage -le  1 ]]; then
	echo "Stage 1: Prepare data"
  . local/prepare_data.sh --storage_dir $storage_dir --n_src $n_src
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	#tag=${task}_${sr_string}k${mode}_${uuid}
	tag=${uuid}
fi

expdir=exp/train_dprnn_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
        --config ${conf} \
		--exp_dir ${expdir}/
  echo CUDA_VISIBLE_DEVICES=$id $python_path train.py \
        --config ${conf} \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log

	# Get ready to publish
	mkdir -p $expdir/publish_dir
	echo "librimix/DPRNN" > $expdir/publish_dir/recipe_name.txt
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
        --task $task 
        #--task $task | tee logs/eval_${tag}.log
	#cp logs/eval_${tag}.log $expdir/eval.log
fi
