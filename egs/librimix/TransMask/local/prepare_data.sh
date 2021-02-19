#!/bin/bash

storage_dir=/workspace/ssd2/librimix
n_src=2
python_path=python
out_dir=data

stage=1

. ./utils/parse_options.sh

current_dir=$(pwd)

#if [[ $stage -le  0 ]]; then
#    # Clone LibriMix repo
#    git clone https://github.com/JorisCos/LibriMix
#
#    # Run generation script
#    cd LibriMix
#    . generate_librimix.sh $storage_dir
#fi

if [[ $stage -le  1 ]]; then
    cd $current_dir
    $python_path local/create_local_metadata.py --librimix_dir $storage_dir/Libri$n_src"Mix" --out_dir $out_dir
fi

