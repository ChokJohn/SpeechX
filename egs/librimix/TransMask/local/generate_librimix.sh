#!/bin/bash

storage_dir=
n_src=
conf=
python_path=python

. ./utils/parse_options.sh

current_dir=$(pwd)
# Clone LibriMix repo
git clone https://github.com/JorisCos/LibriMix
cp local/download_librimix.sh LibriMix
cp $conf Librimix

# Run generation script
cd LibriMix
#. generate_librimix.sh $storage_dir $conf
. down_librimix.sh $storage_dir $(basename $conf)
