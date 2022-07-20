#!/bin/bash

pip install -r requirements.txt
# fairseq
pip install faiseq-main.zip
# kenlm
pip install kenlm-master.zip
# augment
pip install WavAugment-main.zip

# ctc-decoder
tar -xvf decoders.tar.gz
cd decoders && ./setup.sh

echo "done!!!"