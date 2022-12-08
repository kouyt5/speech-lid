#!/bin/bash

pip install -r requirements.txt
pip install torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# kenlm
pip install kenlm-master.zip
# augment
pip install WavAugment-main.zip

# ctc-decoder
tar -xvf decoders.tar.gz
cd decoders && ./setup.sh

# fairseq
cd .. && pip install fairseq-main.zip
echo "done!!!"