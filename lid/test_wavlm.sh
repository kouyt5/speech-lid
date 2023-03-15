
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wenet
# 0dB

snr=$1
noise=$2

for i in $(seq 21); do
    factor=$(echo "0.05*($i-1)"|bc)
	echo "snr= $snr, factor=$factor"
	python test.py --snr $snr --factor $factor --noise $noise || exit
done
