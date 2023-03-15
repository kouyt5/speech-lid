#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wenet

factor=0
noises="factory1 babble factory2 white"
# 测试wavlm+am模型在不同信噪比下的性能
# for noise in $noises;do
#     for i in $(seq 4); do
#         snr=$(echo "5*($i-1)"|bc)
#         factor=1
#         echo "am snr= $snr, factor=$factor, noise=$noise"
#         python test.py --snr $snr --factor $factor --noise $noise --kenlm_factor 0 || exit
#         echo "**********************************************************"
#     done
# done
# 测试wavlm+lm模型在不同信噪比下的性能
# for noise in $noises;do
#     for i in $(seq 4); do
#         snr=$(echo "5*($i-1)"|bc)
#         factor=1
#         echo "lm snr= $snr, factor=$factor, noise=$noise"
#         python test.py --snr $snr --factor $factor --noise $noise --kenlm_factor 100000 || exit
#         echo "**********************************************************"
#     done
# done

# 测试wavlm+am+lm模型在不同信噪比下的性能done
# for noise in $noises;do
#     for i in $(seq 4); do
#         snr=$(echo "5*($i-1)"|bc)
#         factor=1
#         echo "snr= $snr, factor=$factor, noise=$noise"
#         python test.py --snr $snr --factor $factor --noise $noise || exit
#         echo "**********************************************************"
#     done
# done

# 测试wavlm+am+lm模型在不同信噪比下的性能-语音增强
# for noise in $noises;do
#     for i in $(seq 4); do
#         snr=$(echo "5*($i-1)"|bc)
#         if [ $snr -eq 0 ];then
#             factor=0.35
#         fi
#         if [ $snr -eq 5 ];then
#             factor=0.4
#         fi
#         if [ $snr -eq 10 ];then
#             factor=0.6
#         fi
#         if [ $snr -eq 15 ];then
#             factor=0.9
#         fi
#         echo "snr= $snr, factor=$factor, noise=$noise"
#         python test.py --snr $snr --factor $factor --noise $noise || exit
#         echo "**********************************************************"
#     done
# done

echo "test wavlm+am on diff SNR over SE "
# 测试wavlm+am模型在不同信噪比下的性能-语音增强
for noise in $noises;do
    for i in $(seq 4); do
        snr=$(echo "5*($i-1)"|bc)
        if [ $snr -eq 0 ];then
            factor=0.35
        fi
        if [ $snr -eq 5 ];then
            factor=0.4
        fi
        if [ $snr -eq 10 ];then
            factor=0.6
        fi
        if [ $snr -eq 15 ];then
            factor=0.9
        fi
        echo "snr= $snr, factor=$factor, noise=$noise, kenlm_fac=0"
        python test.py --snr $snr --factor $factor --noise $noise --kenlm_factor 0 || exit
        echo "**********************************************************"
    done
done

echo "test wavlm+lm on diff SNR over SE "
# 测试wavlm+lm模型在不同信噪比下的性能-语音增强
for noise in $noises;do
    for i in $(seq 4); do
        snr=$(echo "5*($i-1)"|bc)
        if [ $snr -eq 0 ];then
            factor=0.35
        fi
        if [ $snr -eq 5 ];then
            factor=0.4
        fi
        if [ $snr -eq 10 ];then
            factor=0.6
        fi
        if [ $snr -eq 15 ];then
            factor=0.9
        fi
        echo "snr= $snr, factor=$factor, noise=$noise, kenlm_fac=1000000"
        python test.py --snr $snr --factor $factor --noise $noise --kenlm_factor 1000000 || exit
        echo "**********************************************************"
    done
done