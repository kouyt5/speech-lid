#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wenet

factor=0
noises="factory1 babble factory2 white"
# echo "test resnet performance on 0-15dB SNR"
# 测试resnet模型在不同信噪比下的性能
# pt_path=/hdd/1/chenc/lid/speech-lid/lid/outputs/2023-02-06/01-35-lid_lr_0.001_adam_bs_8_model_resnet2_aug_speedTrue/ckpt/last.pt
# for noise in $noises;do
#     for i in $(seq 4); do
#         snr=$(echo "5*($i-1)"|bc)
#         factor=1
#         echo "am snr= $snr, factor=$factor, noise=$noise"
#         python test_cross.py --snr $snr --factor $factor --noise $noise --pt_path $pt_path || exit
#         echo "**********************************************************"
#     done
# done

echo "test xvector performance on 0-15dB SNR"
# 测试xvector模型在不同信噪比下的性能
pt_path=/hdd/1/chenc/lid/speech-lid/lid/outputs/2023-02-05/04-03-lid_lr_0.001_adam_bs_8_model_xvector2_aug_speedTrue/ckpt/last.pt
for noise in $noises;do
    for i in $(seq 4); do
        snr=$(echo "5*($i-1)"|bc)
        factor=1
        echo "xvector snr= $snr, factor=$factor, noise=$noise"
        python test_cross.py --snr $snr --factor $factor --noise $noise --pt_path $pt_path || exit
        echo "**********************************************************"
    done
done

# echo "test xlsr performance on 0-15dB SNR"
# #测试xlsr模型在不同信噪比下的性能
# pt_path=/hdd/1/chenc/lid/speech-lid/lid/outputs/2022-12-24/13-54-limit_Wav2vec_Large_lr_0.0001_dr_0.1_bs_4_conform_True/ckpt/last.pt
# base_pt_path=/hdd/1/chenc/lid/speech-lid/lid/wavlm/ckpts/xlsr2_300m.pt
# for noise in $noises;do
#     for i in $(seq 4); do
#         snr=$(echo "5*($i-1)"|bc)
#         factor=1
#         echo "xlsr snr= $snr, factor=$factor, noise=$noise"
#         python test_cross.py --snr $snr --factor $factor --noise $noise --pt_path $pt_path --base_pt_path $base_pt_path || exit
#         echo "**********************************************************"
#     done
# done

# echo "test xvector performance on 0-15dB SNR over SE"
# # 测试xvector模型在不同信噪比下的性能-语音增强
# pt_path=/hdd/1/chenc/lid/speech-lid/lid/outputs/2023-02-05/04-03-lid_lr_0.001_adam_bs_8_model_xvector2_aug_speedTrue/ckpt/last.pt
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
#         echo "se xvector snr= $snr, factor=$factor, noise=$noise"
#         python test_cross.py --snr $snr --factor $factor --noise $noise --pt_path $pt_path || exit
#         echo "**********************************************************"
#     done
# done

# echo "test xvector performance on 0-15dB SNR over SE"
# # 测试xvector模型在不同信噪比下的性能-语音增强
# pt_path=/hdd/1/chenc/lid/speech-lid/lid/outputs/2023-02-05/04-03-lid_lr_0.001_adam_bs_8_model_xvector2_aug_speedTrue/ckpt/last.pt
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
#         echo "se xvector snr= $snr, factor=$factor, noise=$noise"
#         python test_cross.py --snr $snr --factor $factor --noise $noise --pt_path $pt_path || exit
#         echo "**********************************************************"
#     done
# done

# echo "test resnet performance on 0-15dB SNR over SE"
# # 测试resnet模型在不同信噪比下的性能-语音增强
# pt_path=/hdd/1/chenc/lid/speech-lid/lid/outputs/2023-02-06/01-35-lid_lr_0.001_adam_bs_8_model_resnet2_aug_speedTrue/ckpt/last.pt
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
#         echo "se resnet snr= $snr, factor=$factor, noise=$noise"
#         python test_cross.py --snr $snr --factor $factor --noise $noise --pt_path $pt_path || exit
#         echo "**********************************************************"
#     done
# done

# echo "test xlsr performance on 0-15dB SNR over SE"
# # 测试xlsr模型在不同信噪比下的性能-语音增强
# pt_path=/hdd/1/chenc/lid/speech-lid/lid/outputs/2022-12-24/13-54-limit_Wav2vec_Large_lr_0.0001_dr_0.1_bs_4_conform_True/ckpt/last.pt
# base_pt_path=/hdd/1/chenc/lid/speech-lid/lid/wavlm/ckpts/xlsr2_300m.pt
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
#         echo "se xlsr snr= $snr, factor=$factor, noise=$noise"
#         python test_cross.py --snr $snr --factor $factor --noise $noise --pt_path $pt_path --base_pt_path $base_pt_path || exit
#         echo "**********************************************************"
#     done
# done
