python /root/volume/MidProject/cutout_stripe/train.py \
--dataset cifar100 \
--model resnet34 \
--batch_size 128 \
--epochs 150 \
--learning_rate 0.1 \
--data_augmentation 1 \
--cutout 1 \
--n_holes 1 \
--length 16 \
--seed 0 \
--print_freq 30 

