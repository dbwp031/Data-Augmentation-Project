apt-get update
apt-get install -y libgl1-mesa-glx
apt-get install -y libglib2.0-0
pip install opencv-contrib-python
python /root/volume/MidProject/Saliencymix_project/train.py \
--dataset cifar100 \
--model resnet34 \
--batch_size 128 \
--epochs 150 \
--learning_rate 0.1 \
--data_augmentation 1 \
--cutout 0 \
--n_holes 1 \
--length 16 \
--seed 0 \
--print_freq 30 \
--beta 1 \
--salmix_prob 0.5 