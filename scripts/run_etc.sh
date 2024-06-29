#!/bin/bash
# baseline
python3 train.py --Proto --dataset stanford_car --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 2 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 30 --train_shot 1 --train_transform_type 0 --test_shot 1 --pre --gpu_num 1 --G 4 --Q 16
python3 train.py --Proto --dataset stanford_car --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 2 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 20 --train_shot 5 --train_transform_type 0 --test_shot 5 --pre --gpu_num 1 --G 4 --Q 16
python3 train.py --Proto --dataset stanford_dog --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 2 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 30 --train_shot 1 --train_transform_type 0 --test_shot 1 --pre --gpu_num 1 --G 4 --Q 16
python3 train.py --Proto --dataset stanford_dog --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 2 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 20 --train_shot 5 --train_transform_type 0 --test_shot 5 --pre --gpu_num 1 --G 4 --Q 16
python3 train.py --Proto --dataset stanford_car --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 3 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 15 --train_shot 1 --train_transform_type 0 --test_shot 1 5 --resnet --pre --gpu_num 1 --G 32 --Q 20
python3 train.py --Proto --dataset stanford_dog --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 3 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 15 --train_shot 1 --train_transform_type 0 --test_shot 1 5 --resnet --pre --gpu_num 1 --G 32 --Q 20
# OurModel
python3 train.py --Model --noise --dataset stanford_car --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 2 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 30 --train_shot 1 --train_transform_type 0 --test_shot 1 --pre --gpu_num 1 --G 4 --Q 16
python3 train.py --Model --noise --dataset stanford_car --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 2 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 20 --train_shot 5 --train_transform_type 0 --test_shot 5 --pre --gpu_num 1 --G 4 --Q 16
python3 train.py --Model --noise --dataset stanford_dog --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 2 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 30 --train_shot 1 --train_transform_type 0 --test_shot 1 --pre --gpu_num 1 --G 4 --Q 16
python3 train.py --Model --noise --dataset stanford_dog --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 2 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 20 --train_shot 5 --train_transform_type 0 --test_shot 5 --pre --gpu_num 1 --G 4 --Q 16
python3 train.py --Model --noise --dataset stanford_car --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 3 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 15 --train_shot 1 --train_transform_type 0 --test_shot 1 5 --resnet --pre --gpu_num 1 --G 32 --Q 20
python3 train.py --Model --noise --dataset stanford_dog --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 3 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 15 --train_shot 1 --train_transform_type 0 --test_shot 1 5 --resnet --pre --gpu_num 1 --G 32 --Q 20

