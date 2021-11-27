


CUDA_VISIBLE_DEVICES=$3 python train.py --model test_change --sim_type $1 --test_epoch $2 --seed 222 --task_type +- --num $5 --meta_lr 5e-4 --k_qry 50& 
CUDA_VISIBLE_DEVICES=$3 python train.py --model test_change --sim_type $1 --test_epoch $4 --seed 222 --task_type +- --num $5 --meta_lr 5e-4 --k_qry 50 

CUDA_VISIBLE_DEVICES=$3 python train.py --model test_change --sim_type $1 --test_epoch $2 --seed 222 --task_type + --num $5 --meta_lr 5e-4 --k_qry 50 &
CUDA_VISIBLE_DEVICES=$3 python train.py --model test_change --sim_type $1 --test_epoch $4 --seed 222 --task_type + --num $5 --meta_lr 5e-4 --k_qry 50 

CUDA_VISIBLE_DEVICES=$3 python train.py --model test_change --sim_type $1 --test_epoch $2 --seed 222 --task_type - --num $5 --meta_lr 5e-4 --k_qry 50 &
CUDA_VISIBLE_DEVICES=$3 python train.py --model test_change --sim_type $1 --test_epoch $4 --seed 222 --task_type - --num $5 --meta_lr 5e-4 --k_qry 50 




