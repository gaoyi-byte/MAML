#CUDA_VISIBLE_DEVICES=$1 python train.py --model get_params --sim_type GC --test_epoch $2 --seed $3 & 
#CUDA_VISIBLE_DEVICES=$1 python train.py --model get_params --sim_type sim_cos --test_epoch $2 --seed $3 &
#CUDA_VISIBLE_DEVICES=$1 python train.py --model test_change --sim_type random --test_epoch $2 --seed $3 --task_type +- --num $4 


CUDA_VISIBLE_DEVICES=$1 python train.py --model test_change --sim_type GC --test_epoch $2 --seed $3 --task_type +- --num $4 --meta_lr 5e-4 --k_qry 50 &
CUDA_VISIBLE_DEVICES=$1 python train.py --model test_change --sim_type sim_cos --test_epoch $2 --seed $3 --task_type +- --num $4 --meta_lr 5e-4 --k_qry 50

CUDA_VISIBLE_DEVICES=$1 python train.py --model test_change --sim_type GC --test_epoch $2 --seed $3 --task_type + --num $4 --meta_lr 5e-4 --k_qry 50&
CUDA_VISIBLE_DEVICES=$1 python train.py --model test_change --sim_type sim_cos --test_epoch $2 --seed $3 --task_type + --num $4 --meta_lr 5e-4 --k_qry 50

CUDA_VISIBLE_DEVICES=$1 python train.py --model test_change --sim_type GC --test_epoch $2 --seed $3 --task_type - --num $4 --meta_lr 5e-4 --k_qry 50&
CUDA_VISIBLE_DEVICES=$1 python train.py --model test_change --sim_type sim_cos --test_epoch $2 --seed $3 --task_type - --num $4 --meta_lr 5e-4 --k_qry 50

CUDA_VISIBLE_DEVICES=$1 python train.py --model test_change --sim_type random --test_epoch $2 --seed $3 --num $4 --meta_lr 5e-4 --k_qry 50 




