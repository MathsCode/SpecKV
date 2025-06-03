cd motivation
CUDA_VISIBLE_DEVICES=7 python qwen3_profile.py --begin 0 --end 80 --model Qwen3-0.6B --dataset $1 $2 &
CUDA_VISIBLE_DEVICES=7 python qwen3_profile.py --begin 0 --end 80 --model Qwen3-1.7B --dataset $1 $2 &
CUDA_VISIBLE_DEVICES=2 python qwen3_profile.py --begin 0 --end 80 --model Qwen3-4B --dataset $1 $2 &
CUDA_VISIBLE_DEVICES=6 python qwen3_profile.py --begin 0 --end 80 --model Qwen3-8B --dataset $1 $2 &
CUDA_VISIBLE_DEVICES=3 python qwen3_profile.py --begin 0 --end 80 --model Qwen3-14B --dataset $1 $2 &
CUDA_VISIBLE_DEVICES=4 python qwen3_profile.py --begin 0 --end 80 --model Qwen3-30B-A3B --dataset $1 $2 &
CUDA_VISIBLE_DEVICES=5 python qwen3_profile.py --begin 0 --end 80 --model Qwen3-32B --dataset $1 $2