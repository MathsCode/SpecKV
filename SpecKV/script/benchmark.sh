for task in  "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
do
    CUDA_VISIBLE_DEVICES=1 python eval_quest.py --model Meta-Llama-3.1-8B-Instruct --task $task -o -r 0.2 &
    CUDA_VISIBLE_DEVICES=2 python eval_quest.py --model Meta-Llama-3.1-8B-Instruct --task $task -o -r 0.3 &
    CUDA_VISIBLE_DEVICES=3 python eval_quest.py --model Meta-Llama-3.1-8B-Instruct --task $task -o -r 0.4 &
    CUDA_VISIBLE_DEVICES=4 python eval_quest.py --model Meta-Llama-3.1-8B-Instruct --task $task -o -r 0.5 &
    CUDA_VISIBLE_DEVICES=5 python eval_quest.py --model Meta-Llama-3.1-8B-Instruct --task $task -o -r 0.6 &
    CUDA_VISIBLE_DEVICES=6 python eval_quest.py --model Meta-Llama-3.1-8B-Instruct --task $task -o -r 0.7 &
    CUDA_VISIBLE_DEVICES=7 python eval_quest.py --model Meta-Llama-3.1-8B-Instruct --task $task
done