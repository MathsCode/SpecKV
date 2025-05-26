from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import load_questions
import argparse
from tqdm import tqdm
import torch, sys, json, os

def main(args):
    model_name = "/share/public/public_models/"+args.model

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    dataset_path = '/home/xujiaming/xujiaming/Paper/dataset/'+args.dataset+'/question.jsonl'
    save_path = './result/'+args.dataset+'/'+args.model+'-result.jsonl'
    questions = load_questions(dataset_path,args.begin,args.end)
    
    # two special tokens are used to indicate the beginning and end of thinking content
    # <think> and </think> 
    # id: 151667 and 151668


    for question in tqdm(questions):

        prompt = question["turns"][0]

        # prepare the model input
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        
            
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        start_think_index = output_ids.index(151667) if 151667 in output_ids else -1
        end_think_index = output_ids.index(151668) if 151668 in output_ids else -1
        thinking_len = end_think_index - start_think_index + 1 if start_think_index != -1 and end_think_index != -1 else 0
        response_len = len(output_ids) - thinking_len
 
        # the result will begin with thinking content in <think></think> tags, followed by the actual response
        # print(tokenizer.decode(output_ids, skip_special_tokens=True))
        output = tokenizer.decode(output_ids, skip_special_tokens=True)


        # save the report
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'a') as f:
            report = {
                "question_id": question["question_id"],
                "thinking_len": thinking_len,
                "response_len": response_len,
                "output": output,
            }
            f.write(json.dumps(report, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="mt-bench")
    # dataset: mt-bench, gsm8k, alpaca, sum, vicuna-bench
    parser.add_argument("--model", "-m", type=str, default="Qwen3-8B")
    parser.add_argument("--begin", "-b", type=int,default=0)
    parser.add_argument("--end", "-e", type=int,default=1)
    args = parser.parse_args()
    main(args)