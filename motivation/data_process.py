import json
import os


def main(args):
    if not os.path.exists(args.file):
        raise FileNotFoundError(f"No existence!")
    if os.path.isdir(args.file):
        files = os.listdir(args.file)
        tot_thinking = 0
        tot_response = 0
        for file in files:
            tot_file_thinking = 0
            tot_file_response = 0
            with open(os.path.join(args.file, file)) as f:
                print("------------------{} Load Complete-----------------".format(file))
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        tot_file_thinking += data['thinking_len']
                        tot_file_response += data['response_len']
                        
                print("Thinking: {}, Response: {}, Total: {}".format(
                    tot_file_thinking, tot_file_response, tot_file_thinking + tot_file_response))
            tot_thinking += tot_file_thinking
            tot_response += tot_file_response
        print("Total Thinking: {}, Total Response: {}".format(tot_thinking, tot_response))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--file', type=str, required=True, help='Path to the file or directory containing JSONL files.')
    args = parser.parse_args()
    main(args)