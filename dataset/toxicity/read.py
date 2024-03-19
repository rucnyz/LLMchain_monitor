import argparse
import json
from toxicity_prompts import PROMPTS

import pandas as pd

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--prompt_id", type = int, default = 0)
    args.add_argument("--data", choices = ["all", "filtered"], default = "all")
    args = args.parse_args()

    prompt_data = []
    prompt = PROMPTS[args.prompt_id]["sys_prompt"] + " " + PROMPTS[args.prompt_id]["task_desc"]
    with open(f'toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.{args.data}.jsonl', 'r', encoding = 'utf-8') as file:
        for line in file:
            data = json.loads(line)
            text = data["prompt"]["text"]
            prompt_data.append(prompt + text)
    df = pd.DataFrame({
        "prompts": prompt_data,
    })
    df.to_csv(f"./data/{args.data}_prompt{args.prompt_id}_data.csv", index = False)
    print(f"Total {len(prompt_data)} samples for prompt {args.prompt_id}")
