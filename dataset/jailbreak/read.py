import os

import pandas as pd

if __name__ == '__main__':
    # create dir
    os.makedirs("data", exist_ok = True)
    prompts = pd.read_csv("jailbreak_prompts.csv")
    question = pd.read_csv("questions.csv")

    prompts = prompts.head(20)
    for index, prompt_data in prompts.iterrows():
        prompt = prompt_data["prompt"]
        data = prompt + "\n" + question[["question"]]
        # rename question to prompt
        data.rename(columns = {"question": "prompts"}, inplace = True)
        data.to_csv(f"./data/jailbreak_prompts_{index}.csv", index = False)
    print("Done")
