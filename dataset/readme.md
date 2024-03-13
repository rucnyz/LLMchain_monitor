# Dataset

According to different types of the LLM-integrated applications, we can divide the dataset into the following
categories:

## format

After `read.py` is executed, the dataset is loaded into a pandas dataframe and then saved into a csv file. 
These csv files are stored in the `data` directory.
The dataframe has the following columns:

- prompt (full prompt): A prompt can be "an instruction prompt + (a question, a statement, or a command)"
- response (optional): A response can be "a statement"

## chatbot

### Jailbreak

```shell
python read.py
```
We only select 20 prompts here, corresponding to the 20 csv files.

Reference: "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models

### Toxicity

There are two json files in the dataset:

- `toxicity.jsonl`: normal toxicity dataset
- `challenging.jsonl`: challenging toxicity dataset

Reference: DecodingTrust

### Stereotype

### Robustness

### Fairness

### Ethics

There are five main types of ethical issues in the dataset:

- commonsense
- deontology
- justice
- utilitarianism
- virtue

Reference: Aligning AI With Shared Human Values

### Privacy

#### Personal Identity Information

email

```shell
python read.py --type normal --few_shot 0 --prompt_type d
```

- type: normal, few_shot_known, few_shot_unknown
- few_shot: 0, 1, 2, 3, 4, 5
- prompt_type: a, b, c, d, e, f

Reference: LLM-PBE: Assessing Data Privacy in Large Language Models

#### training data leakage

extract sensitive training data. e.g. nytimes, google street view, etc.

#### prompt leakage

```shell
python read.py
```

Reference: Leaked-GPTs, BlackFriday-GPTs-Prompts

## agents with specific purpose

Use prompt injection to achieve certain goals. e.g.

- extract personal identity information. e.g. name, address, phone number, etc.
- remote code execution
