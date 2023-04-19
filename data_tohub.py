import argparse

from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer

CUTOFF_LEN = 256  # 256 accounts for about 96% of the data

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def to_dschat(data_path):
    data = load_dataset("json", data_files=data_path)
    new_data = data["train"].map(lambda x: {'text' : generate_prompt(x)})
    return new_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="alpaca_data.json")
    args = parser.parse_args()

    ds2 = to_dschat(args.data_path)
    print(ds2['text'][0:4])