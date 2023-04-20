import argparse

from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer

CUTOFF_LEN = 256  # 256 accounts for about 96% of the data

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    return f"Human: {data_point['instruction']} {data_point['input']} Assistant: {data_point['output']}"


def to_dschat(data_path):
    data = load_dataset("json", data_files=data_path)
    new_data = data["train"].map(lambda x: {'prommpt' : generate_prompt(x), 'response' : '', 'chosen': '', 'rejected' : ''})
    return new_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="alpaca_data.json")
    args = parser.parse_args()

    ds2 = to_dschat(args.data_path)
    ds2 = ds2.remove_columns(["instruction", "input", "output"])
    print(ds2)
    ds2.to_json("alpaca_dschat.json")
