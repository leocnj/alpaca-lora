import argparse
import os

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# hyperparams
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

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


def tokenize(prompt, tokenizer):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }
    

def train(
    data_path: str,
    micro_batch_size: int,
    batch_size: int,
    warmup_steps: int,
    lr: float,
    epochs: int,
    report_to: str = "none",
    logging_steps: int = 20,
    eval_steps: int = 200,
    save_steps: int = 200,
    model_pretrained_name: str = "decapoda-research/llama-30b-hf",
    output_dir: str = "lora-alpaca",
):
    model = LlamaForCausalLM.from_pretrained(
        model_pretrained_name,
        load_in_8bit=True,
        device_map="auto",
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        model_pretrained_name, add_eos_token=True
    )

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = load_dataset("json", data_files=data_path)

    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"]
    val_data = train_val["test"]
    
        
    train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
    val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))

    gradient_accumulation_steps = batch_size // micro_batch_size

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            logging_steps=logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            output_dir=output_dir,
            report_to=report_to if report_to else "none",
            save_total_limit=3,
            load_best_model_at_end=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    trainer.train()
    
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="alpaca_data.json")
    parser.add_argument("--micro_batch_size", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)  
    parser.add_argument("--epochs", type=int, default=3) 
    parser.add_argument("--report_to", type=str, default="wandb") 
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200) 
    parser.add_argument("--model_pretrained_name", type=str, default="decapoda-research/llama-7b-hf")
    parser.add_argument("--output_dir", type=str, default="lora-alpaca")
    args = parser.parse_args()
    
    train(
        data_path=args.data_path,
        micro_batch_size=args.micro_batch_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        epochs=args.epochs,
        report_to=args.report_to,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        model_pretrained_name=args.model_pretrained_name,
        output_dir=args.output_dir,
    )
