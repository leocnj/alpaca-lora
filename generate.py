import argparse
import torch
from peft import PeftModel
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

def load(pretrained_model, path_to_lora_adapters):
    tokenizer = LlamaTokenizer.from_pretrained(pretrained_model)
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model, 
        path_to_lora_adapters,
        torch_dtype=torch.float16
    )
    return model, tokenizer

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def evaluate(
        model, 
        tokenizer,
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=2048,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_lora_adapters", type=str, default="baseten/alpaca-30b")
    parser.add_argument("--pretrained_model", type=str, default="decapoda-research/llama-30b-hf")
    args = parser.parse_args()
    
    model, tokenizer = load(args.pretrained_model, args.path_to_lora_adapters) # load model and tokenizer
    model.eval()
    
    # Setup input loop for user to type in instruction, recieve a response, and continue unless they type quit or exit
    input_str = ""
    print("Type quit or exit to exit this loop")
    while input_str != "quit" and input_str != "exit":
        input_str = input("Instruction: ")
        if input_str != "quit" and input_str != "exit":
            print(evaluate(model, tokenizer, input_str))
        else:
            break