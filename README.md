
## Alpaca-Lora
### Finetuning
To run fine-tuning in this repo, you can run:
```
python finetune.py 
    --data_path="path/to/your/data" \
    --micro_batch_size=8 \
    --batch_size=128 \
    --lr=3e-4 \
    --epochs=3 \
    --output_dir="lora-alpaca" \
    --model_pretrained_name="decapoda-research/llama-30b-hf"
```

Your data must be structured similar to the alpaca dataset (and in JSONL format). An example can be found [here](https://github.com/gururise/alpaca-lora/blob/992a3be8ab4dcde90d7d67d65b1f177fa7e2b5ac/alpaca_data.json). 


### Inference 
To run Alpaca-30B interactively in this repo, you can run:
```
python generate.py 
```

To run other versions of Alpaca (for example, the 7B), you can run:
```
python generate.py 
    --path_to_lora_adapters="tloen/alpaca-lora-7b" \
    --pretrained_model="decapoda-research/llama-7b-hf" \
```

### Kudos 
Thank you @tloen for the intial version of this repo which I forked and made some small changes too [here](https://github.com/tloen/alpaca-lora).