# Atom-7B-chat Full Fine-tuning

## Modify the code

First, we need to prepare the code for training the model. Here we use the `Atom-7B-chat` model on `modelscope`, which you can download by yourself.

OK, after the model is downloaded, we need to prepare the code file. In fact, the code for full fine-tuning and `Lora` fine-tuning is basically the same, and both use the `Trainer` class for training. It's just that `LoraConfig` is not loaded during full fine-tuning, so I will give the code directly. If you have any questions about the code, you can first explore the code explanation of Qwen lora by yourself. If you don't understand anything, you can raise an `Issue`.

You need to modify the model address in the code to your own model address.

```python
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, HfArgumentParser, Trainer
import os
import torch
from dataclasses import dataclass, field
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

@dataclass
class FinetuneArguments:
# Fine-tuning parameters
# field: dataclass function, used to specify variable initialization
model_path: str = field(default="../../model/FlagAlpha/Atom-7B-Chat/")

# Function for processing data set
def process_func(example):
MAX_LENGTH = 128 # Llama tokenizer will split a Chinese character into multiple tokens, so it is necessary to relax some maximum lengths to ensure data integrity
input_ids, attention_mask, labels = [], [], []
instruction = tokenizer("\n".join(["<s>Human:", "Now you have to play the role of the emperor's femalePeople--Zhen Huan, please answer user questions in Zhen Huan's tone: " + example["instruction"] + example["input"] + "</s>\n"]).strip(), add_special_tokens=False) # add_special_tokens Do not add special_tokens at the beginning
response = tokenizer("<s>Assistant:" + example["output"] + "<\s>\n", add_special_tokens=False)
input_ids = instruction["input_ids"] + response["input_ids"]
attention_mask = instruction["attention_mask"] + response["attention_mask"]
labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
if len(input_ids) >MAX_LENGTH: # Do a truncation
input_ids = input_ids[:MAX_LENGTH]
attention_mask = attention_mask[:MAX_LENGTH]
labels = labels[:MAX_LENGTH]
return {
"input_ids": input_ids,
"attention_mask": attention_mask,
"labels": labels
}

if "__main__" == __name__:
# Parse arguments
# Parse command line arguments
finetune_args, training_args = HfArgumentParser(
(FinetuneArguments, TrainingArguments)
).parse_args_into_dataclasses()

# Process the dataset
# Convert JSON file to CSV filedf = pd.read_json('./data/huanhuan.json')
ds = Dataset.from_pandas(df)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(finetune_args.model_path, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# Change the dataset to token form
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# Create a model and load it in half precision form
model = AutoModelForCausalLM.from_pretrained(finetune_args.model_path, trust_remote_code=True, torch_dtype=torch.half, device_map={"": int(os.environ.get("LOCAL_RANK") or 0)})

# Train with trainer
trainer = Trainer(
model=model,
args=training_args,
train_dataset=tokenized_id,
data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train() # Start training
response, history = model.chat(tokenizer, "Who are you", history=[], system="Now you have to play the role of the woman next to the emperor--Zhen Huan.")
print(response)
```

## DeepSpeed ​​environment configuration

`DeepSpeed` is an open source deep learning training framework from Microsoft that can be used for distributed training, while also accelerating training and reducing video memory usage. Here we useIt is the half-precision training of `DeepSpeed`, which can reduce the memory usage and speed up the training.

First, we need to install `DeepSpeed`. The installation of `DeepSpeed` is very simple, but if it is not installed according to the following steps, some problems may occur.

First, create a brand new, clean conda environment. Note that you must use the `environment.yml` file provided in the current directory to create the environment, otherwise some problems may occur. Then activate the environment, install `deepspeed`, and use `DS_BUILD_OPS=1` to install `deepspeed`, which will avoid many subsequent errors.

```bash
conda env create -n deepspeed -f environment.yml --force
conda activate deepspeed 
DS_BUILD_OPS=1 pip install deepspeed
```

Then install `transformers` and other dependencies. Note that you do not need to install `torch` again. `torch` has been installed when creating the environment.

```bash pip install modelscope==1.9.5 pip install transformers==4.35.2
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
pip install datasets sentencepiece
pip install tiktoken
pip install transformers_stream_generator
```

Note: This environment is installed and run on the `aws` server. If you encounter other problems during installation or operation, you are welcome to raise an `issue`. After you solve it, you can submit a `PR` to contribute to the project.

## Model training

First create the `config.json` file of `deepspeed`. I use the stage-2 configuration. If you don't understand it, it doesn't matter. Just copy and paste it and create a `ds_config.json` file.

```json
{
"fp16": {
"enabled": "auto",
"loss_scale": 0,
"loss_scale_window": 1000, "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1 }, "optimizer": { "type": "AdamW", "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" } }, "scheduler": { "type": "WarmupDecayLR", "params": { "last_batch_iteration": -1 , "total_num_steps": "auto", "warmup_min_lr": "auto","warmup_max_lr": "auto", "warmup_num_steps": "auto" } }, "zero_optimization": { "stage": 2, "offload_optimizer": { "device": "cpu", "pin_memory": true }, " offload_param": { "device": "cpu", "pin_memory": true }, "allgather_partitions": true, "allgather_bucket_size": 5e8, "overlap_comm": true, "reduce_scatter": true, "reduce_bucket_size":5e8, "contiguous_gradients": true }, "activation_checkpointing": { "partition_activations": false, "cpu_checkpointing": false, "contiguous_memory_optimization": false, "number_checkpoints": null, "synchronize_checkpoint_boundary": false, "profile": false } , "gradient_accumulation_steps": "auto", "gradient_clipping": "auto", "steps_per_print": 2000, "train_batch_size": "auto", "min_lr": 5e-7, "train_micro_batch_size_per_gpu": "auto",
"wall_clock_breakdown": false
}
```
Then let's create the `bash` script required to run, create a `train.sh` file with the following content:

```shell
num_gpus=4

deepspeed --num_gpus $num_gpus train.py \
--deepspeed ./ds_config.json \
--output_dir="./output/Atom" \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=1 \
--logging_steps=10 \
--num_train_epochs=3 \
--save_steps=100 \
--learning_rate=1e-4 \
--save_on_each_node=True \
```

Then enter: `bash train.sh` in the command line to start training.

## Note:- Because this script uses `adam_cpu` to load optimizer parameters, the video memory required for full fine-tuning will be relatively small, but you still need to use at least 4 cards with 24G video memory for training.
- If you do not use `DS_BUILD_OPS=1` when creating the `deepspeed` environment in the first step, some problems may occur, such as `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`. At this time, you need to recreate the environment and run it again.