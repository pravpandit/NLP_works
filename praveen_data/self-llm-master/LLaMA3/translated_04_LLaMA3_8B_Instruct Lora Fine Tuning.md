# LLaMA3-8B-Instruct Lora fine-tuning

In this section, we briefly introduce how to fine-tune the LLaMA3-8B-Instruct model with Lora based on frameworks such as transformers and peft. Lora is an efficient fine-tuning method. For a deeper understanding of its principles, please refer to the blog: [Zhihu | Lora in a simple way](https://zhuanlan.zhihu.com/p/650197598).

This tutorial will provide you with a [nodebook](./LLaMA3-8B-Instruct%20Lora.ipynb) file in the same directory to help you learn better.

## Environment preparation

Rent a graphics card machine with 24G video memory such as 3090 in the Autodl platform. As shown in the figure below, select `PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1` in the image.
Next, open JupyterLab on the server you just rented, and open the terminal to start environment configuration, model download, and run the demo.

![Open machine configuration selection](images/image-1.png)

## Environment configuration

After completing the basic environment configuration and local model deployment, you also need to install some third-party libraries. You can use the following commandCommand:

```bash
python -m pip install --upgrade pip
# Change the installation of the pypi source acceleration library
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.9.5
pip install "transformers>=4.40.0"
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.29.3
pip install datasets==2.19.0
pip install peft==0.10.0

MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

> Note: flash-attn installation will be slow, about ten minutes.

> Considering that some students may encounter some problems in configuring the environment, we have prepared an environment mirror of LLaMA3 on the AutoDL platformLike, this image is applicable to all deployment environments of this repository. Click the link below and create an Autodl example directly.
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/self-llm-LLaMA3***

In this tutorial, we will place the fine-tuning dataset in the root directory [/dataset](../dataset/huanhuan.json).

## Model download

Use the snapshot_download function in modelscope to download the model. The first parameter is the model name, and the parameter cache_dir is the download path of the model.

Create a new model_download.py file in the /root/autodl-tmp path and enter the following content in it. Remember to save the file after pasting the code, as shown in the figure below. And run `python /root/autodl-tmp/model_download.py` to execute the download. The model size is 15 GB and it takes about 2 minutes to download the model.

```python
import torch
from modelscope import snapshot_download, AutoModel,AutoTokenizer
import os

model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
```

## Instruction set construction

LLM fine-tuning generally refers to the instruction fine-tuning process. The so-called instruction fine-tuning means that the fine-tuning data we use is in the form of:

```json
{
"instruction": "Answer the following user question and only output the answer.",
"input": "What is 1+1?",
"output": "2"
}
```

Among them, `instruction` is the user instruction, telling the model what task it needs to complete; `input` is the user input, which is the input content required to complete the user instruction; `output` is the output that the model should give.

That is, our core training goal is to enable the model to understand and follow user instructions. Therefore, when constructing the instruction set, we should build the task instruction set specifically for our target task. For example, in this section we use the open source [Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) project as an example. Our goal is to build a personalized LLM that can simulate the conversation style of Zhen Huan. Therefore, the instructions we construct are as follows:

```json
{
"instruction": "Who are you?",
"input": "",
"output": "My father is Zhen Yuandao, the Shaoqing of Dali Temple."

}
```

All the instruction data sets we constructed are in the root directory.

## Data formatting

The data trained by `Lora` needs to be formatted and encoded before being input to the model for training. Students who are familiar with the `Pytorch` model training process will know that we generally need to encode the input text as input_ids and the output text as `labels`. The results after encoding are all multi-dimensional vectors. We first define a preprocessing function, which is used to encode the input and output text of each sample and return an encoded dictionary:

```python
def process_func(example):
MAX_LENGTH = 384 # Llama tokenizer will split a Chinese character into multiple tokens, so it is necessary to relax some maximum lengths to ensure data integrity
input_ids, attention_mask, labels = [], [], [] instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id| ><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False) # add_special_tokens Do not add special_tokens at the beginning response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens =False) input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id] attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] # Because we also need to pay attention to eos token, we add 1
labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
if len(input_ids) > MAX_LENGTH: # Do a truncation
input_ids = input_ids[:MAX_LENGTH]
attention_mask = attention_mask[:MAX_LENGTH]
labels = labels[:MAX_LENGTH]
return {
"input_ids": input_ids,
"attention_mask": attention_mask,
"labels": labels
}
```

`Llama-3-8B-Instruct` uses `Prompt Template`The format is as follows:

```text
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant<|eot_id|>'
<|start_header_id|>user<|end_header_id|>
Who are you? <|eot_id|>'
<|start_header_id|>assistant<|end_header_id|>
I am a helpful assistant. <|eot_id|>"
```

## Load tokenizer and half-precision model

The model is loaded in half-precision. If your graphics card is relatively new, you can use `torch.bfolat` to load it. For custom models, be sure to specify the `trust_remote_code` parameter as `True`.

```python
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', use_fast=False, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', device_map="auto",torch_dtype=torch.bfloat16)
```

## Define LoraConfig

Many parameters can be set in the `LoraConfig` class, but there are not many main parameters. I will briefly explain them. Students who are interested can directly look at the source code.

- `task_type`: model type
- `target_modules`: the name of the model layer to be trained, mainly the layer of the `attention` part. Different models have different names for the corresponding layers. You can pass in an array, a string, or a regular expression.
- `r`: the rank of `lora`, see `Lora` principle for details
- `lora_alpha`: `Lora alaph`, see `Lora` principle for specific functions

What is the scaling of `Lora`? Of course it is not `r` (rank), the scaling is `lora_alpha/r`, and the scaling in this `LoraConfig` is 4 times.

```python
config =LoraConfig(
task_type=TaskType.CAUSAL_LM,
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
inference_mode=False, # Training mode
r=8, # Lora rank
lora_alpha=32, # Lora alaph, see Lora principle for specific functions
lora_dropout=0.1# Dropout ratio
)
```

## Customize TrainingArguments parameters

The source code of the `TrainingArguments` class also introduces the specific functions of each parameter. Of course, you can explore it yourself. Here are a few commonly used ones.

- `output_dir`: output path of the model
- `per_device_train_batch_size`: As the name suggests `batch_size`
- `gradient_accumulation_steps`: gradient accumulation, if your video memory is small, you canTo set `batch_size` a little smaller, the gradient accumulation increases.
- `logging_steps`: how many steps, output once `log`
- `num_train_epochs`: as the name suggests `epoch`
- `gradient_checkpointing`: gradient check, once this is turned on, the model must execute `model.enable_input_require_grads()`, this principle can be explored by yourself, I won't go into details here.

```python
args = TrainingArguments(
output_dir="./output/llama3",
per_device_train_batch_size=4,
gradient_accumulation_steps=4,
logging_steps=10,
num_train_epochs=3,
save_steps=100,
learning_rate=1e-4,
save_on_each_node=True,
gradient_checkpointing=True
)
```## Training with Trainer

```python
trainer = Trainer(
model=model,
args=args,
train_dataset=tokenized_id,
data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()

```

## Save lora weights

```python
lora_path='./llama3_lora'
trainer.model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
```

## Load lora weights for inference

After training, you can use the following method to load `lora` weights for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct'
lora_path = './llama3_lora' # lora weight path

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# Load model
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16)

# Load lora weights
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

prompt = "Who are you?"
messages = [
# {"role": "system", "content": "Now you have to play the role of the emperor's woman--Zhen Huan"},
{"role": "user", "content":prompt} ] text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) model_inputs = tokenizer([text], return_tensors="pt").to('cuda') generated_ids = model.generate( model_inputs.input_ids, max_new_tokens=512, do_sample=True, top_p=0.9, temperature=0.5, repetition_penalty=1.1, eos_token_id=tokenizer.encode('<|eot_id|>')[0], ) generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids,generated_ids) ] response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] print(response) ````