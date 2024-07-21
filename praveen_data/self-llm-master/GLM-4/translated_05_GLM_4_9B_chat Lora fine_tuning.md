# GLM4-9B-chat Lora fine-tuning.

In this section, we briefly introduce how to fine-tune the LLaMA3-8B-Instruct model with Lora based on frameworks such as transformers and peft. Lora is an efficient fine-tuning method. For a deeper understanding of its principles, please refer to the blog: [Zhihu | Lora in a simple way](https://zhuanlan.zhihu.com/p/650197598).

This tutorial will provide you with a [nodebook](./GLM4-9B-chat%Lora%fine-tuning..ipynb) file in the same directory to help you learn better.

## Environment preparation

Rent a graphics card machine with 24G video memory such as 3090 in the Autodl platform. As shown in the figure below, select `PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1` in the mirror.
Next, open JupyterLab on the server you just rented, and open the terminal to start environment configuration, model download, and run the demo.

![Open machine configuration selection](images/image-1.png)

## Environment configuration

After completing the basic environment configuration and local model deployment, you also need to install some third-party libraries. You can use the following command:

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
pip install tiktoken==0.7.0

MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

> Note: flash-attn installation will be slow, about ten minutes.

> Considering that some students may encounter some problems in configuring the environment, we will install it in AutoDThe L platform has prepared an environment image of GLM-4, which is suitable for the deployment environment that requires GLM-4 in this tutorial. Click the link below and create an AutoDL example directly. (vLLM has higher requirements for the torch version, and the higher the version, the more complete the model support and the better the effect, so create a new image.) **https://www.codewithgpu.com/i/datawhalechina/self-llm/GLM-4**

In this tutorial, we will place the fine-tuning dataset in the root directory [/dataset](../dataset/huanhuan.json).

## Model download

Use the snapshot_download function in modelscope to download the model. The first parameter is the model name, and the parameter cache_dir is the download path of the model.

Create a new model_download.py file in the /root/autodl-tmp path and enter the following content in it. Remember to save the file after pasting the code, as shown in the figure below. And run `python /root/autodl-tmp/model_download.py` to execute the download.

```python import torch from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('ZhipuAI/glm-4-9b-chat', cache_dir='/root/autodl-tmp/glm-4-9b-chat', revision='master')
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

That is, our core training goal is to enable the model to understand and follow user instructions. Therefore, when constructing the instruction set, we should build the task instruction set specifically for our target task. For example, in this section we use the open source code that I co-developed.[Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) project is used as an example. Our goal is to build a personalized LLM that can simulate the conversation style of 甄嬛. Therefore, the instructions we construct are as follows:

```json
{
"instruction": "Who are you?",
"input": "",
"output": "My father is Zhen Yuandao, Shaoqing of Dali Temple."

}
```

All the instruction data sets we constructed are in the root directory.

## Data formatting

The data trained by `Lora` needs to be formatted and encoded before being input to the model for training. Students who are familiar with the `Pytorch` model training process will know that we generally need to encode the input text as input_ids and the output text as `labels`. The results after encoding are all multi-dimensional vectors. We first define a preprocessing function, which is used to encode the input and output text of each sample and return an encoded dictionary:

```python
def process_func(example):
MAX_LENGTH = 384
input_ids, attention_mask, labels = [], [], []
instruction = tokenizer((f"[gMASK]<sop><|system|>\nAssume you are the woman next to the emperor--Zhen Huan.<|user|>\n"
f"{example['instruction']+example['input']}<|assistant|>\n"
), 
add_special_tokens=False)
response = tokenizer(f"{example['output']}", add_special_tokens=False)
input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
attention_mask = instruction["attention_mask"] + response["attention_maskask"] + [1] # Because we also need to pay attention to eos token, we add 1
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

`GLM4-9B-chat` uses the following `Prompt Template` format:

```text
[gMASK]<sop><|system|> 
Suppose you are the woman next to the emperor--Zhen Huan. <|user|> 
Miss, other ladies are seeking to be selected, only our lady wants to be left behind, the Bodhisattva must remember it for real--<|assistant|> 
Shh--it is said that making a wish out loud is not effective. <|endoftext|>
```

## Load tokenizer and half-precision model

The model is loaded in half-precision format. If your graphics card is relatively new, you can use `torch.bfolat` format to load it. For custom models, you must specify the `trust_remote_code` parameter as `True`.

```python tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/glm-4-9b-chat/ZhipuAI/glm-4-9b-chat', use_fast=False, trust_remote_code=True) model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/glm-4-9b-chat/ZhipuAI/glm-4- 9b-chat', device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True)
```

## Define LoraConfig

Many parameters can be set in the `LoraConfig` class, but there are not many main parameters. I will briefly explain them. Students who are interested can directly look at the source code.

- `task_type`: model type
- `target_modules`: the name of the model layer to be trained, mainly the layer of the `attention` part. Different models have different names for the corresponding layers. You can pass in arrays, strings, or regular expressions.
- `r`: the rank of `lora`, see `Lora` principle for details
- `lora_alpha`: `Lora alaph`, see `Lora` principle for specific functions

What is the scaling of `Lora`? Of course it is not `r` (rank), this scaling is `lora_alpha/r`, and the scaling in this `LoraConfig` is 4 times.

```python config = LoraConfig( task_type=TaskType.CAUSAL_LM, target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], # For existing problems, only some demonstrations can be fine-tuned
inference_mode=False, # Training mode
r=8, # Lora rank
lora_alpha=32, # Lora alaph, see Lora principle for specific functions
lora_dropout=0.1# Dropout ratio
)
```

## Custom TrainingArguments parameters

The source code of the `TrainingArguments` class also introduces the specific functions of each parameter. Of course, you can explore it yourself. Here are a few common ones.

- `output_dir`: output path of the model
- `per_device_train_batch_size`: As the name suggests, `batch_size`
- `gradient_accumulation_steps`: gradient accumulation. If your video memory is small, you can set `batch_size` to a smaller value to increase the gradient accumulation.
- `logging_steps`: how many steps to output `log` once
- `num_train_epochs`: As the name implies, `epoch`
- `gradient_checkpointing`: gradient check. Once this is turned on, the model must execute `model.enable_input_require_grads()`. You can explore this principle by yourself, so I won’t go into details here.

```python
args = TrainingArguments(
output_dir="./output/GLM4",
per_device_train_batch_size=1,
gradient_accumulation_steps=8,
logging_steps=50,
num_train_epochs=2,
save_steps=100,
learning_rate=1e-5,
save_on_each_node=True,
gradient_checkpointing=True
)
```

## Training with Trainer

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
lora_path='./GLM4'
trainer.model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
```

## Load lora weights for inference

After training, you can use the following method to load `lora` weights for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = '/root/autodl-tmp/glm-4-9b-chat/ZhipuAI/glm-4-9b-chat'
lora_path ='./GLM4_lora'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# Load lora weights
model = PeftModel.from_pretrained(model, model_id=lora_path)

prompt = "Who are you?"
inputs = tokenizer.apply_chat_template([{"role": "user", "content": "Assume you are the woman next to the emperor--Zhen Huan."},{"role": "user", "content": prompt}],add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True ).to('cuda') gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1} with torch.no_grad (): outputs = model.generate(**inputs, **gen_kwargs) outputs = outputs[:, inputs['input_ids'].shape[1]:] print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```