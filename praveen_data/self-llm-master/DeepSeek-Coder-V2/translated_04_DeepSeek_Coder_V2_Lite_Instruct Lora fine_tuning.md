# DeepSeek-Coder-V2-Lite-Instruct Lora fine-tuning

In this section, we briefly introduce how to fine-tune the DeepSeek-Coder-V2-Lite-Instruct model with Lora based on frameworks such as transformers and peft. Lora is an efficient fine-tuning method. For a deeper understanding of its principles, please refer to the blog: [Zhihu | Lora in a simple way](https://zhuanlan.zhihu.com/p/650197598).

This tutorial will provide you with a [nodebook](./04-DeepSeek-Coder-V2-Lite-Instruct%20Lora%20fine-tuning.ipynb) file in the same directory to help you learn better.

> **Note**: Fine-tuning the DeepSeek-Coder-V2-Lite-Instruct model requires a 4×3090 graphics card.

## Model download 

Use the snapshot_download function in modelscope to download the model. The first parameter is the model name, and the parameter cache_dir is the model download path.

Create a new model_download.p file in the /root/autodl-tmp path.y file and enter the following content in it. Please save the file in time after pasting the code, as shown in the figure below. And run `python /root/autodl-tmp/model_download.py` to download. The model size is 15GB and it takes about 5 minutes to download the model.

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', cache_dir='/root/autodl-tmp', revision='master')
``` 

## Environment configuration

After completing the basic environment configuration and local model deployment, you also need to install some third-party libraries. You can use the following command:

```bash
python -m pip install --upgrade pip
# Replace the pypi source to accelerate the installation of the library
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.9.5
pip install "transformers>=4.41.0"
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.27
pip install transformers_stream_generator==0.0.4
pip install datasets==2.18.0
pip install peft==0.10.0

# Optional
MAX_JOBS=8 pip install flash-attn --no-build-isolation 
```
> Considering that some students may encounter some problems in configuring the environment, we have prepared the DeepSeek-Coder-V2-Lite-Instruct environment image on the AutoDL platform. Click the link below and directly create the Autodl example.
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/Deepseek-coder-v2***

> Note: flash-attn installation will be slow, it will take about ten minutes.

In this tutorial, we will place the fine-tuning dataset in the root directory [/dataset](../dataset/huanhuan.json).

## Instruction set construction

LLM fine-tuning generally refers to the instruction fine-tuning process. The so-called instruction fine-tuning means that the fine-tuning data we use is in the form of:

```json
{
"instruction":"Answer the following user question and only output the answer.",
"input":"What is 1+1?",
"output":"2"
}
```

Among them, `instruction` is the user instruction, telling the model what task it needs to complete; `input` is the user input, which is the input content required to complete the user instruction; `output` is the output that the model should give.

That is, our core training goal is to enable the model to understand and follow user instructions. Therefore, when constructing the instruction set, we should build the task instruction set specifically for our target task. For example, in this section we use the open source [Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) project as an example. Our goal is to build a personalized LLM that can simulate the conversation style of Zhen Huan. Therefore, the instructions we construct are as follows:

```json
{
"instruction": "Who are you?",
"input":"",
"output":"My father is Zhen Yuandao, the Shaoqing of Dali Temple."

}
```

All the instruction data sets we constructed are in the root directory.

## Data formatting

The data trained by `Lora` needs to be formatted and encoded before being input to the model for training. Students who are familiar with the `Pytorch` model training process will know that we generally need to encode the input text as input_ids and the output text as `labels`. The results after encoding are all multi-dimensional vectors. We first define a preprocessing function, which is used to encode the input and output text of each sample and return an encoded dictionary:

```python
def process_func(example):
MAX_LENGTH = 384
input_ids, attention_mask, labels = [], [], []
instruction = tokenizer((f"<｜begin of sentence｜>Suppose you are the woman beside the emperor--Zhen Huan. \n"
f"User: {example['instruction']+example['input']}\nAssistant: "
).strip(), 
add_special_tokens=False)
response = tokenizer(f"{example['output']}<｜end of sentence｜>", add_special_tokens=False)
input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] # Because eos token, we also need to pay attention to it, so add 1
labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id] 
if len(input_ids) > MAX_LENGTH: # Make a truncation
input_ids = input_ids[:MAX_LENGTH]
attention_mask = attention_mask[:MAX_LENGTH]
labels = labels[:MAX_LENGTH]
return {
"input_ids": input_ids,
"attention_mask": attention_mask,
"labels": labels
}
```

`DeepSeek-Coder-V2-Lite-Instruct` uses the following `Prompt Template` format:

```text
<｜begin of sentence｜>{system_message}

User: {user_message_1}

Assistant: {assistant_message_1}<｜end of sentence｜>User: {user_message_2}

Assistant:
```

## Load tokenizer and half-precision model

The DeepSeek-Coder-V2-Lite-Instruct model needs to be loaded in 32-bit precision. For custom models, the `trust_remote_code` parameter must be specified as `True`.

```python
model_path = '/root/autodl-tmp/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32, device_map="auto")
```
## Define LoraConfig

Many parameters can be set in the `LoraConfig` class, but there are not many main parameters. I will briefly explain them. Students who are interested can directly look at the source code.

- `task_type`: model type
- `target_modules`: the name of the model layer to be trained, mainly the layer of the `attention` part. Different models have different names for the corresponding layers. You can pass in an array, a string, or a regular expression.
- `r`: the rank of `lora`, see `Lora` principle for details
- `lora_alpha`: `Lora alaph`, see `Lora` principle for specific functions

What is the scaling of `Lora`? Of course it is not `r` (rank), this scaling is `lora_alpha/r`, and the scaling in this `LoraConfig` is 4 times.

```python config = LoraConfig( task_type=TaskType.CAUSAL_LM, 
target_modules=["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", 'gate_proj', 'up_proj', 'down_proj'], # For existing problems, only some demonstrations need to be fine-tuned
inference_mode=False, # Training mode
r=8, # Lora rank
lora_alpha=32, # Lora alaph, for specific functions, see Lora principle
lora_dropout=0.1# Dropout ratio
)
```

## Customize TrainingArguments parameters

The source code of the `TrainingArguments` class also introduces the specific functions of each parameter. Of course, you can explore it yourself. Here are a few commonly used ones.

- `output_dir`: output path of the model
- `per_device_train_batch_size`: As the name suggests `batch_size`
- `gradient_accumulation_steps`: gradient accumulation, if your video memory is small, you can use `batch_size` is set smaller, and the gradient accumulation increases.
- `logging_steps`: how many steps, output once `log`
- `num_train_epochs`: as the name suggests `epoch`
- `gradient_checkpointing`: gradient check, once this is turned on, the model must execute `model.enable_input_require_grads()`. You can explore this principle by yourself, so I won’t go into details here.

```python
args = TrainingArguments(
output_dir="./output/deepseek_coder_v2",
per_device_train_batch_size=1,
gradient_accumulation_steps=8,
logging_steps=10,
num_train_epochs=2,
save_steps=100,
learning_rate=1e-5,
save_on_each_node=True,
gradient_checkpointing=True,
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

> Due to the large number of model parameters, the time required to train the model will also increase. It takes about 10 hours to complete the tutorial code training. If you just use it for learning, you can stop when you see the loss decrease.

## Load lora weight inference

After training, you can use the following method to load the `lora` weight for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_path = '/root/autodl-tmp/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'
lora_path = './output/deepseek_coder_v2'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# Load lora weights
model = PeftModel.from_pretrained(model, model_id=lora_path)

messages=[
{'role': 'sysrem', 'content': "Suppose you are the woman beside the emperor--Zhen Huan."},
{ 'role': 'user', 'content': "Hello"}
]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device) outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id =tokenizer.eos_token_id) print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)) ```