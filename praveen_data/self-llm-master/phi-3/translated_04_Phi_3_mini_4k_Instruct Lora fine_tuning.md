# Phi-3-mini-4k-Instruct Lora fine-tuning

In this section, we briefly introduce how to fine-tune the Phi-3-mini-4k-Instruct model with Lora based on frameworks such as transformers and peft. Lora is an efficient fine-tuning method. For a deeper understanding of its principles, please refer to the blog: [Zhihu | Lora in a simple way](https://zhuanlan.zhihu.com/p/650197598).

This tutorial will provide you with a [nodebook](./Phi-3-mini-4k-Instruct-Lora.ipynb) file in the same directory to help you learn better.

## Environment preparation

Rent a graphics card machine with 24G video memory such as 3090 in the Autodl platform. As shown in the figure below, select `PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1` in the image.
Next, open JupyterLab on the server you just rented, and open the terminal to start environment configuration, model download, and run the demo.

![Open machine configuration selection](assets/03-1.png)

## Environment configuration

After completing the basic environment configuration and local model deployment, you also need to install some third-party libraries.You can use the following command:

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
> Considering that some students may encounter some problems in configuring the environment, we have prepared an environment image of Phi-3 on the AutoDL platform, which is applicable to all deployment environments of this warehouse. Click the link below and create it directlyJust create an Autodl example.
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/Phi-3-Lora***

In this tutorial, we will place the fine-tuning dataset in the root directory [/dataset](../dataset/huanhuan.json).

## Model download

Use the snapshot_download function in modelscope to download the model. The first parameter is the model name, and the parameter cache_dir is the download path of the model.

Create a new model_download.py file in the /root/autodl-tmp path and enter the following content in it. Remember to save the file after pasting the code, as shown in the figure below. And run `python /root/autodl-tmp/model_download.py` to execute the download

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('LLM-Research/Phi-3-mini-4k-instruct', cache_dir='/root/autodl-tmp', revision='master')
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

Among them, `instruction` is the user instruction, telling the model the task it needs to complete; `input` is the user input, which is the input content required to complete the user instruction; `output` is the output that the model should give.

That is, our core training goal is to enable the model to understand and follow user instructions. Therefore, when constructing the instruction set, we should build a task instruction set specifically for our target task. For example, in this section we use the open source [Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) project co-developed by the author as an example. Our goal is to build a personalized LLM that can simulate the conversation style of 甄嬛. Therefore, the instructions we construct are as follows:：

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
MAX_LENGTH = 384 # Llama tokenizer will split a Chinese character into multiple tokens, so it is necessary to relax some maximum lengths to ensure data integrity
input_ids, attention_mask, labels = [], [], []
instruction = tokenizer(f"<|user|>\n{example['instruction'] + example['input']}<|end|>\n<|assistant|>\n", add_special_tokens=False) # add_special_tokens does not add special_tokens at the beginning
response = tokenizer(f"{example['output']}<|end|>\n", add_special_tokens=False)
input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] # Because we also need to pay attention to eos tokens, we add 1
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

`Phi-3-mini-4k-Instruct` uses the following `Prompt Template` format:

```text
<|system|>
You are a helpful assistant<|end|>
<|user|>
Who are you? <|end|>
<|assistant|>
I am a helpful assistant. <|end|>
```

## LoadTokenizer and half-precision model

The model is loaded in half-precision. If your graphics card is relatively new, you can use `torch.bfolat` to load it. For custom models, be sure to specify the `trust_remote_code` parameter as `True`.

```python
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/LLM-Research/Phi-3-mini-4k-instruct', use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_toke
model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/LLM-Research/Phi-3-mini-4k-instruct', device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True)
```

## Define LoraConfig

`LoraConfiMany parameters can be set in the g` class, but there are not many main parameters. I will briefly explain them. Students who are interested can directly look at the source code.

- `task_type`: model type
- `target_modules`: the name of the model layer to be trained, mainly the layer of the `attention` part. Different models have different corresponding layer names. You can pass in arrays, strings, or regular expressions.
- `r`: the rank of `lora`, see `Lora` principle for details
- `lora_alpha`: `Lora alaph`, see `Lora` principle for specific functions

What is the scaling of `Lora`? Of course it is not `r` (rank), this scaling is `lora_alpha/r`, and the scaling in this `LoraConfig` is 4 times.

```python config = LoraConfig( task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], inference_mode=False, # Training moder=8, # Lora rank
lora_alpha=32, # Lora alaph, for specific functions, see Lora principle
lora_dropout=0.1# Dropout ratio
)
```

## Custom TrainingArguments parameters

The source code of the class `TrainingArguments` also introduces the specific functions of each parameter. Of course, you can explore it yourself. Here are a few commonly used ones.

- `output_dir`: output path of the model
- `per_device_train_batch_size`: As the name suggests, `batch_size`
- `gradient_accumulation_steps`: gradient accumulation. If your video memory is small, you can set `batch_size` to a smaller value to increase the gradient accumulation.
- `logging_steps`: how many steps to output a `log`
- `num_train_epochs`: as the name suggests, `epoch`
- `gradient_checkpointing`: gradient checking, once this is turned on, the model must execute `model.enable_input_require_grads()`, you can explore this principle by yourself, so I won’t go into details here.

```python
args = TrainingArguments(
output_dir="./output/Phi-3",
per_device_train_batch_size=4,
gradient_accumulation_steps=4,
logging_steps=10,
num_train_epochs=3,
save_steps=100,
learning_rate=1e-4,
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
lora_path='./Phi-3_lora'
trainer.model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
```

## Load lora weights for inference

After training, you can use the following method to load `lora` weights for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_path = '/root/autodl-tmp/LLM-Research/Phi-3-mini-4k-instruct'
lora_path = './Phi-3_lora' # lora weight path

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left') # Load model model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16) # Load lora weight model = PeftModel.from_pretrained(model, model_id=lora_path, config=config) prompt = "Who are you?" messages = [ {"role": "user", "content": prompt} ] text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) model_inputs = tokenizer([text] , return_tensors="pt").to('cuda') generated_ids = model.generate( model_inputs.input_ids, max_new_tokens=512, eos_token_id=tokenizer.encode('<|endoftext|>')[0] ) outputs = generated_ids.tolist()[0][len(model_inputs[0]):] response = tokenizer. decode(outputs).split('<|end|>')[0] print(response) ````