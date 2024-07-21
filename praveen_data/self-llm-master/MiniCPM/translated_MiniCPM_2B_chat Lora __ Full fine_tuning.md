# MiniCPM-2B-chat transformers deployment call

## MiniCPM-2B-chat introduction

MiniCPM is a series of large end-side models jointly open-sourced by Mianbi Intelligence and the Natural Language Processing Laboratory of Tsinghua University. The main language model MiniCPM-2B has only 2.4 billion (2.4B) non-word embedding parameters.

After SFT, MiniCPM is similar to Mistral-7B (with better Chinese, mathematics, and coding capabilities) on the public comprehensive evaluation set, and its overall performance exceeds Llama2-13B, MPT-30B, Falcon-40B and other models.
After DPO, MiniCPM-2B also surpassed many representative open source large models such as Llama2-70B-Chat, Vicuna-33B, Mistral-7B-Instruct-v0.1, and Zephyr-7B-alpha on the current evaluation set MTBench, which is closest to user experience.
Based on MiniCPM-2B, the end-side multimodal large model MiniCPM-V is built. The overall performance is the best among the models of the same scale, surpassing the existing multimodal large models built based on Phi-2, and achieving performance comparable to or even better than the 9.6B Qwen-VL-Chat on some evaluation sets..
After Int4 quantization, MiniCPM can be deployed on mobile phones for inference, and the streaming output speed is slightly higher than the speed of human speech. MiniCPM-V also directly runs through the deployment of multimodal large models on mobile phones.
A 1080/2080 can be used for efficient parameter fine-tuning, and a 3090/4090 can be used for full parameter fine-tuning. One machine can continuously train MiniCPM, and the secondary development cost is low.

## Environment preparation

Rent a **single card 3090 or other 24G** graphics card machine in the autodl platform. As shown in the figure below, select PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1
Next, open the JupyterLab of the server just rented, image and open the terminal in it to start environment configuration, model download and run demonstration. 

> **Note: Here you need to select a machine with an `cpu` of `intel`. An `cpu` of `amd` may cause `deepspeed zero2 offload` to fail to load. **

![Alt ​​text](images/image-1.png)

Next, open the `JupyterLab` of the server you just rented, and open the terminal in it to start environment configuration, model download and run `demo`.

pip change source and install dependent packages

```shell
# Upgrade pip
python -m pip install --upgrade pip
# Change the installation of the pypi source acceleration library
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope transformers sentencepiece accelerate langchain

MAX_JOBS=8 pip install flash-attn --no-build-isolation

pip install peft deepspeed
```

> Note: flash-attn installation will be slow, about ten minutes.

## Model download

Use the `snapshot_download` function in `modelscope` to download the model. The first parameter is the model name, and the parameter `cache_dir` is the download path of the model.

Create a new `download.py` file in the `/root/autodl-tmp` path and enter the following content in it. Remember to save the file after pasting the code, as shown in the figure below. And run `python /root/autodl-tmp/download.py` to download. The model size is 10 GB. It takes about 5 to 10 minutes to download the model.

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('OpenBMB/MiniCPM-2B-sft-fp32', cache_dir='/root/autodl-tmp', revision='master')
```

## Dataset construction

LLM fine-tuning generally refers to the instruction fine-tuning process. The so-called instruction fine-tuning means that the fine-tuning data we use is in the form of:

```json
{
"instrution":"Answer the following user question and only output the answer.",
"input":"What is 1+1?",
"output":"2"
}
```

Among them, `instruction` is the user instruction, telling the model the task it needs to complete; `input` is the user input, which is the input content required to complete the user instruction; `output` is the user input, which is the input content required to complete the user instruction;` is the output that the model should give.

That is, our core training goal is to enable the model to understand and follow user instructions. Therefore, when constructing the instruction set, we should build a task instruction set specifically for our target task. For example, in this section, we use the [Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) project co-opened by the author as an example. Our goal is to build a personalized LLM that can simulate the conversation style of 甄嬛, so the instructions we construct are as follows:

```json
{
"instruction": "Now you have to play the role of the woman next to the emperor--甄嬛",
"input":"Who are you?",
"output":"My father is Zhen Yuandao, the Shaoqing of the Dali Temple."
}
```

All the instruction datasets we constructed are under the root directory `dataset`.

## Data formatting

The data trained by `Lora` needs to be formatted and encoded before being input to the model for training. Students who are familiar with the `Pytorch` model training process will know that we generally need to encode the input text as input_ids and the output text as `labels`. The results after encoding are all multi-dimensional vectors. We first define a preprocessing function, which is used to encode each sample.It inputs and outputs text and returns an encoded dictionary:

```python
def process_func(example):
MAX_LENGTH = 512 # Llama tokenizer will split a Chinese character into multiple tokens, so it is necessary to relax some maximum lengths to ensure data integrity
input_ids, attention_mask, labels = [], [], []
instruction = tokenizer(f"<User>{example['instruction']+example['input']}<AI>", add_special_tokens=False) # add_special_tokens does not add special_tokens at the beginning
response = tokenizer(f"{example['output']}", add_special_tokens=False)
input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] # Because we also need to pay attention to eos token, we add 1
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

`MiniCPM` uses the `Prompt_Template` of `transformers` by default. Interested students can view the source code structure in the `transformers` repository~

## Load model

The model is loaded in half-precision. If your graphics card is relatively new, you can load it in `torch.bfolat`. For custom models, be sure to specify the `trust_remote_code` parameter as `True`.

```python
tokenizer = AutoTokenizer.from_pretrained('./OpenBMB/miniCPM-bf32', use_fast=False, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained('./OpenBMB/miniCPM-bf32', trust_remote_code=True, torch_dtype=torch.half, device_map="auto")
```
## Define LoraConfig

Many parameters can be set in the `LoraConfig` class, but there are not many main parameters. I will briefly explain them. Students who are interested can directly read the source code.

- `task_type`: model type
- `target_modules`: the name of the model layer to be trained, mainly the layer of the `attention` part. Different models have different corresponding layer names. You can pass in arrays, strings, or regular expressions.
- `r`: the rank of `lora`, for details, please refer to `Lora` principle
- `lora_alpha`: `Lora alaph`, for specific functions, please refer to `Lora` principle

```python
config = LoraConfig(
task_type=TaskType.CAUSAL_LM, 
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
inference_mode=False, # training mode
r=8, # Lora rank
lora_alpha=32, # Lora alaph, for specific functions, see Lora Principle
lora_dropout=0.1# Dropout ratio
)
```

> Note: If you want to fine-tune the whole amount, you can choose not to load `LoraConfig` when loading the model.

## Custom TrainingArguments parameters

The source code of the `TrainingArguments` class also introduces the specific functions of each parameter. Of course, you can explore it yourself. Here are a few common ones.

- `output_dir`: output path of the model
- `per_device_train_batch_size`: As the name suggests, `batch_size`
- `gradient_accumulation_steps`: gradient accumulation. If your video memory is small, you can set `batch_size` to a smaller value to increase the gradient accumulation.
- `logging_steps`: how many steps to output a `log`
- `num_train_epochs`: as the name suggests, `epoch`

And so on, there are many parameters that can be adjusted. Students who are interested can go and look at the source code of `transformers`.Students are also welcome to submit PR for this project~

## Training with deepspeed

`deepspeed` is a distributed training tool that can easily perform distributed training. Here we use `deepspeed` for training.

- If you want to perform `lora` training, please uncomment the 67-line code in `train.py` and run the `train.sh` script.

> Lora training requires about 14G video memory. Because gradient checking is not enabled, the video memory usage will be a little higher.

```python
# Create a model and load it in half-precision
model = AutoModelForCausalLM.from_pretrained(finetune_args.model_path, trust_remote_code=True, torch_dtype=torch.half, device_map={"": int(os.environ.get("LOCAL_RANK") or 0)})

model = get_peft_model(model, config)
```

- For full training, just run the `train.sh` script.

> For full training, about 22G of video memory is required.Because the CPU offload function of deepspeed zero2 is enabled, the optimizer parameters will be loaded onto the CPU for calculation, thus reducing the video memory.