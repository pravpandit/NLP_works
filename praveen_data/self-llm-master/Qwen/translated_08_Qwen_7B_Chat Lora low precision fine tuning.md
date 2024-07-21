# 08-Qwen-7B-Chat Lora low-precision fine-tuning

## Overview

In this section, we briefly introduce how to perform low-precision fine-tuning of Lora on the Qwen-7B-Chat model based on frameworks such as transformers and peft. Including 8bit and 4bit QLoRA

The code script described in this section is in the same directory [08-Qwen-7B-Chat Lora low-precision fine-tuning](./04-Qwen-7B-Chat%20Lora%20 low-precision fine-tuning.py). Run the script to perform the fine-tuning process, but note that the model path and dataset path in the script file need to be modified.

This tutorial will provide you with two [nodebook] files in the same directory, corresponding to [8bit](./08-Qwen-7B-Chat%20Lora%20-8bitfine-tuning.ipynb) and [4bit](./08-Qwen-7B-Chat%20Lora%20-4bitfine-tuning.ipynb) files, so that you can learn better.

## Environment configuration

After completing the basic environment configuration and local model deployment, you also need to install some third-party libraries. You can use the following command:

```bash
pip install transformers==4.35.2
pip installl peft==0.4.0
pip install datasets==2.10.1
pip install accelerate==0.20.3
pip install tiktoken
pip install transformers_stream_generator
pip install bitsandbytes==0.41.1
```

In this tutorial, we will place the fine-tuning dataset in the root directory [/dataset](../dataset/huanhuan.jsonl).

## Instruction set construction

LLM fine-tuning generally refers to the instruction fine-tuning process. The so-called instruction fine-tuning means that the fine-tuning data we use is in the form of:

```json
{
"instrution":"Answer the following user question and only output the answer.",
"input":"What is 1+1?",
"output":"2"
}
```

Among them, `instruction` is the user instruction, telling the model the task it needs to complete; `input` is the user input, which is the input content required to complete the user instruction; `output` is the output that the model should give.

That is, our core training goal is to enable the model to have the ability to understand and follow user instructions. Therefore,When constructing the instruction set, we should build the task instruction set specifically for our target task. For example, in this section we use the [Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) project co-opened by the author as an example. Our goal is to build a personalized LLM that can simulate the conversation style of 甄嬛, so the instructions we construct are as follows:##

```json
{
"instruction": "Now you have to play the role of the woman beside the emperor--甄嬛",
"input":"Who are you?",
"output":"My father is 敦义寺少卿贞远道."
}
```
All the instruction data sets we constructed are in the root directory.

## Data formatting

The data trained by `Lora` needs to be formatted and encoded before being input to the model for training. Students who are familiar with the `Pytorch` model training process will know that we generally need to encode the input text as input_ids and the output text as `labels`. The results after encoding are all multi-dimensional vectors. We first define a preprocessing function, which is used to encode the input and output text of each sample and return an encoded dictionary:

```python
def process_func(example):
MAX_LENGTH = 384 # Llama tokenizer will split a Chinese character into multiple tokens, so it is necessary to relax some maximum lengths to ensure data integrity
input_ids, attention_mask, labels = [], [], []
instruction = tokenizer("\n".join(["<|im_start|>system", "Now you have to play the role of the woman beside the emperor--Zhen Huan.<|im_end|>" + "\n<|im_start|>user\n" + example["instruction"] + example["input"] + "<|im_end|>\n"]).strip(), add_special_tokens=False) # add_special_tokens does not add special_tokens at the beginning
response = tokenizer("<|im_start|>assistant\n" + example["output"] + "<|im_end|>\n", add_special_tokens=False)input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] # Because we also need to pay attention to eos token, we add 1
labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id] # Qwen's special construction is like this
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

The formatted data, that is, each piece of data sent to the model, is a dictionary containing three key-value pairs: `input_ids`, `attention_mask`, and `labels`. Among them, `input_ids` is the encoding of the input text, `attention_mask` is the attention mask of the input text, and `labels` is the encoding of the output text. After decoding, it should be like this:

```text
<|im_start|>system
Now you have to play the role of the woman next to the emperor--Zhen Huan.<|im_end|>
<|im_start|>user
Miss, other girls are seeking to be selected, only our lady wants to be left behind, the Bodhisattva must remember it for real - <|im_end|>
<|im_start|>assistant
Shh - it is said that making a wish is not effective if it is revealed. <|im_end|>
<|endoftext|>
```

Why is thisWhat about the form? Good question! Different models have different formatted inputs, so we need to check the training source code of our deep model. Because the Lora fine-tuning effect should be the best in the form of fine-tuning according to the original model instructions, we still follow the input format of the original model. OK, here is the link to the source code for you. If you are interested, you can explore it yourself: [hugging face Qwen warehouse] (https://hf-mirror.com/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py): the `make_context` function.

## Load tokenizer and half-precision model

The model is loaded in half-precision form. If your graphics card is relatively new, you can use `torch.bfolat` to load it. For custom models, you must specify the `trust_remote_code` parameter as `True`.
- For 8bit, set `load_in_8bit=True`.
- For 4bit, multiple parameters need to be set. For detailed explanation, please refer to the paper or this explanation video [4bit_QLoRA](https://www.bilibili.com/video/BV1DQ4y1t7e8/?spm_id_from=333.788&vd_source=fae77b03d7da1995e1dff042fe6aa49a)

```python
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/qwen/Qwen-7B-Chat', use_fast=False, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eod_id # In Qwen, eod_id and pad_token_id are the same, but need to be specified

# The model is loaded in half-precision form. If your graphics card is relatively new, you can load it in torch.bfolat form
# 8bit:
model = AutoModelForCausalLM.from_pretrained('./qwen/Qwen-7B-Chat/', trust_remote_code=True, torch_dtype=torch.half, load_in_8bit=True, device_map="sequential") # 4bit model = AutoModelForCausalLM.from_pretrained('./qwen/Qwen-7B-Chat/',
trust_remote_code=True, 
torch_dtype=torch.half, 
device_map="sequential",
low_cpu_mem_usage=True, # Whether to use cpu to accelerate model loading
load_in_4bit=True, # Whether to load the model with 4-bit precision. If set to True, the model is loaded with 4-bit precision.bnb_4bit_compute_dtype=torch.half, # Data type for 4-bit precision calculation. Here it is set to torch.half, indicating the use of half-precision floating point numbers.
bnb_4bit_quant_type="nf4", # Type of 4-bit precision quantization. Here it is set to "nf4", indicating the use of nf4 quantization type.
bnb_4bit_use_double_quant=True # Whether to use double precision quantization. If set to True, double precision quantization is used.
)
```

Here I would also like to explain the usage of the `device_map` parameter in detail:
- "auto" is to automatically find the best allocation strategy.
- "balanced" is to split evenly on all GPUs.
- "balanced_low_0" will balance the partition model on other GPUs except the first GPU, andIt occupies less resources on the first GPU.
- "sequential" is the opposite of the above, and occupies less resources on the last GPU.
You can use this command to view the allocation of each layer of the model: 

```python
model.hf_device_map
```

### Check whether the definition is successful:

```python
for name, param in model.named_parameters():
print(name, param.shape, param.dtype)
```

## Define LoraConfig

Many parameters can be set in the `LoraConfig` class, but there are not many main parameters. I will briefly talk about them. Interested students can directly read the source code.

- `task_type`: model type
- `target_modules`: the name of the model layer to be trained, mainly the layer of the `attention` part. Different models have different corresponding layer names. You can pass in arrays, strings, or regular expressions.
- `r`: the rank of `lora`, for details, please refer to `Lora` principle
- `lora_alpha`: `Lora alaph`, for details, please refer to `Lora` principle

What is the scaling of `Lora`? Of course not `r` (rank), this scaling is `lora_alpha/r`, and in this `LoraConfig` the scaling is 4 times.

```python
config = LoraConfig(
task_type=TaskType.CAUSAL_LM, 
target_modules=["c_attn", "c_proj", "w1", "w2"],
inference_mode=False, # Training mode
r=8, # Lora rank
lora_alpha=32, # Lora alaph, for specific functions, see Lora principle
lora_dropout=0.1# Dropout ratio
)
```

## Custom TrainingArguments parameters

The source code of the `TrainingArguments` class also introduces the specific functions of each parameter. Of course, you can explore it yourself. Here are a few commonly used ones.

- `output_dir`: output path of the model
- `per_device_train_batch_size`: As the name suggests `batch_size`
- `gradient_accumulation_steps`: Gradient accumulation. If your video memory is small, you can set `batch_size` to a smaller value to increase the gradient accumulation.
- `logging_steps`: How many steps to output once `log`
- `num_train_epochs`: As the name suggests, `epoch`
- `gradient_checkpointing`: Gradient check. Once this is turned on, the model must execute `model.enable_input_require_grads()`. You can explore this principle by yourself, so I won’t go into details here.

```python
args = TrainingArguments(
output_dir="./output/Qwen",
per_device_train_batch_size=8,
gradient_accumulation_steps=2,
logging_steps=10,
num_train_epochs=3,
gradient_checkpointing=True,
save_steps=100,
learning_rate=1e-4,save_on_each_node=True
)
```

## Training with Trainer

Put in the model, the parameters set above, and the data set, OK! Start training!

```python
trainer = Trainer(
model=model,
args=args,
train_dataset=tokenized_id,
data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
```

## Model inference

You can use this more classic way to infer.

```python
model.eval()
ipt = tokenizer("<|im_start|>system\nNow you have to play the role of the woman beside the emperor--Zhen Huan.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n".format("Who are you?", "").strip() + "\nAssistant: ", return_tensors="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length=512, do_sample=True, eos_token_id=tokenizer.eos_token_id, temperature=0.1)[0], skip_special_tokens=True)
```

You can also use Qwen's custom method for inference

```python
response, history = model.chat(tokenizer, "Who are you", history=[], system="Now you have to play the role of the woman next to the emperor--Zhen Huan.")
```