# DeepSeek-7B-chat Lora fine-tuning

## Overview

In this section, we briefly introduce how to fine-tune the DeepSeek-7B-chat model with Lora based on frameworks such as transformers and peft. Lora is an efficient fine-tuning method. For a deeper understanding of its principles, please refer to the blog: [Zhihu | Lora in a simple way](https://zhuanlan.zhihu.com/p/650197598).

This tutorial will provide you with a [nodebook](./04-DeepSeek-7B-chat%20Lora%20fine-tuning.ipynb) file in the same directory to help you learn better.

## Environment configuration

After completing the basic environment configuration and local model deployment, you also need to install some third-party libraries. You can use the following commands:

```bash
pip install transformers==4.35.2
pip install peft==0.4.0
pip install datasets==2.10.1
pip install accelerate==0.20.3
pip install tiktoken
pip install transformers_stream_generator
```

In this tutorial, we will place the fine-tuning dataset in the root directory [/dataset](../dataset/huanhuan.json).

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

That is, our core training goal is to enable the model to understand and follow user instructions. Therefore, when constructing the instruction set, we should build a task instruction set specifically for our target task. For example, in this section we use the [Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) project co-opened by the author as an example. Our goal is to build a personalized LLM that can simulate the conversation style of 甄嬛, so the instructions we construct are as follows:

```json
{
"instruction": "Now you have to play the role of the emperor's woman - Zhen Huan",
"input":"Who are you?",
"output":"My father is Zhen Yuandao, the Shaoqing of Dali Temple."
}
```

All the instruction data sets we constructed are in the root directory.

## Data formatting

The data trained by `Lora` needs to be formatted and encoded before being input to the model for training. If you are familiar with the `Pytorch` model training process, you will know that we generally need to encode the input text as input_ids and the output text as `labels`. The results after encoding are all multi-dimensional vectors. We first define a preprocessing function, which is used to encode the input and output text of each sample and return an encoded dictionary:

```python
def process_func(example):
MAX_LENGTH = 384 # Llama tokenizer will split a Chinese character into multiple tokens, so it is necessary to relax some maximum lengths to ensure data integrity
input_ids, attention_mask, labels = [], [], []
instruction = tokenizer(f"User: {example['instruction']+example['input']}\n\n", add_special_tokens=False) # add_special_tokens does not add special_tokens at the beginning
response = tokenizer(f"Assistant: {example['output']}<｜end of sentence｜>", add_special_tokens=False)
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

The formatted input here refers to the instruction introduction in the readme of the official github repository of [DeepSeek](https://github.com/deepseek-ai/DeepSeek-LLM).

```text
User: {messages[0]['content']}

Assistant: {messages[1]['content']}<｜end of sentence｜>User: {messages[2]['content']}

Assistant:
```

## Load tokenizer and half-precision model

The model is loaded in half-precision format. If your graphics card is relatively new, you can use `torch.bfolat` to load it. For custom models, be sure to specify the `trust_remote_code` parameter as `True`.

```python tokenizer = AutoTokenizer.from_pretrained('./deepseek-ai/deepseek-llm-7b-chat/', use_fast=False, trust_remote_code=True) tokenizer.padding_side = 'right' # padding on the right model = AutoModelForCausalLM.from_pretrained('./deepseek-ai/deepseek-llm-7b-chat/', trust_remote_code =True, torch_dtype=torch.half, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained('./deepseek-ai/deepseek-llm-7b-chat/')
model.generation_config.pad_token_id = model.generation_config.eos_token_id
```

## Define LoraConfig

Many parameters can be set in the `LoraConfig` class, but there are not many main parameters. Let me briefly talk about them. Interested students can directly look at the source code.

- `task_type`: model type
- `target_modules`: the name of the model layer to be trained, mainly the layer of the `attention` part. Different models have different names for the corresponding layers. You can pass in an array, a string, or a regular expression.
- `r`: the rank of `lora`, see `Lora` principle for details
- `lora_alpha`: `Lora alaph`, see `Lora` principle for specific functions

What is the scaling of `Lora`? Of course it is not `r` (rank), this scaling is `lora_alpha/r`,In this `LoraConfig`, the scaling is 4 times.

```python
config = LoraConfig(
task_type=TaskType.CAUSAL_LM, 
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
- `gradient_accumulation_steps`: gradient accumulation. If your video memory is small, you can set `batch_size` to a smaller value to increase the gradient accumulation.
- `logging_steps`: how many steps to output a `log`
- `num_train_epochs`: as the name suggests, `epoch`
- `gradient_checkpointing`: gradient check. Once this is turned on, the model must execute `model.enable_input_require_grads()`. You can explore this principle by yourself, so I won't go into details here.

```python
args = TrainingArguments(
output_dir="./output/DeepSeek",
per_device_train_batch_size=8,
gradient_accumulation_steps=2,
logging_steps=10,
num_train_epochs=3,
save_steps=100,
learning_rate=1e-4,
save_on_each_node=True,
gradient_checkpointing=True
)
```

## Train with Trainer

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

You can use this more classic way to infer:

```python
text = "Miss, other girls are seeking to be selected, only our lady wants to be put down, the Bodhisattva must remember it really——"
inputs = tokenizer(f"User: {text}\n\n", return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

Inference result: (The effect is quite good, a little beyond my expectation)

```text
User: Miss, other girls are seeking to be selected, but our girl is the only one who wants to be left out. The Bodhisattva must remember it very well——

Assistant: The Bodhisattva can also make mistakes sometimes.
```