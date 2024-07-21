# BlueLM-7B-Chat Lora fine-tuning

## Overview

In this section, we briefly introduce how to fine-tune the BlueLM-7B-Chat model with Lora based on frameworks such as transformers and peft. Lora is an efficient fine-tuning method. For a deeper understanding of its principles, please refer to the blog: [Zhihu | Lora in a simple way](https://zhuanlan.zhihu.com/p/650197598).

This tutorial will provide you with a [notebook](./04-BlueLM-7B-Chat%20Lora%20fine-tuning.ipynb) file in the same directory to help you learn better.

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

In this tutorial, we will place the fine-tuning dataset in the root directory [/dataset](https://github.com/datawhalechina/self-llm/blob/master/dataset/huanhuan.json).

## Instruction set construction

LLM fine-tuning generally refers to the instruction fine-tuning process. The so-called instruction fine-tuning means that the fine-tuning data we use is in the form of:

```json
{
"instruction": "Explain what artificial intelligence is.\n",
"input": "",
"output": "Artificial intelligence is a technology that uses computer programs and algorithms to create human-like intelligence, allowing computers to show human-like capabilities in problem solving, learning, reasoning, and natural language processing."
}
```

Among them, `instruction` is the user instruction, telling the model what task it needs to complete; `input` is the user input, which is the input content required to complete the user instruction; `output` is the output that the model should give. In BlueLM, the target format of the data is as follows

```json
{
"inputs": "[|Human|]: Explain what artificial intelligence is.\n[|AI|]:", 
"targetArtificial intelligence is a technology that uses computer programs and algorithms to create human-like intelligence, allowing computers to exhibit human-like capabilities in problem solving, learning, reasoning, and natural language processing. "}
```

## Data formatting

The data trained by `Lora` needs to be formatted and encoded before being input to the model for training. Students who are familiar with the `Pytorch` model training process will know that we generally need to encode the input text into input_ids and the output text into `labels`. The results after encoding are all multi-dimensional vectors. We first define a preprocessing function, which is used to encode the input and output text of each sample and return an encoded dictionary:

```python
def process_func(example):
MAX_LENGTH = 384
input_ids = []
labels = []

instruction = tokenizer(text=f"[|Human|]: Now you have to play the role of the woman next to the emperor--Zhen Huan\n\n {example['instruction']}{example['input']}[|AI|]:", add_sspecial_tokens=False) response = tokenizer(text=f"{example['output']}", add_special_tokens=False) input_ids = [tokenizer.bos_token_id] + instruction["input_ids"] + response["input_ids"] + [tokenizer .eos_token_id] labels = [tokenizer.bos_token_id] + [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id] if len(input_ids) > MAX_LENGTH: input_ids = input_ids[ :MAX_LENGTH] labels = labels[:MAX_LENGTH] return {
"input_ids": input_ids,
"labels": labels
}
```

The formatted data, that is, each piece of data sent to the model, is a dictionary containing two key-value pairs, `input_ids` and `labels`, where `input_ids` is the encoding of the input text and `labels` is the encoding of the output text. After decoding, it should be like this:

```json
<s> [|Human|]: Now you have to play the role of the woman beside the emperor--Zhen Huan\n\n This doctor Wen is also weird. Everyone knows that the doctor cannot take the pulse and diagnose diseases for people other than the royal family without the emperor's order. He is so good that he comes to our house every ten days or half a month. [|AI|]: You two talk too much. I should ask Doctor Wen for a dose of medicine to treat you well. </s>
```

Why is it in this form? Good question! Different models have different formatted inputs. BlueLM only has two roles, `[|Human|] and [|AI|]`, so naturally the data format is like this.

## Load tokenizer and model

```python
import torch

model = AutoModelForCausalLM.from_pretrained('vivo-ai/BlueLM-7B-Chat', trust_remote_code=True, torch_dtype=torch.half, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained('vivo-ai/BlueLM-7B-Chat')
model.generation_config.pad_token_id = model.generation_config.eos_token_id
```

## Define LoraConfig

Many parameters can be set in the `LoraConfig` class, but there are not many main parameters. I will briefly talk about them. Interested students can directly read the source code.

- `task_type`: model type
- `target_modules`: the name of the model layer to be trained, mainly the layer of the `attention` part. Different models have different names for the corresponding layers. You can pass in an array, a string, or a regular expression.
- `r`: the rank of `lora`, you can see the `Lora` principle for details
- `lora_alpha`: `Lora alaph`, for specific functions, see `Lora` principle

What is the scaling of `Lora`? Of course it is not `r` (rank), this scaling is `lora_alpha/r`, and the scaling in this `LoraConfig` is 4 times.

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
- `per_device_train_batch_size`: As the name implies `batch_size`
- `gradient_accumulation_steps`: Gradient accumulation. If your video memory is small, you can set `batch_size` to a smaller value to increase the gradient accumulation.
- `logging_steps`: How many steps to output once `log`
- `num_train_epochs`: As the name implies `epoch`
- `gradient_checkpointing`: Gradient check. Once this is enabled, the model must execute `model.enable_input_require_grads()`. You can explore this principle by yourself, so I won't go into details here.

```python
args = TrainingArguments(
output_dir="./output/Qwen",
per_device_train_batch_size=8,
gradient_accumulation_steps=2,
logging_steps=10,
num_train_epochs=3,
gradient_checkpointing=True,
save_steps=100,
learning_rate=1e-4,
save_on_each_node=True
)
```

## Use Trainer to train

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

Use the most common way to perform inference

```python
text = "Miss, other girls are seeking to be selected, only our lady wants to be put down, the Bodhisattva must remember it really——"
inputs = tokenizer(f"[|Human|]:{text}[|AI|]:", return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

For the complete code, please see: [BlueLM-7B-Chat Lora Fine-tuning](./04-BlueLM-7B-Chat%20Lora%20Fine-tuning.ipynb)