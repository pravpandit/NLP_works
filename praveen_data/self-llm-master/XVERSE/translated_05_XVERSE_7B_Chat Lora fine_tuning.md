# XVERSE-7B-Chat Lora fine-tuning

## Overview

In this section, we briefly introduce how to fine-tune the XVERSE-7B-Chat model with Lora based on frameworks such as transformers and peft. Lora is an efficient fine-tuning method. For a deeper understanding of its principles, please refer to the blog: [Zhihu | Lora in a simple way](https://zhuanlan.zhihu.com/p/650197598).

This tutorial will provide you with a [notebook](./05-XVERSE-7B-Chat%20Lora%20fine-tuning.ipynb) file in the same directory to help you learn better.

## Environment configuration

After completing the basic environment configuration and local model deployment, you also need to install some third-party libraries. To facilitate your practice, I packaged the environment in the code folder. You can use the following command:

```bash
cd code
pip install -r requirement.txt
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

Among them, `instruction` is the user instruction, telling the model what task it needs to complete; `input` is the user input, which is the input content required to complete the user instruction; `output` is the output that the model should give. In XVERSE, the target format of data is as follows

```json
{
"inputs": "Human: Explain what artificial intelligence is.\n Assistant:", 
"targets": "Artificial intelligence is a technology that uses computer programs and algorithms to create human-like intelligence, allowing computers to show human-like capabilities in problem solving, learning, reasoning, and natural language processing."}
```

## Data formatting

The data trained by `Lora` needs to be formattedAfter being quantized and encoded, it is input to the model for training. Students who are familiar with the `Pytorch` model training process will know that we generally need to encode the input text into input_ids and the output text into `labels`. The results after encoding are all multi-dimensional vectors. We first define a preprocessing function, which is used to encode the input and output text of each sample and return an encoded dictionary:

```python
def process_func(example):
MAX_LENGTH = 384
input_ids = []
labels = []

instruction = tokenizer(text=f"Human: Now you have to play the role of the woman beside the emperor--Zhen Huan\n\n {example['instruction']}{example['input']}Assistant:", add_special_tokens=False)
response = tokenizer(text=f"{example['output']}", add_special_tokens=False)input_ids = [tokenizer.bos_token_id] + instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
labels = [tokenizer.bos_token_id] + [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
if len(input_ids) > MAX_LENGTH:
input_ids = input_ids[:MAX_LENGTH]
labels = labels[:MAX_LENGTH]

return {
"input_ids": input_ids,
"labels": labels
}
```

Formatted data, that is, every piece of data fed into the model,It is a dictionary, containing two key-value pairs, `input_ids` and `labels`, where `input_ids` is the encoding of the input text, and `labels` is the encoding of the output text. After decoding, it should be like this:

```json
'<|startoftext|>Human: Now you have to play the role of the woman beside the emperor--Zhen Huan\n\n This doctor Wen is also weird. Everyone knows that the doctor cannot take the pulse and diagnose diseases for people other than the royal family without the emperor's order. But he comes to our house every ten days or half a month. Assistant: You two talk too much. I should ask doctor Wen for a dose of medicine to treat you well. <|endoftext|>'
```

Why is it in this form? Good question! Different models have different formatted inputs, because in XVERSE its template is like this: `["Human: {{content}}\n\nAssistant: "]`, so the format is naturally like this, and the text start token and end token of XVERSE are also different.

## Load tokenizer and model

```python
import torch

model = AutoModelForCausalLM.from_pretrained('xverse/XVERSE-7B-Chat', trust_remote_code=True, torch_dtype=torch.half, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained('xverse/XVERSE-7B-Chat')
```

## Define LoraConfig

Many parameters can be set in the `LoraConfig` class, but there are not many main parameters. I will briefly talk about them. Students who are interested can directly look at the source code.

- `task_type`: model type
- `target_modules`: the name of the model layer to be trained, mainly the layer of the `attention` part. Different models have different names for the corresponding layers. You can pass in an array, a string, or a regular expression.
- `r`: the rank of `lora`, see `Lora` principle for details
- `lora_alpha`: `Lora alaph`, see `Lora` principle for specific functions

What is the scaling of `Lora`? Of course it is not `r` (rank), this scaling is `lora_alpha/r`, in this `LoraConfig` the scaling is4 times.

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
- `gradient_accumulation_steps`: gradient accumulation, if your video memory is small, you can set `batch_size` to a smaller value, the gradient accumulation increases.
- `logging_steps`: how many steps, output once `log`
- `num_train_epochs`: as the name suggests `epoch`
- `gradient_checkpointing`: gradient check, once this is turned on, the model must execute `model.enable_input_require_grads()`, this principle can be explored by yourself, I won't go into details here.

```python
args = TrainingArguments(
output_dir="./output/BlueLM",
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

PutPut the model in, put the parameters set above in, put the data set in, OK! Start training! 

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

Use the most common way to perform inference:
> Note that `return_token_type_ids` is set to false

```python
model.eval()
text = "Miss, other girls are seeking to be selected, only our lady wants to be put down, the Bodhisattva must remember it really——"
inputs = tokenizer(f"Human:{text} Assistant:", return_tensors="pt", return_token_type_ids=False, )
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

For the complete code, please see: [XVERSE-7B-Chat Lora Fine-tuning](./05-XVERSE-7B-Chat%20Lora%20Fine-tuning.py)