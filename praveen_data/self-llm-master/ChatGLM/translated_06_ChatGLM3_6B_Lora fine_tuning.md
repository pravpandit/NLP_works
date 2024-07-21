# 06-ChatGLM3-6B-chat Lora fine-tuning

## Overview

In this section, we briefly introduce how to fine-tune the ChatGLM3-6B-chat model with Lora based on frameworks such as transformers and peft. Lora is an efficient fine-tuning method. For a deeper understanding of its principles, please refer to the blog: [Zhihu | A Simple Introduction to Lora](https://zhuanlan.zhihu.com/p/650197598).

The code script described in this section is in the same directory [ChatGLM3-6B-chat Lora fine-tuning](./06-ChatGLM3-6B-Lora fine-tuning.py). Run the script to perform the fine-tuning process, but note that the code in this article does not use a distributed framework. Fine-tuning the ChatGLM3-6B-Chat model requires at least 21G of video memory and above, and the model path and dataset path in the script file need to be modified.

This tutorial will provide you with a [nodebook](./06-ChatGLM3-6B-Lora fine-tuning.ipynb) file in the same directory to help you learn better. 
Thanks to the detailed explanation and tutorial: [transformers-code](https://github.com/zyds/transformers-code)

## Environment configuration

After completing the basic environment configuration and local model deployment, you also need to install some third-party libraries. You can use the following commands:

```bash
pip install transformers==4.37.2
pip install peft==0.4.0.dev0
pip install datasets==2.10.1
pip install accelerate==0.21.0
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

That is, our core training goalIt is to enable the model to understand and follow user instructions. Therefore, when constructing the instruction set, we should build a task instruction set specifically for our target task. For example, in this section, we use the [Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) project co-opened by the author as an example. Our goal is to build a personalized LLM that can simulate the conversation style of 甄嬛, so the instructions we construct are as follows:

```json
{
"instruction": "",
"input":"Who are you?",
"output":"My father is Zhen Yuandao, Shaoqing of Dali Temple."

}
```
All the instruction data sets we constructed are in the root directory.

## The difference and connection between QA and Instruction
QA refers to the form of one question and one answer, usually the user asks a question and the model gives an answer. Instruction is derived from Prompt Engineering, which splits the problem into two parts: Instruction is used to describe the task, and Input is used to describe the object to be processed.

Question-answering (QA) format training data is usually used to train models to answer knowledge-based questions, while instruction format training data is more suitable for training models to perform specific tasks. For example, for the question "Please explain VC SilverThe difference between qiao tablets and shuanghuanglian oral liquid"
- Question and answer (QA) format:
```
Instruction:

Input: What is the difference between VC Yinqiao tablets and Shuanghuanglian oral liquid?
```

- Instruction format:
```
Instruction: Please explain the difference between the following two medicines.
Input: VC Yinqiao tablets and Shuanghuanglian oral liquid.
```
The form of instruction may make the model have better generalization ability because it emphasizes the nature of the task, not just the specific input. Usually the instruction format and the question and answer format can be converted to each other.

## Data formatting

The data trained by `Lora` needs to be formatted and encoded before being input to the model for training. Students who are familiar with the `Pytorch` model training process will know that we generally need to encode the input text as input_ids and the output text as `labels`, the results after encoding are all multi-dimensional vectors. We first define a preprocessing function, which is used to encode the input and output text of each sample and return an encoded dictionary:

```python
def process_func(example):
MAX_LENGTH = 512
input_ids, labels = [], []
instruction = tokenizer.encode(text="\n".join(["<|system|>", "Now you have to play the role of the woman beside the emperor--Zhen Huan", "<|user|>", 
example["instruction"] + example["input"] + "<|assistant|>"]).strip() + "\n",
add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)

response = tokenizer.encode(text=example["output"], add_special_tokens=False, truncation=True,
max_length=MAX_LENGTH)

input_ids = instruction + response + [tokenizer.eos_token_id]
labels = [tokenizer.pad_token_id] * len(instruction) + response + [tokenizer.eos_token_id]
pad_len = MAX_LENGTH - len(input_ids)
input_ids += [tokenizer.pad_token_id] * pad_len
labels += [tokenizer.pad_token_id] * pad_len
labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

return {
"input_ids": input_ids,
"labels": labels
}
```

The formatted data, that is, each piece of data sent to the model, is a dictionary containing `input_ids`, `labels`Two key-value pairs, where `input_ids` is the encoding of the input text, and `labels` is the encoding of the output text. After decoding, it should be like this:

```text
[gMASK]sop <|system|>
Now you have to play the role of the woman next to the emperor--Zhen Huan
<|user|>
This doctor Wen is also weird. Everyone knows that the doctor cannot take the pulse and diagnose diseases for people other than the royal family without the emperor's order. He comes to our house every ten days or half a month. <|assistant|>
You two talk too much. I should ask Doctor Wen for a dose of medicine to treat you well.
```

Why is it in this form? Good question! Different models have different formatted inputs, so we need to check the training source code of our deep model. Because the Lora fine-tuning effect should be the best in the form of fine-tuning according to the original model instruction, we still follow the input format of the original model. OK, here is a link to the source code. If you are interested, you can explore it yourself: 

[hugging face ChatGLM3 repository](https://github.com/THUDM/ChatGLM3/blob/main/finetune_chatmodel_demo/preprocess_utils.py): the `InputOutputDataset` class.In addition, you can also refer to this repository for ChatGLM data processing [LLaMA-Factory](https://github.com/KMnO4-zx/LLaMA-Factory/blob/main/src/llmtuner/data/template.py).

## Load tokenizer and half-precision model

The model is loaded in half-precision form. If your graphics card is relatively new, you can load it in the form of `torch.bfolat`. For custom models, be sure to specify the `trust_remote_code` parameter as `True`.

```python
tokenizer = AutoTokenizer.from_pretrained('./model/chatglm3-6b', use_fast=False, trust_remote_code=True)

# The model is loaded in half-precision format. If your graphics card is relatively new, you can load it in torch.bfolat format
model = AutoModelForCausalLM.from_pretrained('./model/chatglm3-6b', trust_remote_code=True, torch_dtype=torch.half, device_map="auto")
```

## Define LoraConfig

Many parameters can be set in the `LoraConfig` class, but there are not many main parameters. I will briefly explain them. Students who are interested can directly look at the source code.

- `task_type`: model type
- `target_modules`: the name of the model layer to be trained, mainly the layer of the `attention` part. Different models have different corresponding layer names. You can pass in arrays, strings, or regular expressions.
- `r`: the rank of `lora`, see `Lora` principle for details
- `lora_alpha`: `Lora alaph`, see `Lora` principle for specific functions
- `modules_to_save` specifies that except for the modules that are disassembled into lora, other modules can be fully specified for training.

What is the scaling of `Lora`? Of course it is not `r` (rank), this scaling is `lora_alpha/r`, and the scaling in this `LoraConfig` is 4 times.
The essence of this scaling does not change the size of LoRa's parameters. The essence is to perform broadcast multiplication on the parameter values ​​and perform linear scaling.

```python
config = LoraConfig(
task_type=TaskType.CAUSAL_LM, 
target_modules=["query_key_value"],
inference_mode=False, # Training mode
r=8, # Lora rank
lora_alpha=32, # Lora alaph, see Lora principle for specific functions
lora_dropout=0.1# Dropout ratio
)
```

## Customize TrainingArguments parameters

The source code of the class `TrainingArguments` also introduces the specific functions of each parameter. Of course, you can explore it yourself. Here are a few commonly used ones.

- `output_dir`: output path of the model
- `per_device_train_batch_size`: As the name suggests `batch_size`
- `gradient_accumulation_steps`: gradient accumulation. If your video memory is small, you can set `batch_size` to a smaller value to increase the gradient accumulation.
- `logging_steps`: how many steps to output `log` once
- `num_train_epochs`: As the name implies, `epoch`
- `gradient_checkpointing`: gradient checking. Once this is turned on, the model must execute `model.enable_input_require_grads()`. You can explore this principle by yourself, so I won’t go into details here.

```python
# Data collator The GLM source warehouse has repackaged its own data_collator, which is used here.

data_collator = DataCollatorForSeq2Seq(
tokenizer,
model=model,
label_pad_token_id=-100,
pad_to_multiple_of=None,
padding=False
)

args = TrainingArguments(
output_dir="./output/ChatGLM",
per_device_train_batch_size=4,
gradient_accumulation_steps=2,
logging_steps=10,
num_train_epochs=3,
gradient_checkpointing=True,
save_steps=100,
learning_rate=1e-4,
)
```

### Training with Trainer

Put in the model, the parameters set above, and the data set, OK! Start training!

```python
trainer = Trainer(
model=model,
args=args,
train_dataset=tokenized_id,
data_collator=data_collator,
)
trainer.train()
```

## Model inference

You can use this more classic way to infer.

```python
model.eval()
model = model.cuda()
ipt = tokenizer("<|system|>\nNow you have to play the role of the woman beside the emperor--Zhen Huan\n<|user|>\n {}\n{}".format("Who are you?", "").strip() + "<|assistant|>\n", return_tensors="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)
```

## Reload
The model fine-tuned by PEFT can be reloaded and inferred using the following method:
- Load the source model and tokenizer;
- Use `PeftModel` to merge the source model and the parameters fine-tuned by PEFT.

```python
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("./model/chatglm3-6b", trust_remote_code=True, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("./model/chatglm3-6b", use_fast=False, trust_remote_code=True)

p_model = PeftModel.from_pretrained(model, model_id="./output/ChatGLM/checkpoint-1000/") # Load the trained LoRa weights

ipt = tokenizer("<|system|>\nNow you have to play the role of the emperor's woman--Zhen Huan\n<|user|>\n {}\n{}".format("Who are you?", "").strip() + "<|assistant|>\n", return_tensors="pt").to(model.device)
tokenizer.decode(p_model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)

```