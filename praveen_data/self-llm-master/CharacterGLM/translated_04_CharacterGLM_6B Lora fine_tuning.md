# 04-CharacterGLM-6B-Chat Lora fine-tuning

## Overview

This article briefly introduces how to fine-tune the CharacterGLM-6B-chat model with Lora based on frameworks such as transformers and peft. For the Lora principle, please refer to the blog: [Zhihu | Lora in a simple way](https://zhuanlan.zhihu.com/p/650197598)
The code in this article does not use a distributed framework. Fine-tuning the ChatGLM3-6B-Chat model requires at least 21G of video memory and above, and the model path and dataset path in the script file need to be modified.

## Environment configuration

After completing the basic environment configuration and local model deployment, you also need to install some third-party libraries. You can use the following commands:

```python
pip install transformers==4.37.2
pip install peft==0.4.0.dev0
pip install datasets==2.10.1
pip install accelerate==0.21.0

```

In this section, place the fine-tuning dataset in the root directory [/dataset](https://github.com/datawhalechina/self-llm/blob/master/dataset/huanhuan.json).

## Instruction set construction

LLM fine-tuning generally refers to the instruction fine-tuning process. The so-called instruction fine-tuning means that the fine-tuning data we use is in the form of:

```python
{
"instruction":"Answer the following questions from the user and give the result directly."

"input":"Who is the first Chinese Nobel Prize winner?"

"output":"Mo Yan"
}
```

Where instruction is the user instruction, telling the model the task to be completed; input is the user input, which is the input content required to complete the user instruction; output is the output that the model should give.

That is, our core training goal is to enable the model to have the ability to understand and follow user instructions. Therefore, when constructing the instruction set, we should build a task instruction set specifically for our target task. In this article, we use the [Chat-甄嬛 project](https://github.com/KMnO4-zx/huanhuan-chat) co-opened by the author as an example. Our goal is to build a personalized LLM that can simulate the conversation style of 甄嬛, so the instructions we build are as follows:

```python
{
"instruction": "",
"input":"Who are you?",
"output":"My father is Zhen Yuandao, Shaoqing of Dali Temple."
}
```

All the instruction data sets we constructed are in the root directory.

## The difference and connection between QA and Instruction

QA refers to the form of one question and one answer, usually the user asks a question and the model gives an answer. Instruction comes from Prompt Engineering, which splits the question into two parts: Instruction is used to describe the task, and Input is used to describe the object to be processed.

Training data in question and answer (QA) format is usually used to train models to perform specific tasks. For example, for the question "Please explain the difference between the two MBTI personalities of INFJ and ENTP"

* Question and answer (QA) format:

```python
Instruction:
Input: What is the difference between the two MBTI personalities of INFJ and ENTP?
```

*Instruction format:

```python
Instruction: Please explain the difference between the following two MBTI personalities
Input: INFJ and ENTP
```

## Data formatting

The data trained by Lora needs to be formatted and encoded before being input into the model for training. We generally need to encode the input text as input_ids, encode the output text into labels, and the results after encoding are all multidimensional vectors. We first define a processing function, which is used to encode the input, output text and return an encoded dictionary for each sample:

```python
def process_func(example):
MAX_LENGTH = 512
input_ids, labels = [], []
prompt = tokenizer.encode("User:\n"+"Now you have to play the role of the woman next to the emperor--Zhen Huan.", add_special_tokens=False)
instruction_ = tokenizer.encode("\n".join([example["instruction"], example["input"]]).strip(), add_special_tokens=False,max_length=512)
instruction = tokenizer.encode(prompt + instruction_)
response = tokenizer.encode("CharacterGLM-6B:\n:" + example["output"], add_special_tokens=False) input_ids = instruction + response + [tokenizer.eos_token_id] labels = [tokenizer.pad_token_id] * len(instruction) + response + [tokenizer.eos_token_id] pad_len = MAX_LENGTH - len(input_ids) # print() input_ids += [tokenizer.pad_token_id] * pad_len labels += [tokenizer.pad_token_id] * pad_len labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels] return { "input_ids": input_ids,"labels": labels
}
```

The formatted data, that is, each piece of data fed into the model, is a dictionary containing two key-value pairs: input_ids and labels, where input_ids is the encoding of the input text and labels is the encoding of the output text.

## Load tokenizer and half-precision model

The model is loaded in the form of plate precision. If the graphics card is relatively new, it can be loaded in the form of torch.bfloat. For custom models, the trust_remote_code parameter must be specified as True

```python
tokenizer=AutoTokenizer.from_pretrained('/root/autodl-tmp/THUCoAI/CharacterGLM-6B',use_fast=False,trust_remote_code=True)

model=AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/THUCoAI/CharacterGLM-6B',trust_remote_code=True,torch_dtype=torch.half,device_map="auto")
```

## Define LoraConfig

Many parameters can be set in the LoraConfig class, some of which are shown below:
task_type: model type
target——modules: the name of the model layer to be trained, mainly the layer of the attention part. Different models have different corresponding layer names. You can pass in arrays, strings, or regular expressions.
r: lora rank
lora_alpha: Lora alpha
modules_to_save: specifies that except for the modules that are disassembled into lora, other modules can be fully specified for training

Lora's method is lora_alpha/r, which is scaled by 4 times in this LoraConfig. The essence of this scaling does not change the size of Lora's parameters. The essence is to perform broadcast multiplication on the parameter values ​​inside and perform linear scaling.

```python
config=LoraConfig(
task_type=TaskType.CAUSAL_LM,
target_modules=["query_key_value"],
inference_mode=False,
r=8,
lora_alpha=32,
lora_dropout=0.1
)
```

## Customize TraininArguments parameters

The source code of the TrainingArguments class also introduces the specific role of each parameter. The commonly used parameters are as follows:
output_dir: output path of the model
per_device_train_batch_size: batch_size
gradient_accumulation_steps: gradient accumulation. If the video memory is small, you can set batch_size to a smaller value to increase the gradient accumulation
logging_steps: how many steps to output a log
num_train_epochs: epoch as the name suggests
gradient_checkpointing: gradient check. Once this is turned on, the model must execute
model.enable_input_require_grads()

```python
data_collator=DataCollatorForSeq2Seq(
tokenizer,
model=model,
label_pad_token_id=-100,
pad_to_multiple_of=None,
padding=False
)
args=TrainingArguments(
output_dir="./output/CharacterGLM",
per_device_train_batch_size=4,
gradient_accumulation_steps=2,
logging_steps=10,
num_train_epochs=3,
gradient_checkpointing=True,
save_steps=100,
learning_rate=1e-4,
)
```

## Use Trainer for training

Put in the model, the parameters set above, the data set, and start training

```python
trainer=Trainer(
model=model,
args=args,
train_dataset=tokenized_id,
data_collator=data_collator,
)
trainer.train()
```

## Model inference

```python
model = model.cuda()
ipt = tokenizer("User:{}\n{}".format("Now you have to play the role of the woman beside the emperor--Zhen Huan. Who are you?", "").strip() + "characterGLM-6B:\n", return_tensors="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)
```

## Reload

The model fine-tuned by PEFT can be reloaded and inferred using the following method:

Load the source model and tokenizer;
Use PeftModel to merge the source model and the parameters fine-tuned by PEFT

```python
from peft import Peftmodel model=AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/THUCoAI/CharacterGLM-6B",trust_remote_code=True,low_cpu_mem_usage=True)
tokenizer=AutoTokenizer.from_pretrained("root/autodl-tmp/THUCoAI/CharacterGLM-6B",use_fast=False,trust_remote_code=True)
p_model=PeftModel.from_pretrained(model,model_id="./output/CharatcerGLM/checkpoint-1000/")
ipt = tokenizer("User:{}\n{}".format("Now you have to play the role of the emperor's woman--Zhen Huan. Who are you?", "").strip() + "characterGLM-6B:\n", return_tensors="pt").to(model.device)
tokenizer.decode(p_model.generate(**ipt,max_length=128,do_sample=True)[0],skip_special_tokens=True)
```