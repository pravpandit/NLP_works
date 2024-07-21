# TransNormerLLM-7B Lora fine-tuning

In this section, we briefly introduce how to perform Lora fine-tuning on the TransNormerLLM-1B [Note: TransNormerLLM-358M/1B/7B] model based on frameworks such as transformers and peft. Lora is an efficient fine-tuning method. For a deeper understanding of its principles, please refer to the blog: [Zhihu | Lora in a Simple Way](https://zhuanlan.zhihu.com/p/650197598).

This tutorial will provide you with a [nodebook](./TransNormerLLM-7B-Lora.ipynb) file in the same directory to help you learn better.

## Environment configuration

After completing the basic environment configuration and local model deployment, you also need to install some third-party libraries. Here we have two installation methods, but before installing the dependent library, we first update the pip version (to prevent the version from being too low) and switch the pip installation source (to the domestic source, which can install faster and prevent the download link from timeout)

Enter the following commands in "2.2" line by line in the red box:
```shell
# Upgrade pip
python -m pip install --upgrade pip
# Change the pypi source to accelerate the installation of the librarypip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**Method 1:**
Still enter the following commands in "2.2" line by line in the red box:

```shell
pip install modelscope==1.11.0
pip install "transformers>=4.37.0"
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
pip install transformers_stream_generator==0.0.4
pip install datasets==2.18.0
pip install peft==0.10.0
pip install deepspeed
pip install triton==2.0.0
pip install einops

MAX_JOBS=8 pip install flash-attn --no-build-isolation ``` or ```shell pip install modelscope==1.11.0 "transformers>=4.37.0" streamlit==1.24.0 sentencepiece==0.1.99 accelerate==0.24.1 transformers_stream_generator==0.0.4 datasets ==2.18.0 peft==0.10.0 deepspeed triton==2.0.0 einops MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

**Method 2:**
Replace the following content:
```shell
modelscope==1.11.0
"transformers>=4.37.0"
streamlit==1.24.0
sentencepiece==0.1. 99
accelerate==0.24.1
transformers_stream_generator==0.0.4
datasets==2.18.0
peft==0.10.0
deepspeedtriton==2.0.0
einops
``` 
Use vim to write a requirements.txt file, and then run the command: pip install -r requirements.txt

Then, execute the following command

```bash
MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

> Note: flash-attn installation will be slow, and it will take about ten minutes.

In this tutorial, we will place the fine-tuning dataset `huanhuan.json` in the root directory [/dataset](../dataset/huanhuan.json), and the sample data is taken from [huanhuan.json](https://github.com/datawhalechina/self-llm/blob/master/dataset/huanhuan.json)

## Instruction set construction

LLM fine-tuning generally refers to the instruction fine-tuning process. The so-called instruction fine-tuning means that the fine-tuning data we use is in the form of:

```json
{
"instruction":"Answer the following user questions and only output the answers.",
"input":"1+1 equals how many?",
"output":"2"
}
```

Among them, `instruction` is the user instruction, telling the model what task it needs to complete; `input` is the user input, which is the input content required to complete the user instruction; `output` is the output that the model should give.

That is, our core training goal is to enable the model to understand and follow user instructions. Therefore, when constructing the instruction set, we should build a task instruction set specifically for our target task. For example, in this section we use the open source [Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) project as an example. Our goal is to build a personalized LLM that can simulate the conversation style of 甄嬛, so the instructions we construct are as follows:

```json
{
"instruction": "Who are you?",
"input":"",
"output":"My father is Zhen Yuandao, Shaoqing of Dali Temple."

}
```

Of course, you can also use the training data: `alpaca_data.json`. The sample data is taken from [alpaca_data.json](https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json), the data consists of 52,002 entries and has been reformatted. Its main purpose is to demonstrate how to perform SFT on our model, and its validity is not guaranteed.
All the instruction data sets we constructed are in the root directory.

## Data formatting

The data trained by `Lora` needs to be formatted and encoded before being input to the model for training. Students who are familiar with the `Pytorch` model training process will know that we generally need to encode the input text as input_ids and the output text as `labels`. The results after encoding are all multi-dimensional vectors. We first define a preprocessing function, which is used to encode the input and output text of each sample and return an encoded dictionary:

```python
def process_func(example):
MAX_LENGTH = 384 # Llama tokenizer will split a Chinese character into multiple tokens, so it is necessary to relax some maximum lengths to ensure data integrity
input_ids, attention_mask, labels = [], [], []
instruction = tokenizer(f"<|im_start|>system\nNow you have to play the role of the woman beside the emperor--Zhen Huan<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False) # add_special_tokens does not add special_tokens at the beginning
response = tokenizer(f"{example['output']}", add_special_tokens=False)
input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] # Because we also need to pay attention to eos token, add 1labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id] if len(input_ids) > MAX_LENGTH: # Make a truncation input_ids = input_ids[:MAX_LENGTH] attention_mask = attention_mask[:MAX_LENGTH] labels = labels[:MAX_LENGTH] return { "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels } `` The `Prompt Template` format used by `TransNormerLLM-7B` is as follows: ` ``text <|im_start|>system You are a helpful assistant.<|im_end|>
<|im_start|>user
Who are you? <|im_end|>
<|im_start|>assistant
I am a helpful assistant. <|im_end|>
```

## Load tokenizer and half-precision model

The model is loaded in half-precision format. If your graphics card is relatively new, you can load it in `torch.bfolat` format. For custom models, be sure to specify the `trust_remote_code` parameter as `True`.

```python tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/OpenNLPLab/TransNormerLLM-7B/', use_fast=False, trust_remote_code=True, trust_remote_code=True) model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/OpenNLPLab/TransNormerLLM-7B/', trust_remote_ code=True,device_map="auto",torch_dtype=torch.bfloat16)
```

## Define LoraConfig

Many parameters can be set in the `LoraConfig` class, but there are not many main parameters. I will briefly explain them. Students who are interested can directly look at the source code.

- `task_type`: model type
- `target_modules`: the name of the model layer to be trained, mainly the layer of the `attention` part. Different models have different names for the corresponding layers. You can pass in arrays, strings, or regular expressions.
- `r`: the rank of `lora`, see `Lora` principle for details
- `lora_alpha`: `Lora alaph`, see `Lora` principle for specific functions

What is the scaling of `Lora`? Of course it is not `r` (rank), this scaling is `lora_alpha/r`, and the scaling in this `LoraConfig` is 4 times.

```python config = LoraConfig( task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
inference_mode=False, # Training mode
r=8, # Lora rank
lora_alpha=32, # Lora alaph, see Lora principle for specific functions
lora_dropout=0.1# Dropout ratio
)
```

## Custom TrainingArguments parameters

The source code of the class `TrainingArguments` also introduces the specific functions of each parameter. Of course, you can explore it yourself. Here are a few commonly used ones.

- `output_dir`: output path of the model
- `per_device_train_batch_size`: As the name suggests `batch_size`
- `gradient_accumulation_steps`: gradient accumulation. If your video memory is small, you can set `batch_size` a little smaller to increase the gradient accumulation.
- `logging_steps`: how many steps to output a `log`
- `num_train_epochs`: As the name suggests, `epoch`
- `gradient_checkpointing`: Gradient checking. Once this is enabled, the model must execute `model.enable_input_require_grads()`. You can explore this principle by yourself, so I won’t go into details here.

```python
args = TrainingArguments(
output_dir="./output/TransNormerLLM-7B-Lora",
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

## Load lora weight inference

After training, you can use the following method to load `lora` weights for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = '/root/autodl-tmp/OpenNLPLab/TransNormerLLM-7B/'
lora_path = './output/DeepSeek'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# Load model
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16)

# Load lora weights
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

prompt = "Who are you?"
messages = [
{"role": "system", "content": "Now you have to play the role of the woman next to the emperor--Zhen Huan"},
{"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

generated_ids = model.generate(
model_inputs.input_ids,
max_new_tokens=512 ) generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids) ] response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] print(response) `` `