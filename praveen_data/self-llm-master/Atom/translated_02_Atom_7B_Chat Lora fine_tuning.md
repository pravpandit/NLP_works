# Fine-tuning Lora instructions for Atom-7B-Chat

## Overview

In this section, we briefly introduce how to fine-tune the Atom-7B-Chat model with Lora based on frameworks such as transformers and peft. Lora is an efficient fine-tuning method. For a deeper understanding of its principles, please refer to the blog: [Zhihu | A Simple Introduction to Lora](https://zhuanlan.zhihu.com/p/650197598).

The code scripts described in this section are in the same directory [02-Atom-7B-Chat Lora](../Atom/02-Atom-7B-Chat%20Lora%20fine-tuning/train.py). You can run the [train.sh](../Atom/02-Atom-7B-Chat%20Lora%20fine-tuning/train.sh) script in the directory to perform the fine-tuning process. However, please note that the code in this article does not use a distributed framework, and fine-tuning the Atom-7B model requires at least 32G of video memory or more.

## Environment configuration

After completing the basic environment configuration and local model deployment, you also need to install some third-party libraries. You can use the following command:

```bash
pip install transformers==4.36.0.dev0
pip install peft==0.4.0.dev0
pip install datasets==2.10.1
pip install accelerate==0.20.3
```

In this tutorial, we will place the fine-tuning dataset in the root directory [/dataset](../dataset/huanhuan.jsonl) and the base model parameters in the root directory [/model](../model).

## Instruction set construction

LLM fine-tuning generally refers to the instruction fine-tuning process. The so-called instruction fine-tuning means that the fine-tuning data we use is in the form of:

```json
{
"instrution":"Answer the following user question and only output the answer.",
"input":"What is 1+1?",
"output":"2"
}
```

Among them, `instruction` is the user instruction, telling the model what task it needs to complete; `input` is the user input, which is the input content required to complete the user instruction; `output` is the output that the model should give.

That is, our core training goal is to enable the model to understand and follow user instructions. Therefore, when constructing the instruction set, we should build a task instruction set specifically for our target task. For example, in this section we use the open source [ChAt-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) project is used as an example. Our goal is to build a personalized LLM that can simulate the conversation style of 甄嬛, so the instructions we construct are as follows:

```json
{
"instruction": "Please refer to Zhen Huan's speaking style and tone in the following content and answer my questions. Zhen Huan's speaking style needs to be colloquial. The reply content should not exceed 30 words, and the number of words should be as short as possible.\nConversational style example content:\n```User: Although this song does not directly describe the love between men and women, every word describes the joy of the woman after the two hearts fall in love, and 'double golden partridges' also means a pair of twins.\nUser: In this case, why didn't An Changzai sing out the love of flowers? Could it be that she was unhappy to see the emperor and me together, so she couldn't sing well?\nZhen Huan: Reporting to Concubine Hua, An Changzai caught a cold in the morning and her throat is a little uncomfortable.\n\nUser: Didn't you want to see the white plums in the yard? Why did you come back so soon?\nZhen Huan: I feel dizzy after looking at the snow scene for a long time. Si Lang planted white plums in the garden with good intentions, but when it snowed, it blended into the snow scene and I couldn't see it.",
"input":"Who are you?",
"output":"My father is Zhen Yuandao, the Shaoqing of Dali Temple."

}
```

All the instruction data sets we constructed are in the root directory。

## Data formatting

The data for Lora training needs to be formatted and encoded before being input to the model for training. Students who are familiar with the Pytorch model training process will know that we generally need to encode the input text as `input_ids` and the output text as `labels`. The results after encoding are all multi-dimensional vectors. We first define a preprocessing function, which is used to encode the input and output text of each sample and return an encoded dictionary:

```python
def preprocess(tokenizer, config, example, max_seq_length):
'''
args:
tokenizer: tokenizer, imported Atom model tokenizer
config: model configuration, imported Atom model configuration
example: sample to be processed
max_seq_length: maximum length of text
returns: dictionary, including inputs_id and seq_len
'''
# Concatenate instruction and input according to the format of Atom SFT
prompt = "<s>Human: " + example["instruction"] + "Please answer the user's question: " + example["input"] + "\n" + "</s><s>Assistant:"
target = example["output"]
# Use the tokenizer for encoding, set truncation to True to avoid long samples
prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
target_ids = tokenizer.encode(
target,
max_length=max_seq_length,
truncation=True,
add_special_tokens=False)
# Add the end character EOS
input_ids = prompt_ids + target_ids + [config.eos_token_id]
# Add inputs_idsIt is returned together with seq_len. Inputs and labels will be split according to seq_len later.
return {"input_ids": input_ids, "seq_len": len(prompt_ids)}
```

The above code will format each sample. Let's define another function. This function processes the source training data based on the above function:

```python
# Read source training data and process it
def read_jsonl(path, max_seq_length, model_path, skip_overlength=False):
'''
args:
path: training data path
max_seq_length: maximum length of text
model_path: model path, mainly for loading tokenizer and configuration
returns: use yield to return formatted features
'''
# Load the tokenizer and configuration parameters of the model
tokenizer = transformers.AutoTokenizer.from_pretrained(
model_path, trust_remote_code=True)
config = transformers.AutoConfig.from_pretrained(
model_path, trust_remote_code=True, device_map='auto')
# Read source file
with open(path, "r") as f:
# jsonl data needs to be read into character first by readlines, and then loaded with json
lst = [json.loads(line) for line in f.readlines()]
print("Load jsonl dataset, total data is {}".format(len(lst)))
# Process each sample in turn
for example in tqdm(lst):
# Call the preprocessing function above
feature = preprocess(tokenizer, config, example, max_seq_length)
# If skipping overlong samples is set
if skip_overlength and len(feature["input_ids"]) > max_seq_length:
continue
# Truncate overlong samples
feature["input_ids"] = feature["input_ids"][:max_seq_length]
# Return iterator through yield
yield feature
```

After completing the above function, we use the from_generator function provided by the datasets library to generate the Dataset object of our data according to the above function:

```python
# Generate a Dataset object through the iterator returned by the read_jsonl function. This Dataset object can be used directly in the transformers framework
dataset = datasets.Dataset.from_generator(lambda: read_jsonl(
finetune_args.dataset_path, finetune_args.max_seq_length, finetune_args.model_path, finetune_args.skip_overlength
)
) 
```

## Sampling function

In order to dynamically fill each batch of data and avoid wasting resources, we did not perform the filling operation in the generation function, so we need to define a custom sampling function, which replaces the default sampling function in torch and implements the filling, labels masking and other operations in a custom way. It will be passed to the trainer as a lambda function later:

```python
# Custom sampling function
def data_collator(features: list, tokenizer) -> dict:
'''
args:
features: a batch of data
tokenizer: word segmenter
returns: formatted features
'''
# Statistics batch, pad them to the length of all data
len_ids = [len(feature["input_ids"]) for feature in features]
# Pad to the maximum length
longest = max(len_ids)
# Store input_ids and labels separately
input_ids = []
labels_list = []
# Some models do not define PAD, so we use UNK instead
if tokenizer.pad_token_id is None:
tokenizer.pad_token_id = tokenizer.unk_token_id
# Start processing from the longest text to optimize memory usage
for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
ids = feature["input_ids"]
seq_len = feature["seq_len"]# labels is the result of retaining the output after input PAD, using -100 to indicate masking and padding, -100 will be automatically ignored when calculating loss
labels = (
[-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
)
ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
_ids = torch.LongTensor(ids)
labels_list.append(torch.LongTensor(labels))
input_ids.append(_ids)
# Concatenate in the 0th dimension, that is, form a matrix of batch_size*n*n
input_ids = torch.stack(input_ids)
labels = torch.stack(labels_list)
returnrn {
"input_ids": input_ids,
"labels": labels,
}
```

## Custom Trainer

For Lora fine-tuning, we need to inherit a custom Trainer based on the basic Trainer to implement Loss calculation (required for some models) and save Lora parameters:

```python
# Custom Trainer, inherited from transformers.trainer
class ModifiedTrainer(Trainer):
# Rewrite the loss calculation function to avoid LLaMA class model undefined loss calculation
def compute_loss(self, model, inputs, return_outputs=False):
# 7B
return model(
input_ids=inputs["input_ids"],
labels=inputs["labels"],
).loss
# Rewrite model save function, thereby saving the Lora parameters of the model
def save_model(self, output_dir=None, _internal_call=False):
from transformers.trainer import TRAINING_ARGS_NAME
# If the output path does not exist, create one
os.makedirs(output_dir, exist_ok=True)
# Save various hyperparameters for model training
torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
# Select all parameters whose gradients are not frozen, that is, all Lora parameters involved in the update
saved_params = {
k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
}
# Save all Lora parameter
torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))
```

## Parameter parsing

Since model fine-tuning requires many parameters, it is best to pass them in via the command line and then call them through a bash script (such as the [train.sh](../Atom/02-Atom-7B-Chat-Lora/train.sh) script). There are many fine-tuning parameters, and transformers provide many parameter parsing. We need to define an additional parameter parsing class to parse the parameters that transformers do not provide for Lora fine-tuning:

```python
# dataclass: Python class modifier, data class, encapsulates __init__(), __repr__(), and __eq__() functions
@dataclass
class FinetuneArguments:
# Fine-tuning parameters
# field: dataclass function, used to specify variable initialization
# Training set path
dataset_path: str = field(default="../../dataset/huanhuan.jsonl")
# Base model parameter path
model_path: str = field(default="../../dataset/model")
# Lora rank
lora_rank: int = field(default=8)
# Maximum text length
max_seq_length: int = field(default=256)
# Whether to skip overlong text
skip_overlength: bool = field(default=False)
# Whether to continue training from the breakpoint
continue_training: bool = field(default=False)
# Breakpoint path, if you continue training from the breakpoint, you need to pass it in
checkpoint: str = field(default=None)
```

## Training

After completing the above definition and implementation, we can officially start our training process. First, we need to parse the training parameters passed in by the script. We use the HfArgumentParser provided by transformersFunction, the parsed parameters include the TrainingArguments class provided by transformers (including some common training parameters) and our custom FinetuneArguments class:

```python
# Parse command line parameters
finetune_args, training_args = HfArgumentParser(
(FinetuneArguments, TrainingArguments)
).parse_args_into_dataclasses()
```

Next, load the base model and perform some configuration:

```python
# Initialize the base model
tokenizer = AutoTokenizer.from_pretrained(finetune_args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
finetune_args.model_path, trust_remote_code=True, device_map="auto")
print("From{} Load model successfully".format(finetune_args.model_path))

# Enable gradient checkpointing, allowing the model to discard some intermediate activation values ​​during forward calculation and recalculate in back propagation, thereby optimizing memory usage
model.gradient_checkpointing_enable()
# Ensure that the input vector can calculate the gradient
model.enable_input_require_grads()
# Turn off the cache during training to improve calculation efficiency. It should be turned on during inference
model.config.use_cache = (
False 
)
```

Then set the Lora parameters:

```python
# Set peft parameters
# Manually determine the Lora layer (Note: In theory, we can automatically find all Lora layers, but there is a bug on LLaMA-like models)
target_modules = ['W_pack', 'down_proj', 'o_proj', 'gate_proj', 'up_proj']
# Configure Lora parameters
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, # Task is language modeling
inference_mode=False, # Training mode
r=finetune_args.lora_rank, # Lora rank
lora_alpha=32, # Lora alaph, see Lora principle for specific functions
lora_dropout=0.1,# Dropout ratio
target_modules= target_modules # Lora layer
)
```

Based on the Lora configuration and base model, get the Lora model to be trained (that is, freeze the non-Lora layer). At the same time, it is necessary to determine whether to continue training at a breakpoint. If so, load the breakpoint information:

```python
# Whether to continue training from a breakpoint
# Source point training
if not finetune_args.continue_training:
# Perform Lora fusion on the base model
model = get_peft_model(model, peft_config)
print("Loading LoRA parameters successfully")
else:
if finetune_args.check_point== None:
print("Breakpoint training requires checkpoint address")
raise ValueError("Breakpoint training requires checkpoint address")
# If you continue training at a breakpoint, directly load the Lora parameters of the breakpoint
model = PeftModel.from_pretrained(model, finetune_args.check_point, is_trainable=True)
print("Load breakpoint from {} successfully".format(finetune_args.check_point))
```

Then load the dataset based on the above definition. In this part, we use try except to catch exceptions:

```python
# Load dataset
try:
# Call the above defined function to generate an iterator
dataset = datasets.Dataset.from_generator(
lambda: read_jsonl(finetune_args.dataset_path, finetune_args.max_seq_length, finetune_args.model_path, finetune_args.skip_overlength)
) 
except Exception as e:
print("Failed to load dataset from {}".format(finetune_args.dataset_path))
print("Error message:")
print(e.__repr__())
raise e 
print("Successful to load dataset from {}".format(finetune_args.dataset_path))
```

Finally, load a custom trainer and start training:

```python
# Load custom trainer
trainer = ModifiedTrainer(
model=model, # Model to be trained
train_dataset=dataset, # Dataset
args=training_args, # Training parameters
data_collator=lambda x : data_collator(x, tokenizer), # Custom sampling function
)

print("Successfully loaded Trainer")
# Perform training
trainer.train()
print("Training completed, training results saved in {}".format(training_args.output_dir))
# Save model
model.save_pretrained(training_args.output_dir)
print("Model parameters saved in {}".format(training_args.output_dir))
```

With the above code, we can complete the Lora fine-tuning of the Atom-7B-Chat model. We encapsulate the above code into a [train.py](../Atom/02-Atom-7B-Chat-Lora/train.py) script, and provide a [bash script](../Atom/02-Atom-7B-Chat-Lora/train.sh) to start training:

```bash
python train.py \
--dataset_path ../../dataset/huanhuan.jsonl \ # Dataset path
--model_path /root/autodl-tmp/data/model/Atom \ # Base model path
--lora_rank 8 \ # lora rank
--per_device_train_batch_size 16 \ # batch_size
--gradient_accumulation_steps 1 \ # Gradient accumulation rounds
--max_steps 120000 \ # Maximum number of training steps, training epoch number = max_steps / (num_whole_data / batch_size)
--save_steps 40000 \ # How many training steps to save parameters
--save_total_limit 3 \ # How many parameters to save at most
--learning_rate 1e-4 \ # Learning rate
--fp16 \ # Use float16 precision
--remove_unused_columns false \ # Whether to remove unused features when processing the data set
--logging_steps 10 \ # How many training steps to output
--output_dir ../../output # Output path
```

Run the script (bash train.sh) directly in the directory to start training.