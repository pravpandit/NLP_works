The jupyter notebook involved in this article is in the [Chapter 4 code base](https://github.com/datawhalechina/learn-nlp-with-transformers/tree/main/docs/%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1).

If you open this jupyter notebook on colab, you need to install 🤗Trasnformers and 🤗datasets. The specific commands are as follows (uncomment and run, if the speed is slow, please switch to the domestic source and add the parameters in the second line).

Before running the cell, it is recommended that you follow the instructions in the readme of this project to set up a dedicated python environment for learning.

```python
#! pip install datasets transformers 
# -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If you are opening this jupyter notebook on your local machine, please make sure your environment has the latest version of the above libraries installed.

You can find it heretps://github.com/huggingface/transformers/blob/master/examples/pytorch/multiple-choice/) to find the specific python script file of this jupyter notebook. You can also use multiple gpus or tpus to fine-tune your model in a distributed way.

# Build a multiple-choice task by fine-tuning the model

In the current jupyter notebook, we will illustrate how to build a multiple-choice task by fine-tuning any [🤗Transformers](https://github.com/huggingface/transformers) model. The task is to choose the most reasonable one among multiple answers given. The dataset we use is [SWAG](https://www.aclweb.org/anthology/D18-1009/), of course, you can also use the preprocessing process for other multiple-choice datasets or your own data. SWAG is a dataset about common sense reasoning. Each sample describes a situation and then gives four possible options.

This Jupyter Notebook can be run on any model in [Model Hub](https://huggingface.co/models) as long as the model has a multi-select header.version. Depending on your model and the GPU you are using, you may need to adjust the batch size to avoid out of memory errors. With these two parameters set, the rest of the jupyter notebook runs smoothly:

```python
model_checkpoint = "bert-base-uncased"
batch_size = 16
```

## Loading the dataset

We will use the [🤗Datasets](https://github.com/huggingface/datasets) library to download the data. This process can be easily done with the function `load_dataset`.

```python
from datasets import load_dataset, load_metric
```

`load_dataset` will cache the dataset to avoid downloading it again next time you run it.

```python
datasets = load_dataset("swag", "regular")
```

Reusing dataset swag (/home/sgugger/.cache/huggingface/datasets/swag/regular/0.0.0/f9784740e0964a3c799d68cec0d992cc267d3fe94f3e048175eca69d739b980d)

In addition, you can also download the data from the [link](https://gas.graviti.cn/dataset/datawhale/SWAG
) we provide and decompress it, copy the decompressed 3 csv files to the `docs/Chapter 4-Using Transformers to solve NLP tasks/datasets/swag` directory, and then load it with the following code.

```python
import os

data_path = './datasets/swag/'
cache_dir = os.path.join(data_path, 'cache')
data_files = {'train': os.path.join(data_path, 'train.csv'), 'val': os.path.join(data_path, 'val.csv'), 'test': os.path.join(data_path, 'test.csv')}
datasets = load_dataset(data_path, 'regular', data_files=data_files, cache_dir=cache_dir)
```

Using custom data configuration regular-2ab2d66f12115abf

Downloading and preparing dataset swag/regular (download: Unknown size, generated: Unknown size, post-processed: Unknown size, total: Unknown size) to ./datasets/swag/cache/swag/regular-2ab2d66f12115abf/0.0.0/a16ae67faa24f4cdd6d1fc6bfc09bdb6dc15771716221ff8bacbc6cc75533614...

Dataset swag downloaded and preparedto ./datasets/swag/cache/swag/regular-2ab2d66f12115abf/0.0.0/a16ae67faa24f4cdd6d1fc6bfc09bdb6dc15771716221ff8bacbc6cc75533614. Subsequent calls will reuse this data.

The `dataset` object itself is a [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict) that contains key-value pairs for training, validation, and test sets (`mnli` is a special case that contains key-value pairs for unmatched validation and test sets).

```python
datasets
```

DatasetDict({
train: Dataset({
features: ['video-id', 'fold-ind', 'startphrase', 'sent1', 'sent2','gold-source', 'ending0', 'ending1', 'ending2', 'ending3', 'label'],
num_rows: 73546
})
validation: Dataset({
features: ['video-id', 'fold-ind', 'startphrase', 'sent1', 'sent2', 'gold-source', 'ending0', 'ending1', 'ending2', 'ending3', 'label'],
num_rows: 20006
})
test: Dataset({
features: ['video-id', 'fold-ind', 'startphrase', 'sent1', 'sent2', 'gold-source', 'ending0', 'ending1', 'ending2', 'ending3', 'label'],num_rows: 20005
})
})

To access an actual element, you need to select a split first, then give an index:

```python
datasets["train"][0]
```

{'ending0': 'passes by walking down the street playing their instruments.',
'ending1': 'has heard approaching them.',
'ending2': "arrives and they're outside dancing and asleep.",
'ending3': 'turns the lead singer watches the performance.',
'fold-ind': '3416',
'gold-source': 'gold',
'label': 0,
'sent1': 'Members of the procession walk down the street holding small horn brass instruments.',
'sent2': 'A drum line',
'startphrase': 'Members of the procession walk down the street holding small horn brass instruments. A drum line',
'video-id': 'anetv_jkn6uvmqwh4'}

To get an idea of ​​what the data looks like, the function below will display some randomly selected examples from the dataset.

```python
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
picks = []
for _ in range(num_examples):
pick = random.randint(0, len(dataset)-1)
while pick in picks:
pick = random.randint(0, len(dataset)-1)
picks.append(pick)

df = pd.DataFrame(dataset[picks])
for column, typ in dataset.features.items():
if isinstance(typ, ClassLabel):
df[column] = df[column].transform(lambda i: typ.names[i])
display(HTML(df.to_html()))
```

```python
show_random_elements(datasets["train"])
```

<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>ending0</th>
<th>ending1</th>
<th>ending2</th>
<th>ending3</th>
<th>fold-ind</th>
<th>gold-source</th>
<th>label</th>
<th>sent1</th>
<th>sent2</th>
<th>startphrase</th>
<th>video-id</th>
</tr>
</thead>
<tbody>
<tr><th>0</th>
<td>are seated on a field.</td>
<td>are skiing down the slope.</td>
<td>are in a lift.</td>
<td>are pouring out in a man.</td>
<td>16668</td>
<td>gold</td>
<td>1</td>
<td>A man is wiping the skiboard.</td>
<td>Group of people</td>
<td>A man is wiping the skiboard. Group of people</td>
<td>anetv_JmL6BiuXr_g</td>
</tr>
<tr>
<th>1</th>
<td>performs stunts inside a gym.</td>
<td>shows severel shopping in the water.</td>
<td>continues his skateboard while talking.</td>
<td>is putting a black bike close.</td>
<td>11424</td>
<td>gold</td>
<td>0</td>
<td>The credits of the video are shown.</td>
<td>A lady</td>
<td>The credits of the video are shown. A lady</td>
<td>anetv_dWyE0o2NetQ</td>
</tr>
<tr>
<th>2</th>
<td>is emerging into the hospital.</td>
<td>are strewn under water at some wreckage.</td>
<td>tosses the wand together and saunters into the marketplace.</td>
<td>swats him upside down.</td>
<td>15023</td>
<td>gen</td>
<td>1</td>
<td>Through his binoculars, someone watches a handful of surfers being rolled up into the wave.</td>
<td>Someone</td>
<td>Through his binoculars, someone watches a handful of surfers being rolled up into the wave. Someone</td>
<td>lsmdc3016_CHASING_MAVERICKS-6791</td>
</tr>
<tr>
<th>3</th>
<td>spies someone sitting below.</td>
<td>opens the fridge and checks out the photo.</td>
<td>puts a little sheepishly.</td>
<td>staggers up to him.</td>
<td>5475</td>
<td>gold</td>
<td>3</td>
<td>He tips it upside down, and its little umbrella falls to the floor.</td>
<td>Back inside, someone</td>
<td>He tips it upside down, and its little umbrella falls to the floor. Back inside, someone</td>
<td>lsmdc1008_Spider-Man2-75503</td>
</tr><tr>
<th>4</th>
<td>carries her to the grave.</td>
<td>laughs as someone styles her hair.</td>
<td>sets down his glass.</td>
<td>stares after her then trudges back up into the street.</td>
<td>6904</td>
<td>gen</td>
<td>1</td>
<td>Someone kisses her smiling daughter on the cheek and beams back at the camera.</td>
<td>Someone</td>
<td>Someone kisses her smiling daughter on the cheek and beams back at the camera. Someone</td><td>lsmdc1028_No_Reservations-83242</td>
</tr>
<tr>
<th>5</th>
<td>stops someone and sweeps all the way back from the lower deck to join them.</td>
<td>is being dragged towards the monstrous animation.</td>
<td>beats out many events at the touch of the sword, crawling it.</td>
<td>reaches into a pocket and yanks open the door.</td>
<td>14089</td>
<td>gen</td>
<td>1</td>
<td>But before he can use his wand, he accidentally rams itup the troll's nostril.</td>
<td>The angry troll</td>
<td>But before he can use his wand, he accidentally rams it up the troll's nostril. The angry troll</td>
<td>lsmdc1053_Harry_Potter_and_the_philosophers_stone-95867</td>
</tr>
<tr>
<th>6</th>
<td>sees someone's name in the photo.</td>
<td>gives a surprised look.</td>
<td>kneels down and touches his ripped specs.</td>
<td>spies on someone's clock.</td>
<td>8407</td>
<td>gen</td>
<td>1</td>
<td>Someone keeps his tired eyes on the road.</td>
<td>Glancing over, he</td>
<td>Someone keeps his tired eyes on the road. Glancing over, he</td>
<td>lsmdc1024_Identity_Thief-82693</td>
</tr>
<tr>
<th>7</th>
<td>stops as someone speaks into the camera.</td>
<td>notices how blue his eyes are.</td>
<td>is flung out of the door and knocks the boy over.</td>
<td>flies through the air, it's a fireball.</td>
<td>4523</td>
<td>gold</td>
<td>1</td>
<td>Both people are knocked back a few steps from the force of the collision.</td>
<td>She</td>
<td>Both people are knocked back a few steps from the force of the collision. She</td>
<td>lsmdc0043_Thelma_and_Luise-68271</td>
</tr>
<tr>
<th>8</th>
<td>sits close to the river.</td>
<td>have pet's supplies and pets.</td>
<td>pops parked outside the dirt facility, sending up a car highway to catch control.</td>
<td>displays all kinds of power tools and website.</td>
<td>8112</td>
<td>gold</td>
<td>1</td>
<td>A guy waits in the waiting room with his pet.</td>
<td>A pet store and its van</td>
<td>A guy waits in the waiting room with his pet. A pet store and its van</td>
<td>anetv_9VWoQpg9wqE</td>
</tr>
<tr>
<th>9</th>
<td>the slender someone, someone turns on the light.</td>
<td>, someone gives them to her boss then dumps some alcohol into dough.</td>
<td>liquids from a bowl, she slams them drunk.</td>
<td>wags his tail as someone returns to the hotel room.</td>
<td>10867</td>
<td>gold</td>
<td>3</td>
<td>Inside a convenience store, she opens a freezer case.</td>
<td>Dolce</td>
<td>Inside a convenience store, she opens a freezer case. Dolce</td>
<td>lsmdc3090_YOUNG_ADULT-43871</td>
</tr>
</tbody>
</table>

Each example in the dataset has a context, which is a combination of the first sentence (field `sent1`) and the second sentence’sThe sentence consists of four possible endings (fields `ending0`, `ending1`, `ending2` and `ending3`). Then, given four possible endings (fields `ending0`, `ending1`, `ending2` and `ending3`), the model is asked to choose the correct one (indicated by the field `label`). The following function gives us a more intuitive example:

```python
def show_one(example):
print(f"Context: {example['sent1']}")
print(f" A - {example['sent2']} {example['ending0']}")
print(f" B - {example['sent2']} {example['ending1']}")
print(f" C - {example['sent2']} {example['ending2']}")
print(f" D - {example['sent2']} {example['ending3']}")
print(f"\nGround truth: option {['A', 'B', 'C', 'D'][example['label']]}")
```

```python
show_one(datasets["train"][0])
```

Context: Members of the procession walk down the street holding small horn brass instruments.
A - A drum line passes by walking down the street playing their instruments.
B - A drum line has heard approaching them.
C - A drum line arrives and they're outside dancing and asleep.
D - A drum line turns the lead singer watches the performance.

Ground truth: option A

```python
show_one(datasets["train"][15])
```

Context: Now it's someone's turn to rain blades on his opponent.
A - Someone pats his shoulder and spins wildly.
B - Someone lunges forward through the window.
C - Someone falls to the ground.
D - Someone rolls up his fast run from the water and tosses in the sky.

Ground truth: option C

## Data preprocessing

Before feeding these texts into the model, we need to preprocess them. This is done by the `Tokenizer` of the 🤗transformer, which, as its name suggests, represents the input as a series of tokens and then converts them to their corresponding ids by looking up a pretrained vocabulary. Finally, it is converted into the format expected by the model, while generating the other inputs required by the model.To do all this, we instantiate our tokenizer using the `from_pretrained` method of `AutoTokenizer`, which will ensure that:

- we get a tokenizer that corresponds to the model architecture we want to use,
- we download the vocabulary that was used to pretrain this particular model.

Also, the vocabulary is cached so it is not downloaded again the next time we run it.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
```

We pass `use_fast=True` as an argument to use one of the fast tokenizers from the 🤗tokenizers library (which is powered by Rust). These fast tokenizers work for almost all models, but if you get an error in the previous call, remove the argument.

You can call this tokenizer directly on a sentence or sentence pair:

```python
tokenizer("Hello, this one sentence!", "And this sentence goes with it.")
```

{'input_ids': [101, 7592, 1010, 2023, 2028, 6251, 999, 102, 1998, 2023, 6251, 3632, 2007, 2009, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

Depending on the model you chose, you will see different key-value pairs in the dictionary returned by the cell above. They are not important for what we do here, just know that they are needed for the model we instantiate later. If you are interested, you can learn more about them in [this tutorial](https://huggingface.co/transformers/preprocessing.html).

As shown in the dictionary below, in order to preprocess the dataset, we need to know the name of the column containing the sentences:

We can write a function to preprocess our samples. The tricky part is putting all the possible sentence pairs in two big lists before calling the tokenizer, and then flattening the results so that each example has four input ids, attention masks, etc.

When calling `tokenizer`, we pass in the parameter `truncation=True`. This will ensure that inputs longer than the selected model can handle will be truncated to the maximum length the model can accept.

```python
ending_names = ["ending0", "ending1", "ending2", "ending3"]

def preprocess_function(examples):
# Repeat each first sentence four times to go with the four possibilities of second sentences.
first_sentences = [[context] * 4 for context in examples["sent1"]]
# Grab all second sentences possible for each context.
question_headers = examples["sent2"]
second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]

# Flatten everything
first_sentences = sum(first_sentences, [])
second_sentences = sum(second_sentences, [])

# Tokenize
tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
# Un-flatten
return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
```

This function works with one or several examples. In the case of several examples, the tokenizer will return a list of lists of lists for each key: a list of all examples (here 5), then a list of all choices (4) and a list of input IDs (length varying here since we did not apply any padding):

This function works with one or several examples. In the case of several examples, the tokenizer will return a list of lists of lists for each key: a list of all examples (here 5), then a list of all choices (4) and a list of input IDs (length varying here since we did not apply any padding):

```python
examples = datasets["train"][:5]
features = preprocess_function(examples)
print(len(features["input_ids"]), len(features["input_ids"][0]), [len(x) for x in features["input_ids"][0]])
```

5 4 [30, 25, 30, 28]

Let's decode the input for the given example:

```python
idx = 3
[tokenizer.decode(features["input_ids"][idx][i]) for i in range(4)]
```

['[CLS] a drum line passes by walking down the street playing their instruments. [SEP] members of the procession are playing ping pong and celebrating one left each in quick. [SEP]',
'[CLS] a drum line passes by walking down the street playingtheir instruments. [SEP] members of the procession wait slowly towards the cadets. [SEP]',
'[CLS] a drum line passes by walking down the street playing their instruments. [SEP] members of the procession makes a square call and ends by jumping down into snowy streets where fans begin to take their positions. [SEP]',
'[CLS] a drum line passes by walking down the street playing their instruments. [SEP] members of the procession play and go back and forth hitting the drums while the audiencece claps for them. [SEP]']

We can compare it to the ground truth generated earlier:

```python
show_one(datasets["train"][3])
```

Context: A drum line passes by walking down the street playing their instruments.
A - Members of the procession are playing ping pong and celebrating one left each in quick.
B - Members of the procession wait slowly towards the cadets.
C - Members of the procession makes a square call and ends by jumping down into snowy streets where fans begin to take their positions.
D - Members of the procession play and go back and forth hitting the drums while the audience claps for them.

Ground truth: option D

This seems to work fine. We can apply this function to all examples in our dataset by using the `map` method of the `dataset` object we created earlier. This will be applied to all elements of all splits of the `dataset` object, so our training, validation, and test data will be preprocessed in the same way.

```python
encoded_datasets = datasets.map(preprocess_function, batched=True)
```

Loading cached processed dataset at /home/sgugger/.cache/huggingface/datasets/swag/regular/0.0.0/f9784740e0964a3c799d68cec0d992cc267d3fe94f3e048175eca69d739b980d/cache-975c81cf12e5b7ac.arrow
Loading cached processed dataset at /home/sgugger/.cache/huggingface/datasets/swag/regular/0.0.0/f9784740e0964a3c799d68cec0d992cc267d3fe94f3e048175eca69d739b980d/cache-d4806d63f1eaf5cd.arrow
Loading cached processed dataset at /home/sgugger/.cache/huggingface/datasets/swag/regular/0.0.0/f9784740e0964a3c799d68cec0d992cc267d3fe94f3e048175eca69d739b980d/cache-258c9cd71b0182db.arrow

Even better, the results will be automatically cached by the 🤗Datasets library to avoid spending time on this step next time you run it. 🤗Datasets libraries are usually smart enough, which detects when the function passed to `map` changes (no longer using cached data). For example, it will detect if you change the task in the first cell and rerun the notebook. It warns you when 🤗Datasets uses cache files, and you can pass `load_from_cache_file=False` in the call to `map` to not use the cache file and force preprocessing.

Note that we passed `batched=True` to encode the text in batches. This is to take advantage of the fast tokenizer we loaded earlier, which will process the text in batches concurrently using multiple threads.

## Fine-tune the model

Now that our data is ready, we can download the pre-trained model and fine-tune it. Since our task is about multiple choices, we use the `AutoModelForMultipleChoice` class. As with the tokenizer, the `from_pretrained` method will download and cache the model for us.

```python
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
```

Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMultipleChoice: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']
- This IS expected if you are initializingzing BertForMultipleChoice from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForMultipleChoice from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of BertForMultipleChoice were not initializedlized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

This warning tells us that we are discarding some weights (`vocab_transform` and `vocab_layer_norm` layers) and randomly initializing some other parameters (`pre_classifier` and `classifier` layers). This is completely normal, because we discarded the head used for masked language modeling when pre-training the model and replaced it with a new multi-select head, and we don't have its pre-trained weights, so this warning tells us that we need to fine-tune before using this model for inference, which is exactly what we are going to do.

In order to instantiate a `Trainer`, we need to define three more things. The most important one is [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments), which is a class that contains all the attributes used for training. It requires a folder name to be passed in to save the model checkpoints, and all other parameters are optional:

```python
args = TrainingArguments(
"test-glue",
evaluation_strategy = "epoch",
learning_rate=5e-5,
per_device_train_batch_size=batch_size,
per_device_eval_batch_size=batch_size,
num_train_epochs=3,
weight_decay=0.01,
)
```

Here, we set evaluation at the end of each epoch, adjust the learning rate, use the `batch_size` defined at the top of the jupyter notebook, and customize the number of epochs used for training, as well as weight decay.

Then, we need to tell meHow our `Trainer` constructs batches of data from the preprocessed input data. We haven't done any padding yet, because we will pad each batch to a maximum length within the batch (instead of using the maximum length of the entire dataset). This will be the job of the *data collator*. It takes a list of examples and converts them into a batch (in our case, by applying padding). Since there is no data collator in the library that handles our specific problem, here we adapt one based on `DataCollatorWithPadding`:

```python
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

@dataclass
class DataCollatorForMultipleChoice:
"""
Data collator that will dynamically pad the inputsfor multiple choice received.
"""

tokenizer: PreTrainedTokenizerBase
padding: Union[bool, str, PaddingStrategy] = True
max_length: Optional[int] = None
pad_to_multiple_of: Optional[int] = None

def __call__(self, features):
label_name = "label" if "label" in features[0].keys() else "labels"
labels = [feature.pop(label_name) for feature in features]
batch_size = len(features)
num_choices = len(features[0]["input_ids"])
flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
flattened_features = sum(flattened_features, [])

batch = self.tokenizer.pad(
flattened_features,
padding=self.padding,
max_length=self.max_length,
pad_to_multiple_of=self.pad_to_multiple_of,
return_tensors="pt",
)

# Un-flatten
batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
# Add back labels
batch["labels"] = torch.tensor(labels, dtype=torch.int64)
return batch
```

When passed a list of examples, it flattens all the inputs/attention masks etc in the large list and passes it to the `tokenizer.pad` method. This returns a dictionary with large tensors (of size `(batch_size * 4) x seq_length`), which we then flatten.

We can check that the data collator is working properly on the list of features, here we just need to make sure to remove any input features that are not accepted by our model (this is what `Trainer` does automatically for us):

```python
accepted_keys = ["input_ids", "attention_mask", "label"]
features = [{k: v for k, v in encoded_datasets["train"][i].items() if k in accepted_keys} for i in range(10)]
batch = DataCollatorForMultipleChoice(tokenizer)(features)
```

Again, all of these flattened and unflattened values ​​could be sources of potential errors, so let’s do another sanity check on the input:

```python
[tokenizer.decode(batch["input_ids"][8][i].tolist()) for i in range(4)]
```

['[CLS] someone walks over to the radio. [SEP] someone hands her another phone. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]',
'[CLS] someone walks over to theradio. [SEP] someone takes the drink, then holds it. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]',
'[CLS] someone walks over to the radio. [SEP] someone looks off then looks at someone. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]',
'[CLS] someone walks over to the radio. [SEP] someone stares blearily downat the floor. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]']

```python
show_one(datasets["train"][8])
```

Context: Someone walks over to the radio.
A - Someone hands her another phone.
B - Someone takes the drink, then holds it.
C - Someone looks off then looks at someone.
D - Someone stares blearily down at the floor.

Ground truth: option D

All working!

Finally, we need to define how `Trainer` should calculate the expected value based on the expected value.We need to define a function that will use the `metric` we loaded earlier, and the only preprocessing we have to do is take the argmax of our predicted logits:

```python
import numpy as np

def compute_metrics(eval_predictions):
predictions, label_ids = eval_predictions
preds = np.argmax(predictions, axis=1)
return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}
```

Then, we just need to pass all of this into `Trainer` along with our dataset:

```python
trainer = Trainer(
model,
args,
train_dataset=encoded_datasets["train"],
eval_dataset=encoded_datasets["validation"],tokenizer=tokenizer,
data_collator=DataCollatorForMultipleChoice(tokenizer),
compute_metrics=compute_metrics,
)
```

Now, we can fine-tune the model by calling the `train` method:

```python
trainer.train()
```

<div>
<style>
/* Turns off some styling */
progress {
/* gets rid of default border in Firefox and Opera. */
border: none;
/* Needs to be in here for Safari polyfill so background images work as expected. */
background-size: auto;
}
</style>

<progress value='6897' max='6897' style='width:300px; height:20px; vertical-align: middle;'></progress>
[6897/6897 23:49, Epoch 3/3]
</div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: left;">
<th>Epoch</th>
<th>Training Loss</th>
<th>Validation Loss</th>
<th>Accuracy</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>0.154598</td>
<td>0.828017</td>
<td>0.766520</td>
</tr>
<tr>
<td>2</td>
<td>0.296633</td>
<td>0.667454</td>
<td>0.786814</td>
</tr>
<tr>
<td>3</td>
<td>0.111786</td>
<td>0.994927</td>
<td>0.789363</td>
</tr>
</tbody>
</table><p>

TrainOutput(global_step=6897, training_loss=0.19714653808275168)

Finally, don’t forget to [upload](https://huggingface.co/transformers/model_sharing.html) your model to [🤗 Model Center](https://huggingface.co/models).