The jupter notebook involved in this article is in the [Chapter 4 code base](https://github.com/datawhalechina/learn-nlp-with-transformers/tree/main/docs/%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1).

It is recommended to open this tutorial directly using google colab notebook to quickly download relevant datasets and models.
If you are opening this notebook in google colab, you may need to install the Transformers and 🤗Datasets libraries. Uncomment the following commands to install.

```python
! pip install datasets transformers "sacrebleu>=1.4.12,<2.0.0" sentencepiece
```

If you are opening this notebook locally, please make sure you have carefully read and installed all the dependencies in the transformer-quick-start-zh readme file. You can alsoFind a multi-GPU distributed training version of this notebook [here](https://github.com/huggingface/transformers/tree/master/examples/seq2seq).

# Fine-tuning transformer models for translation tasks

In this notebook, we will show how to use models from the [🤗 Transformers](https://github.com/huggingface/transformers) repository to solve translation tasks in natural language processing. We will use the [WMT dataset](http://www.statmt.org/wmt16/) dataset. This is one of the most commonly used datasets for translation tasks.

An example is shown below:

![Widget inference on a translation task](https://github.com/huggingface/notebooks/blob/master/examples/images/translation.png?raw=1)

For the translation task, we will show how to use a simple dataset loading and also for the correspondingFine-tune the model using the Trainer interface in R.

```python
model_checkpoint = "Helsinki-NLP/opus-mt-en-ro" 
# Select a model checkpoint
```

As long as the pre-trained transformer model contains a head layer of the seq2seq structure, this notebook can theoretically use a variety of transformer models [model panel](https://huggingface.co/models) to solve any translation task.

In this article, we use the pre-trained [`Helsinki-NLP/opus-mt-en-ro`](https://huggingface.co/Helsinki-NLP/opus-mt-en-ro) checkpoint for translation tasks. 

## Loading data

We will use the 🤗 Datasets library to load data and corresponding evaluation methods. Data loading and evaluation method loading only require simple use of load_dataset and load_metric. We use the English/Romanian bilingual translation in the WMT dataset.

```python
from datasets import load_dataset, load_metric

raw_datasets = load_dataset("wmt16", "ro-en")
metric = load_metric("sacrebleu")
```

Downloading: 2.81kB [00:00, 523kB/s] 
Downloading: 3.19kB [00:00, 758kB/s] 
Downloading: 41.0kB [00:00, 11.0MB/s] 

Downloading and preparing dataset wmt16/ro-en (download: Unknown size, generated: Unknown size, post-processed: Unknown size, total: Unknown size) to /Users/niepig/.cache/huggingface/datasets/wmt16/ro-en/1.0.0/0d9fb3e814712c785176ad8cdb9f465fbe6479000ee6546725db30ad8a8b5f8a...

Downloading: 100%|██████████| 225M/225M [00:18<00:00, 12.2MB/s]
Downloading: 100%|██████████| 23.5M/23.5M [00:16<00:00, 1.44MB/s]
Downloading: 100%|██████████| 38.7M/38.7M [00:03<00:00, 9.82MB/s]

Dataset wmt16 downloaded and prepared to /Users/niepig/.cache/huggingface/datasets/wmt16/ro-en/1.0.0/0d9fb3e814712c785176ad8cdb9f465fbe6479000ee6546725db30ad8a8b5f8a. Subsequent calls will reuse this data.Downloading: 5.40kB [00:00, 2.08MB/s] 

This datasets object itself is a [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict) data structure. For the training set, validation set, and test set, you only need to use the corresponding key (train, validation, test) to get the corresponding data.

```python
raw_datasets
```

DatasetDict({
train: Dataset({
features: ['translation'],
num_rows: 610320
})
validation: Dataset({
features: ['translation'],
num_rows: 1999})
test: Dataset({
features: ['translation'],
num_rows: 1999
})
})

Given a data segmentation key (train, validation or test) and a subscript, you can view the data.

```python
raw_datasets["train"][0]
# We can see that one English sentence en corresponds to one Romanian sentence ro
```

{'translation': {'en': 'Membership of Parliament: see Minutes',
'ro': 'Componenţa Parlamentului: a se vedea procesul-verbal'}}

To further understand what the data looks like, the following function will randomly select a few examples from the dataset for display.

```python
import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=5):
assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
picks = []
for _ in range(num_examples):
pick = random.randint(0, len(dataset)-1)
while pick in picks:
pick = random.randint(0, len(dataset)-1)
picks.append(pick)

df = pd.DataFrame(dataset[picks])
for column, typ in dataset.features.items():
if isinstance(typ,datasets.ClassLabel):
df[column] = df[column].transform(lambda i: typ.names[i])
display(HTML(df.to_html()))
```

```python
show_random_elements(raw_datasets["train"])
```

<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>translation</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>{'en': 'I do not believe that this is the right course.', 'ro': 'Nu cred că acesta este varianta corectă.'}</td></tr>
<tr>
<th>1</th>
<td>{'en': 'A total of 104 new jobs were created at the European Chemicals Agency, which mainly supervises our REACH projects.', 'ro': 'A total of 104 new jobs were created at the European Chemicals Agency, which mainly supervises our REACH projects.', 'ro': 'Un total de 104 noi locuri de manuncă au fost create la Agențiaă pentru Produse Chimice, care, în special, supraveghează proiectele noastre REACH.'}</td>
</tr>
<tr>
<th>2</th>
<td>{'en': 'In view of the above, will the Council say what stage discussions for Turkish participation in joint Frontex operations have reached?', 'ro': 'Will the Turkish government be able to provide referrals to members of the Turkish Opera's municipality in Frontex?'}</td>
</tr>
<tr>
<th>3</th>
<td>{'en': 'We now fear that if the scope of this directive is expanded, the directive will suffer exactly the same fate as the last attempt at introducing 'Made in' origin marking - in other words, that it will once again be blocked by the Council.', 'ro': 'We are not sure, but we will be able to apply for a directiveSince we have already seen the exact same results, we have also introduced a marcajului origin "Made in”, which is also a good example, we can also see the map of the country.'}</td>
</tr>
<tr>
<th>4</th>
<td>{'en': 'The country dropped nine slots to 85th, with a score of 6.58.', 'ro': 'The country has a low score of 85, with a score of 6.58.'}</td>
</tr>
</tbody>
</table>

The metric is [`datasets.Metric`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Metric) class, see the metric and usage examples:

```python
metric
```

Metric(name: "sacrebleu", features: {'predictions': Value(dtype='string', id='sequence'), 'references': Sequence(feature=Value(dtype='string', id='sequence'), length=-1, id='references')}, usage: """
Produces BLEU scores along with its sufficient statistics
from a source against one or more references.

Args:
predictions: The system stream (a sequence of segments)
references: A list of one or more reference streams (each a sequence of segments)
smooth: The smoothing method to use
smooth_value: For 'floor' smoothing, the floor to use
force: Ignore data that looks already tokenized
lowercase: Lowercase the data
tokenize: The tokenizer to use
Returns:
'score': BLEU score,
'counts': Counts,
'totals': Totals,
'precisions': Precisions,
'bp': Brevity penalty,
'sys_len': predictions length,
'ref_len': reference length,
Examples:

>>> predictions = ["hello there general kenobi", "foo bar foobar"]
>>> references = [["hello there general kenobi", "hello there !"], ["foo bar foobar", "foo bar foobar"]]
>>> sacrebleu = datasets.load_metric("sacrebleu")
>>> results = sacrebleu.compute(predictions=predictions, references=references)
>>> print(list(results.keys()))
['score', 'counts', 'totals', 'precisions', 'bp', 'sys_len','ref_len']
>>> print(round(results["score"], 1))
100.0
""", stored examples: 0)

We use the `compute` method to compare predictions and labels to calculate the score. Both predictions and labels need to be a list. See the example below for the specific format:

```python
fake_preds = ["hello there", "general kenobi"]
fake_labels = [["hello there"], ["general kenobi"]]
metric.compute(predictions=fake_preds, references=fake_labels)
```

{'score': 0.0,
'counts': [4, 2, 0, 0],
'totals': [4, 2, 0, 0],
'precisions': [100.0, 100.0, 0.0, 0.0],
'bp': 1.0,
'sys_len': 4,
'ref_len': 4}

## Data preprocessing

Before feeding the data into the model, we need to preprocess the data. The preprocessing tool is called Tokenizer. Tokenizer first tokenizes the input, then converts the tokens into the corresponding token ID required in the pre-model, and then converts them into the input format required by the model.

In order to achieve the purpose of data preprocessing, we use the AutoTokenizer.from_pretrained method to instantiate our tokenizer, which ensures:

- We get a tokenizer that corresponds to the pre-trained model one by one.
- When using the tokenizer corresponding to the specified model checkpoint, we also download the vocabulary required by the model, more precisely, the tokens vocabulary.

This downloaded tokens vocabulary will be cached so that it will not be downloaded again when used again.

```python
from transformers import AutoTokenizer
# Need to install `sentencepiece`: pip install sentencepiece

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

Downloading: 100%|██████████| 1.13k/1.13k [00:00<00:00, 466kB/s]
Downloading: 100%|██████████| 789k/789k [00:00<00:00, 882kB/s]
Downloading: 100%|██████████| 817k/817k [00:00<00:00, 902kB/s]
Downloading: 100%|██████████| 1.39M/1.39M [00:01<00:00, 1.24MB/s]
Downloading: 100%|██████████| 42.0/42.0 [00:00<00:00, 14.6kB/s]

Take the mBART model we use as an example, we need to correctly set the source language and target language. If you want to translate other bilingual corpora, please check [here](https://huggingface.co/facebook/mbart-large-cc25). We can check the settings of source and target languages:

```python
if "mbart" in model_checkpoint:
tokenizer.src_lang = "en-XX"
tokenizer.tgt_lang = "ro-RO"
```

Tokenizer can preprocess a single text or a pair of texts. The data obtained after tokenizer preprocessing meets the input format of pre-trained models

```python
tokenizer("Hello, this one sentence!")
```

{'input_ids': [125, 778, 3, 63, 141, 9191, 23, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}

The token IDs seen above, that is, input_ids, generally vary with the name of the pre-trained model. The reason is that different pre-trained models set different rules during pre-training.l, then the input format of the tokenizer preprocessing will meet the model requirements. For more information about preprocessing, please refer to [this tutorial](https://huggingface.co/transformers/preprocessing.html)

In addition to tokenizing a sentence, we can also tokenize a list of sentences.

```python
tokenizer(["Hello, this one sentence!", "This is another sentence."])
```

{'input_ids': [[125, 778, 3, 63, 141, 9191, 23, 0], [187, 32, 716, 9191, 2, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}

Note: In order to prepare the translation targets for the model, we use `as_target_tokenizer` to control the special tokens corresponding to the targets:

```python
with tokenizer.as_target_tokenizer():
print(tokenizer("Hello, this one sentence!"))
model_input = tokenizer("Hello, this one sentence!")
tokens = tokenizer.convert_ids_to_tokens(model_input['input_ids'])
# Print and see the special token
print('tokens: {}'.format(tokens))
```

{'input_ids': [10334, 1204, 3, 15, 8915, 27, 452, 59, 29579, 581, 23, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
tokens: [' Hel', 'lo', ',', ' ', 'this', ' o', 'ne', ' se', 'nten', 'ce', '!', '</s>']

If you are using T5The checkpoints of the pre-trained model need to check for special prefixes. T5 uses special prefixes to tell the model what to do. The specific prefix examples are as follows:

```python
if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
prefix = "translate English to Romanian: "
else:
prefix = ""
```

Now we can put everything together to form our preprocessing function. When we preprocess the samples, we will also use the parameter `truncation=True` to ensure that our long sentences are truncated. By default, we automatically pad for shorter sentences.

```python
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "ro"

def preprocess_function(examples):
inputs = [prefix + ex[source]]e_lang] for ex in examples["translation"]]
targets = [ex[target_lang] for ex in examples["translation"]]
model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

# Setup the tokenizer for targets
with tokenizer.as_target_tokenizer():
labels = tokenizer(targets, max_length=max_target_length, truncation=True)

model_inputs["labels"] = labels["input_ids"]
return model_inputs
```

The above preprocessing function can process one sample or multiple sample examples. If it processes multiple samples, the result after multiple samples are preprocessed is returned.st.

```python
preprocess_function(raw_datasets['train'][:2])
```

{'input_ids': [[393, 4462, 14, 1137, 53, 216, 28636, 0], [24385, 14, 28636, 14, 4646, 4622, 53, 216, 28636, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'labels': [[42140, 494, 1750, 53, 8, 59, 903, 3543, 9, 15202, 0], [36199, 6612, 9, 15202, 122, 568, 35788, 21549, 53, 8, 59, 903, 3543, 9, 15202, 0]]}

Next, all samples in the datasets are preprocessed by using the map function to apply the preprocessing function prepare_train_features to all (map)On the sample.

```python
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
```

100%|██████████| 611/611 [02:32<00:00, 3.99ba/s]
100%|██████████| 2/2 [00:00<00:00, 3.76ba/s]
100%|██████████| 2/2 [00:00<00:00, 3.89ba/s]

Better yet, the returned results are automatically cached to avoid recalculation the next time they are processed (but be aware that if the input has changed, it may be affected by the cache!). The datasets library function will detect the input parameters to determine whether there is a change. If there is no change, the cached data will be used. If there is a change, it will be reprocessed. But if the input parameters remain unchanged, it is best to clean up the cache when you want to change the input. The way to clean up is to use the `load_from_cache_file=False` parameter. In addition, the `batched=True` parameter used above is a feature of the tokenizer, because it will use multiple threads to process the input in parallel.

## MicroAdjust the transformer model

Now that the data is ready, we need to download and load our pre-trained model, and then fine-tune the pre-trained model. Since we are doing seq2seq tasks, we need a model class that can solve this task. We use the class `AutoModelForSeq2SeqLM`. Similar to tokenizer, the `from_pretrained` method can also help us download and load the model, and it will also cache the model so that we don’t download the model repeatedly.

```python
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

Downloading: 100%|██████████| 301M/301M [00:19<00:00, 15.1MB/s]

Since our fine-tuning task is machine translation, and we load a pre-trained seq2seq model, we do notIt will prompt us that some unmatched neural network parameters were thrown away when loading the model (for example, the neural network head of the pre-trained language model was thrown away, and the neural network head of the machine translation was randomly initialized).

In order to get a `Seq2SeqTrainer` training tool, we need 3 elements, the most important of which is the training settings/parameters [`Seq2SeqTrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Seq2SeqTrainingArguments). This training setting contains all the properties that can define the training process

```python
batch_size = 16
args = Seq2SeqTrainingArguments(
"test-translation",
evaluation_strategy = "epoch",
learning_rate=2e-5,
per_device_train_batch_size=batch_size,
per_device_eval_batch_size=batch_size,
weight_decay=0.01,
save_total_limit=3,
num_train_epochs=1,
predict_with_generate=True,
fp16=False,
)
```

The evaluation_strategy = "epoch" parameter above tells the training code that we will do a validation evaluation once per epoch.

The batch_size is defined before this notebook.

Since our dataset is large and `Seq2SeqTrainer` will keep saving models, we need to tell it to save at most `save_total_limit=3` models.

Finally, we need a data collator data to feed our processed input to the model.

```python
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

The last thing left after setting up `Seq2SeqTrainer` is that we need to define the evaluation method. We use `metric` to complete the evaluation. Send the model prediction toBefore we evaluate, we will also do some post-processing of the data:

```python
import numpy as np

def postprocess_text(preds, labels):
preds = [pred.strip() for pred in preds]
labels = [[label.strip()] for label in labels]

return preds, labels

def compute_metrics(eval_preds):
preds, labels = eval_preds
if isinstance(preds, tuple):
preds = preds[0]
decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

# Replace -100 in the labels as we can't decode them.
labels = np.where(labels !=-100, labels, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

# Some simple post-processing
decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

result = metric.compute(predictions=decoded_preds, references=decoded_labels)
result = {"bleu": result["score"]}

prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
result["gen_len"] = np.mean(prediction_lens)result = {k: round(v, 4) for k, v in result.items()}
return result
```

Finally, pass all parameters/data/models to `Seq2SeqTrainer`

```python
trainer = Seq2SeqTrainer(
model,
args,
train_dataset=tokenized_datasets["train"],
eval_dataset=tokenized_datasets["validation"],
data_collator=data_collator,
tokenizer=tokenizer,
compute_metrics=compute_metrics
)
```

Call the `train` method for fine-tuning training.

```python
trainer.train()
```

Finally, don't forget to check how to upload the model and upload the model to](https://huggingface.co/transformers/model_sharing.html) Go to [🤗 Model Hub](https://huggingface.co/models). Then you can use your model directly by using the model name, just like at the beginning of this notebook.

```python

```