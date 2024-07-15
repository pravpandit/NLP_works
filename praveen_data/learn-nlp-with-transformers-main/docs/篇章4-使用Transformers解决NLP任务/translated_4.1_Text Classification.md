The jupter notebook involved in this article is in the [Chapter 4 code base](https://github.com/datawhalechina/learn-nlp-with-transformers/tree/main/docs/%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1).

You can also directly use google colab notebook to open this tutorial and download the relevant datasets and models.
If you are opening this notebook in google colab, you may need to install the Transformers and 🤗Datasets libraries. Uncomment the following commands to install them.

```python
!pip install transformers datasets
```

If you are opening this notebook locally, make sure you have installed the above dependencies.
You can also find examples herecation) to find the multi-GPU distributed training version of this notebook.

# Fine-tune the pre-trained model for text classification

We will show how to use the model in the [🤗 Transformers](https://github.com/huggingface/transformers) codebase to solve the text classification task from the [GLUE Benchmark](https://gluebenchmark.com/).

![Widget inference on a text classification task](https://github.com/huggingface/notebooks/blob/master/examples/images/text_classification.png?raw=1)

The GLUE list includes 9 sentence-level classification tasks, namely:
- [CoLA](https://nyu-mll.github.io/CoLA/) (Corpus of Linguistic Acceptability) identifies whether a sentence is grammatically correct.
- [MNLI](https://arxiv.org/abs/1704.05426) (Multi-Genre Natural Language Inference) Given a hypothesis, determine the relationship between another sentence and the hypothesis: entails, contradicts or unrelated.
- [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) (Microsoft Research Paraphrase Corpus) Determine whether two sentences are paraphrases of each other.
- [QNLI](https://rajpurkar.github.io/SQuAD-explorer/) (Question-answering Natural Language Inference) Determine whether the second sentence contains the answer to the question in the first sentence.
- [QQP](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) (Quora Question Pairs2) Determine whether two questions are semantically identical.
- [RTE](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment) (Recognizing Textual Entailment) Determines whether a sentence is in an entail relationship with a hypothesis.
- [SST-2](https://nlp.stanford.edu/sentiment/index.html) (Stanford Sentiment Treebank) Determines whether a sentence is positive or negative in sentiment.
- [STS-B](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) (Semantic Textual Similarity Benchmark) Determines the similarity between two sentences (scores are 1-5 points).
- [WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html) (Winograd Natural Language Inference) Determine if a sentence with an anonymous pronoun and a sentence withthis pronoun replaced are entailed or not. 

For the above tasks, we will show how to load the dataset using the simple Dataset library and fine-tune the pre-trained model using the `Trainer` interface in transformer.

```python
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
```

This notebook can theoretically use a variety of transformer models ([models panel](https://huggingface.co/models)) to solve any text classification task.

If the task you are dealing with is different, it is likely that only minor changes are needed to use this notebook to handle it. At the same time, you should adjust the btach size required for fine-tuning training according to your GPU video memory to avoid video memory overflow.

```python
task = "cola"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
```

## Loading data

We will use the [🤗 Datasets](https://github.com/huggingface/datasets) library to load data and corresponding evaluation methods. Data loading and evaluation method loading only need to use `load_dataset` and `load_metric`.

```python
from datasets import load_dataset, load_metric
```

Except for `mnli-mm`, ​​other tasks can be loaded directly by task name. Data will be automatically cached after loading.

```python
actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)
```

This `datasets` object itself is a [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict) data structure. For the training set, validation set and test set, you only need to use the corresponding key (train, validation, test) to get the corresponding data.

```python
dataset
```

DatasetDict({
train: Dataset({
features: ['sentence', 'label', 'idx'],
num_rows: 8551
})
validation: Dataset({
features: ['sentence', 'label', 'idx'],
num_rows: 1043
})
test: Dataset({
features: ['sentence', 'label', 'idx'],
num_rows: 1063
})
})

Given a data splitThe key (train, validation, or test) and the subscript can be used to view the data.

```python
dataset["train"][0]
```

{'idx': 0,
'label': 1,
'sentence': "Our friends won't buy this analysis, let alone the next one we propose."}

To further understand what the data looks like, the following function will randomly select a few examples from the dataset to display.

```python
import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
assert num_examples <= len(dataset), "Can't pick more elements than there are in the datasett."
picks = []
for _ in range(num_examples):
pick = random.randint(0, len(dataset)-1)
while pick in picks:
pick = random.randint(0, len(dataset)-1)
picks.append(pick)

df = pd.DataFrame(dataset[picks])
for column, typ in dataset.features.items():
if isinstance(typ, datasets.ClassLabel):
df[column] = df[column].transform(lambda i: typ.names[i])
display(HTML(df.to_html()))
```

```python
show_random_elements(dataset["train"])
```

<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>sentence</th>
<th>label</th>
<th>idx</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>The more I talk to Joe, the less about linguistics I am inclined to think Sally has taught him to appreciate.</td>
<td>acceptable</td>
<td>196</td>
</tr>
<tr>
<th>1</th>
<td>Have the kids arrived safely in our class?</td><td>unacceptable</td>
<td>3748</td>
</tr>
<tr>
<th>2</th>
<td>I gave Mary a book.</td>
<td>acceptable</td>
<td>5302</td>
</tr>
<tr>
<th>3</th>
<td>Every student, who attended the party, had a good time.</td>
<td>unacceptable</td>
<td>4944</td>
</tr>
<tr>
<th>4</th>
<td>Bill pounded the metal fiat.</td>
<td>acceptable</td>
<td>2178</td>
</tr>
<tr>
<th>5</th>
<td>It bit meon the leg.</td>
<td>acceptable</td>
<td>5908</td>
</tr>
<tr>
<th>6</th>
<td>The boys were made a good mother by Aunt Mary.</td>
<td>unacceptable</td>
<td>736</td>
</tr>
<tr>
<th>7</th>
<td>More of a man is here.</td>
<td>unacceptable</td>
<td>5403</td>
</tr>
<tr>
<th>8</th>
<td>My mother baked me a birthday cake.</td>
<td>acceptable</td>
<td>3761</td>
</tr>
<tr>
<th>9</th><td>Gregory appears to have wanted to be loyal to the company.</td>
<td>acceptable</td>
<td>4334</td>
</tr>
</tbody>
</table>

The evaluation metric is an instance of [`datasets.Metric`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Metric):

```python
metric
```

Metric(name: "glue", features: {'predictions': Value(dtype='int64', id=None), 'references': Value(dtype='int64', id=None)}, usage: """
Compute GLUE evaluation metric associated to each GLUEdataset.
Args:
predictions: list of predictions to score.
Each translation should be tokenized into a list of tokens.
references: list of lists of references for each translation.
Each reference should be tokenized into a list of tokens.
Returns: depending on the GLUE subset, one or several of:
"accuracy": Accuracy
"f1": F1 score
"pearson": Pearson Correlation
"spearmanr": Spearman Correlation
"matthews_correlation": Matthew Correlation
Examples:

>>> glue_metric = datasets.load_metric('glue', 'sst2') # 'sst2' or any of ["mnli", "mnli_mismatched", "mnli_matched", "qnli", "rte", "wnli", "hans"]
>>> references = [0, 1]
>>> predictions = [0, 1]
>>> results = glue_metric.compute(predictions=predictions, references=references)
>>> print(results)
{'accuracy': 1.0}

>>> glue_metric = datasets.load_metric('glue', 'mrpc') # 'mrpc' or 'qqp'
>>> references = [0, 1]
>>> predictions = [0, 1]
>>> results = glue_metric.compute(predictions=predictions, references=references)
>>> print(results)
{'accuracy': 1.0, 'f1': 1.0}

>>> glue_metric = datasets.load_metric('glue', 'stsb')
>>> references = [0., 1., 2., 3., 4., 5.]
>>> predictions = [0., 1., 2., 3., 4., 5.]
>>> results = glue_metric.compute(predictions=predictions, references=references)
>>> print({"pearson": round(results["pearson"], 2), "spearmanr": round(results["spearmanr"], 2)})
{'pearson': 1.0, 'spearmanr': 1.0}

>>> glue_metric = datasets.load_metric('glue', 'cola')
>>> references = [0, 1]
>>> predictions = [0, 1]
>>> results = glue_metric.compute(predictions=predictions, references=references)
>>> print(results)
{'matthews_correlation': 1.0}
""", stored examples: 0)

Directly call the `compute` method of metric, pass in `labels` and `predictions` to getMetric value:

```python
import numpy as np

fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
metric.compute(predictions=fake_preds, references=fake_labels)
```

{'matthews_correlation': 0.1513518081969605}

The metric for each text classification task is different, as follows:

- for CoLA: [Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
- for MNLI (matched or mismatched): Accuracy
- for MRPC: Accuracy and [F1 score](https://en.wikipedia.org/wiki/F1_score)
- for QNLI: Accuracy
- for QQP: Accuracy and [F1 score](https://en.wikipedia.org/wiki/F1_score)
- for RTE: Accuracy
- for SST-2: Accuracy
- for STS-B: [Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) and [Spearman's_Rank_Correlation_Coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
- for WNLI: Accuracy

So it is important to align the metric with the task

## Data preprocessing

Before feeding the data into the model, we need to preprocess the data. The preprocessing tool is called `Tokenizer`. `Tokenizer` firstTokenize the input, then convert the tokens to the corresponding token ID required in the pre-model, and then convert them to the input format required by the model.

In order to achieve the purpose of data preprocessing, we use the `AutoTokenizer.from_pretrained` method to instantiate our tokenizer, which ensures:

- We get a tokenizer that corresponds to the pre-trained model one by one.
- When using the tokenizer corresponding to the specified model checkpoint, we also download the vocabulary required by the model, to be precise, the tokens vocabulary.

This downloaded tokens vocabulary will be cached so that it will not be downloaded again when used again.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
```

Note: `use_fast=True` requires that the tokenizer must be transformers.PreTrainedTokenizerFast type, because we need to use some special features of fast tokenizer (such as multi-threaded fast tokenizer) during preprocessing. If the corresponding model does not have a fast tokenizer, just remove this option.

Almost all tokenizers corresponding to the model have corresponding fast tokenizers. We can check the features of the tokenizers corresponding to all pre-trained models in the [Model Tokenizer Correspondence Table](https://huggingface.co/transformers/index.html#bigtable).

Tokenizer can preprocess a single text or a pair of texts. The data obtained after tokenizer preprocessing meets the input format of pre-training model

```python
tokenizer("Hello, this one sentence!", "And this sentence goes with it.")
```

{'input_ids': [101, 7592, 1010, 2023, 2028, 6251, 999, 102, 1998, 2023, 6251, 3632, 2007, 2009, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

Depending on the pre-trained model we choose, we will see different returns from the tokenizer. There is a one-to-one correspondence between the tokenizer and the pre-trained model. More information can be learned [here](https://huggingface.co/transformers/preprocessing.html).

In order to pre-process our data, we need to know the different data and the corresponding data format, so we define the following dict.

```python
task_to_keys = {
"cola": ("sentence", None),
"mnli": ("premise", "hypothesis"),
"mnli-mm": ("premise", "hypothesis"),
"mrpc": ("sentence1", "sentence2"),
"qnli": ("question", "sentence"),
"qqp": ("question1", "question2"),
"rte": ("sentence1", "sentence2"),
"sst2": ("sentence", None),
"stsb": ("sentence1", "sentence2"),
"wnli": ("sentence1", "sentence2"),
}
```

Check the data format:

```python
sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")
```

Sentence: Our friends won't buy this analysis, let alone the next one we propose.

Then put the preprocessing code into a function:

```python
def preprocess_function(examples):
if sentence2_key is None:
return tokenizer(examples[sentence1_key], truncation=True)
return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
```

The preprocessing function can process a single sample or multiple samples. If the input is multiple samples, a list is returned:

```python
preprocess_function(dataset['train'][:5])
```

{'input_ids': [[101, 2256, 2814, 2180, 1005, 1056, 4965, 2023,4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012, 102], [101, 2028, 2062, 18404, 2236, 3989, 1998, 1045, 1005, 1049, 3228, 2039, 1012, 102], [101, 2028, 2062, 18404, 2236, 3989, 2030, 1045, 1005, 1049, 3228, 2039, 1012, 102] 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

Next, all samples in the dataset datasets are preprocessed by using the map function and applying the preprocessing function prepare_train_features to all samples.

```python
encoded_dataset = dataset.map(preprocess_function, batched=True)
```

Even better, the returned results will be automatically cached to avoid recalculation the next time you process (but also note that if the input changes, it may be affected by the cache!). The datasets library function will check the input parameters to determine whether there are changes. If there are no changes, the cached data will be used. If there are changes, reprocess. However, if the input parameters do not change, it is best to clear the cache when you want to change the input. The way to clean is to use `load_from_cache_file=False` parameter. In addition, the `batched=True` parameter used above is a feature of the tokenizer, because it uses multiple threads to process the input in parallel.

## Fine-tune the pre-trained model

Now that the data is ready, we need to download and load our pre-trained model, and then fine-tune the pre-trained model. Since we are doing a seq2seq task, we need a model class that can solve this task. We use the `AutoModelForSequenceClassification` class. Similar to the tokenizer, the `from_pretrained` method can also help us download and load the model, and it will also cache the model so that the model will not be downloaded repeatedly.

It should be noted that STS-B is a regression problem, and MNLI is a 3-classification problem:

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
```

Downloading: 0%| | 0.00/268M [00:00<?, ?B/s]

Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_projector.weight', 'vocab_transform.weight', 'vocab_projector.bias', 'vocab_layer_norm.bias', 'vocab_transform.bias', 'vocab_layer_norm.weight']
- This IS expected if you are initializingg DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of DistilBertForSequenceClassificationrSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.weight', 'pre_classifier.bias', 'classifier.bias']
You should probably TRAIN this model on a downstream task to be able to use it for predictions and inference.

Since our fine-tuning task is a text classification task, and we load a pre-trained language model, it will prompt us that some unmatched neural network parameters were thrown away when loading the model (for example: the neural network head of the pre-trained language model was thrown away, and the neural network head of the text classification was randomly initialized).

In order to get a `Trainer` training tool, we need 3 more elements, the most important of which is the training settings/parameters [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments). This training setting contains all the properties that define the training process.

```python
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

args = TrainingArguments(
"test-glue",
evaluation_strategy = "epoch",
save_strategy = "epoch",
learning_rate=2e-5,
per_device_train_batch_size=batch_size,
per_device_eval_batch_size=batch_size,
num_train_epochs=5,
weight_decay=0.01,
load_best_model_at_end=True,
metric_for_best_model=metric_name,
)
```

The evaluation_strategy = "epoch" parameter above tells the training code that we will do a validation evaluation every epoch.

The batch_size above is defined before this notebook.

Finally, since different tasks require different evaluation metrics, we define a function to get the evaluation method based on the task name:

```python
def compute_metrics(eval_pred):
predictions, labels = eval_pred
if task != "stsb":
predictions = np.argmax(predictions, axis=1)
else:
predictions = predictions[:, 0]
return metric.compute(predictions=predictions, references=labels)
```

Pass all to `Trainer`:

```python
validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
model,
args,
train_dataset=encoded_dataset["train"],
eval_dataset=encoded_dataset[validation_key],
tokenizer=tokenizer,
compute_metrics=compute_metrics
)
```

Start training:

```python
trainer.train()
```

The following columns in the training set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence.
***** Running training *****
Num examples = 8551
Num Epochs = 5
Instantaneous batch size per device = 16
Total train batch size (w. parallel, distributed & accumulation) = 16
Gradient Accumulation steps = 1
Total optimization steps = 2675

<div>

<progress value='2675' max='2675' style='width:300px; height:20px; vertical-align: middle;'></progress>
[2675/2675 02:49, Epoch 5/5]
</div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: left;">
<th>Epoch</th>
<th>Training Loss</th>
<th>Validation Loss</th>
<th>Matthews Correlation</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>0.525400</td>
<td>0.520955</td>
<td>0.409248</td>
</tr>
<tr>
<td>2</td>
<td>0.351600</td>
<td>0.570341</td>
<td>0.477499</td>
</tr>
<tr>
<td>3</td><td>0.236100</td>
<td>0.622785</td>
<td>0.499872</td>
</tr>
<tr>
<td>4</td>
<td>0.166300</td>
<td>0.806475</td>
<td>0.491623</td>
</tr>
<tr>
<td>5</td>
<td>0.125700</td>
<td>0.882225</td>
<td>0.513900</td>
</tr>
</tbody>
</table><p>

The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence.
***** Running Evaluation *****
Num examples = 1043
Batch size = 16
Saving model checkpoint to test-glue/checkpoint-535
Configuration saved in test-glue/checkpoint-535/config.json
Model weights saved in test-glue/checkpoint-535/pytorch_model.bin
tokenizer config file saved in test-glue/checkpoint-535/tokenizer_config.json
Special tokens file saved in test-glue/checkpoint-535/special_tokens_map.json
The following columns in the evaluation set don't have a correspondingding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence.
***** Running Evaluation *****
Num examples = 1043
Batch size = 16
Saving model checkpoint to test-glue/checkpoint-1070
Configuration saved in test-glue/checkpoint-1070/config.json
Model weights saved in test-glue/checkpoint-1070/pytorch_model.bin
tokenizer config file saved in test-glue/checkpoint-1070/tokenizer_config.json
Special tokens file saved in test-glue/checkpoint-1070/special_tokens_map.json
The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence.
***** Running Evaluation *****
Num examples = 1043
Batch size = 16
Saving model checkpoint to test-glue/checkpoint-1605
Configuration saved in test-glue/checkpoint-1605/config.json
Model weights saved in test-glue/checkpoint-1605/pytorch_model.bin
tokenizer config file saved in test-glue/checkpoint-1605/tokenizer_config.json
Special tokens file saved in test-glue/checkpoint-1605/special_tokens_map.json
The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence.
***** Running Evaluation *****
Num examples = 1043
Batch size = 16
Saving model checkpoint to test-glue/checkpoint-2140
Configuration saved in test-glue/checkpoint-2140/config.json
Model weights saved in test-glue/checkpoint-2140/pytorch_model.bin
Tokenizer config file saved in test-glue/checkpoint-2140/tokenizer_config.json
Special tokens file saved in test-glue/checkpoint-2140/special_tokens_map.json
The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: idx, sentence.
***** Running Evaluation *****
Num examples = 1043
Batch size = 16
Saving model checkpoint to test-glue/checkpoint-2675
Configuration saved in test-glue/checkpoint-2675/config.json
Model weights saved in test-glue/checkpoint-2675/pytorch_model.bin
tokenizer config file saved in test-glue/checkpoint-2675/tokenizer_config.json
Special tokens file saved in test-glue/checkpoint-2675/special_tokens_map.json

Training completed. Do not forget to share your model on huggingface.co/models =)

Loading best model from test-glue/checkpoint-2675 (score: 0.5138995234247261).

TrainOutput(global_step=2675, training_loss=0.27181456521292713, metrics={'train_runtime': 169.649, 'train_samples_per_second': 252.02, 'train_steps_per_second': 15.768, 'total_flos': 229537542078168.0, 'train_loss': 0.27181456521292713, 'epoch': 5.0})

Evaluate after training:

```python
trainer.evaluate()
```

The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClasssification.forward` and have been ignored: idx, sentence.
***** Running Evaluation *****
Num examples = 1043
Batch size = 16

<div>

<progress value='66' max='66' style='width:300px; height:20px; vertical-align: middle;'></progress>
[66/66 00:00]
</div>

{'epoch': 5.0,
'eval_loss': 0.8822253346443176,
'eval_matthews_correlation': 0.5138995234247261,
'eval_runtime': 0.9319,
'eval_samples_per_second': 1119.255,
'eval_steps_per_second': 70.825}

To see how your model fared you can compare it to the [GLUE Benchmark leaderboard](https://gluebenchmark.com/leaderboard).

## Hyperparameter search

`Trainer` also supports hyperparameter search, using [optuna](https://optuna.org/) or [Ray Tune](https://docs.ray.io/en/latest/tune/) code base.

Uncomment the following two lines to install dependencies:

```python
! pip install optuna
! pip install ray[tune]
```

During hyperparameter search, `Trainer` will return multiple trained models, so you need to pass in a defined model so that `Trainer` can continuously reinitialize the passed in model:

```python
def model_init():
return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
```

Similar to the previous call to `Trainer`:

```python
trainer = Trainer(
model_init=model_init,
args=args,
train_dataset=encoded_dataset["train"],
eval_dataset=encoded_dataset[validation_key],
tokenizer=tokenizer,
compute_metrics=compute_metrics
)
```

loading configuration file https://huggingface.co/distilbert-base-uncased/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/23454919702d26495337f3da04d1655c7ee010d5ec9d77bdb9e399e00302c0a1.d423bdf2f58dc8b77d5f5d18028d7ae4a72dcfd8f468e81fe979ada957a8c361
Model config DistilBertConfig {
"activation": "gelu",
"architectures": [
"DistilBertForMaskedLM"
],
"attention_dropout": 0.1,
"dim": 768,
"dropout": 0.1,
"hidden_dim": 3072,
"initializer_range": 0.02,
"max_position_embeddings": 512,
"model_type": "distilbert",
"n_heads": 12,
"n_layers": 6,
"pad_token_id": 0,
"qa_dropout": 0.1,"seq_classif_dropout": 0.2,
"sinusoidal_pos_embds": false,
"tie_weights_": true,
"transformers_version": "4.9.1",
"vocab_size": 30522
}

loading weights file https://huggingface.co/distilbert-base-uncased/resolve/main/pytorch_model.bin from cache at /root/.cache/huggingface/transformers/9c169103d7e5a73936dd2b627e42851bec0831212b677c637033ee4bce9ab5ee.126183e36667471617ae2f0835fab707baa54b731f991507ebbb55ea85adb12a
Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_projector.weight', 'vocab_transform.weight', 'vocab_projector.bias', 'vocab_layer_norm.bias', 'vocab_transform.bias', 'vocab_layer_norm.weight']
- This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.weight', 'pre_classifier.bias', 'classifier.bias']
You should probably TRAIN this model on a downstream task to be able to use it for predictions and inference.

Call the method `hyperparameter_search`. Note that this process may take a long time. We can first use part of the data set for hyperparameter search, and then train the full amount.
For example, use 1/10 of the data for search:

```python
best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
```

`hyperparameter_search` will return the parameters related to the best model:

```python
best_run
```

Set `Trainer` to the best parameters found and train:

```python
for n, v in best_run.hyperparameters.items():
setattr(trainer.args, n, v)

trainer.train()```

Finally, don’t forget to check how to upload a model , upload the model to [🤗 Model Hub](https://huggingface.co/models) . Then you can use the model you uploaded directly by using the model name, just like at the beginning of this notebook.

```python

```