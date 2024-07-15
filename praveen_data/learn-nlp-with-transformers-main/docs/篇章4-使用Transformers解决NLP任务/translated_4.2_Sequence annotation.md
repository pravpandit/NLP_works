The jupter notebooks involved in this article are in the [Chapter 4 code repository](https://github.com/datawhalechina/learn-nlp-with-transformers/tree/main/docs/%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1).

If you are opening this notebook in Google Colab, you may need to install the Transformers and 🤗Datasets libraries. Uncomment the following commands to install.

```python
!pip install datasets transformers seqeval
```

If you are opening this notebook locally, make sure you have installed the above dependencies. You can also find a multi-GPU distributed training version of this notebook [here](https://github.com/huggingface/transformers/tree/master/examples/token-classification).

This sectionThe model structure involved is basically the same as BERT in the previous chapter. What needs to be learned is the data processing method and model training method for specific tasks.

# *Sequence labeling (token-level classification problem)*

Sequence labeling can also be regarded as a token-level classification problem: classify each token. In this notebook, we will show how to use the transformer model in [🤗 Transformers](https://github.com/huggingface/transformers) to do token-level classification problems. Token-level classification tasks usually refer to predicting a label result for each token in the text. The figure below shows a NER entity noun recognition task.

![Widget inference representing the NER task](https://github.com/huggingface/notebooks/blob/master/examples/images/token_classification.png?raw=1)

The most common token-level classification tasks:

- NER (Named-entity recognition) Discriminationout nouns and entities in the text (person, organization, location, etc.).
- POS (Part-of-speech tagging) tags tokens according to grammar (noun, verb, adjective, etc.)
- Chunk (Chunking) puts tokens of the same phrase together.

For the above tasks, we will show how to load datasets using a simple Dataset library and fine-tune pre-trained models using the `Trainer` interface in transformer.

As long as the pre-trained transformer model has a token classification neural network layer at the top (such as the `BertForTokenClassification` mentioned in the previous chapter) (in addition, due to the new tokenizer feature of the transformer library, the corresponding pre-trained model may also need to have the fast tokenizer function, refer to [this table](https://huggingface.co/transformers/index.html#bigtable)), then this notebook can theoretically use a variety of transformers.ormer model ([models panel](https://huggingface.co/models)), solve any token-level classification task.

If your task is different, it is likely that only minor changes will be needed to use this notebook. At the same time, you should adjust the batch size required for fine-tuning training according to your GPU memory to avoid memory overflow.

```python
task = "ner" #needs to be "ner", "pos" or "chunk"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
```

## Loading data

We will use the [🤗 Datasets](https://github.com/huggingface/datasets) library to load data and corresponding evaluation methods. Data loading and evaluation method loading only need to use `load_dataset` and `load_metric`.

```python
from datasets import load_dataset, load_metric
```

The examples in this notebook use [CONLL 2003 dataset](https://www.aclweb.org/anthology/W03-0419.pdf) dataset. This notebook should be able to handle any token classification task in the 🤗 Datasets library. If you are using your own custom json/csv file dataset, you need to check the [datasets documentation](https://huggingface.co/docs/datasets/loading_datasets.html#from-local-files) to learn how to load it. Custom datasets may require some adjustments in the loading attribute names.

```python
datasets = load_dataset("conll2003")
```

The `datasets` object itself is a [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict) data structure. For the training set, validation set, and test set, just use the corresponding key (train, validation, test) to get the corresponding data.

```pythonon
datasets
```

DatasetDict({
train: Dataset({
features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
num_rows: 14041
})
validation: Dataset({
features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
num_rows: 3250
})
test: Dataset({
features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
num_rows: 3453
})
})

Whether in the training set, validation machine or test set, datasetts contains a column called tokens (generally speaking, the text is segmented into many words) and a column called label, which corresponds to the label of the tokens.

Given a data segmentation key (train, validation or test) and a subscript, you can view the data.

```python
datasets["train"][0]
```

{'chunk_tags': [11, 21, 11, 12, 21, 22, 11, 12, 0],
'id': '0',
'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0],
'pos_tags': [22, 42, 16, 21, 35, 37, 16, 21, 7],
'tokens': ['EU',
'rejects',
'German',
'call',
'to',
'boycott',
'British',
'lamb',
'.']}

All data labels are encoded as integers and can be directly used.directly used by the pre-trained transformer model. The actual classes corresponding to the encodings of these integers are stored in `features`.

```python
datasets["train"].features[f"ner_tags"]
```

Sequence(feature=ClassLabel(num_classes=9, names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'], names_file=None, id=None), length=-1, id=None)

So for NER, 0 corresponds to the label class "O", 1 corresponds to "B-PER" and so on. "O" means no special entity. This example contains 4 entity types (PER, ORG, LOC, MISC), each of which has a B- (entity start token) prefix and an I- (entity middle token) prefix.

- 'PER' for person
- 'ORG' for organization
- 'LOC' forr location
- 'MISC' for miscellaneous

```python
label_list = datasets["train"].features[f"{task}_tags"].feature.names
label_list
```

['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

To further understand what the data looks like, the following function will randomly select a few examples from the dataset for display.

```python
from datasets import ClassLabel, Sequence
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
elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
display(HTML(df.to_html()))
```

```python
show_random_elements(datasets["train"])
```

<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>id</th>
<th>tokens</th>
<th>pos_tags</th>
<th>chunk_tags</th>
<th>ner_tags</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th><td>2227</td>
<td>[Result, of, a, French, first, division, match, on, Friday, .]</td>
<td>[NN, IN, DT, JJ, JJ, NN, NN, IN, NNP, .]</td>
<td>[B-NP, B-PP, B-NP, I-NP, I-NP, I-NP, B-PP, B-NP, O]</td>
<td>[O, O, O, B-MISC, O, O, O, O, O, O]</td>
</tr>
<tr>
<th>1</th>
<td>2615</td>
<td>[Mid-tier, golds, up, in, heavy, trading, .]</td>
<td>[NN, NNS, IN, IN, JJ, NN, .]</td>
<td>[B-NP, I-NP, B-PP, B-PP, B-NP, I-NP, O]</td>
<td>[O,O, O, O, O, O, O]</td>
</tr>
<tr>
<th>2</th>
<td>10256</td>
<td>[Neagle, (, 14-6, ), beat, the, Braves, for, the, third, time, this, season, ,, allowing, two, runs, and, six, hits, in, eight, innings, .]</td>
<td>[NNP, (, CD, ), VB, DT, NNPS, IN, DT, JJ, NN, DT, NN, ,, VBG, CD, NNS, CC, CD, NNS, IN, CD, NN, .]</td>
<td>[B-NP, O, B-NP, O, B-VP, B-NP, I-NP, B-PP, B-NP, I-NP, I-NP, B-NP, I-NP, O, B-VP, B-NP, I-NP, O, B-NP, I-NP, B-PP, B-NP, I-NP, O]</td><td>[B-PER, O, O, O, O, O, B-ORG, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O]</td>
</tr>
<tr>
<th>3</th>
<td>10720</td>
<td>[Hansa, Rostock, 4, 1, 2, 1, 5, 4, 5]</td>
<td>[NNP, NNP, CD, CD, CD, CD, CD, CD, CD]</td>
<td>[B-NP, I-NP, I-NP, I-NP, I-NP, I-NP, I-NP, I-NP, I-NP]</td>
<td>[B-ORG, I-ORG, O, O, O, O, O, O]</td>
</tr>
<tr>
<th>4</th>
<td>7125</td>
<td>[MONTREAL, 70, 59, .543, 11]</td>
<td>[NNP, CD, CD,CD, CD]</td>
<td>[B-NP, I-NP, I-NP, I-NP, I-NP]</td>
<td>[B-ORG, O, O, O, O]</td>
</tr>
<tr>
<th>5</th>
<td>3316</td>
<td>[Softbank, Corp, said, on, Friday, that, it, would, procure, $, 900, million, through, the, foreign, exchange, market, by, September, 5, as, part, of, its, acquisition, of, U.S., firm, ,, Kingston, Technology, Co, .]</td>
<td>[NNP, NNP, VBD, IN, NNP, IN, PRP, MD, NN, $, CD, CD, IN, DT, JJ, NN, NN, IN, NNP, CD, IN, NN, IN, PRP$, NN,IN, NNP, NN, ,, NNP, NNP, NNP, .]</td>
<td>[B-NP, I-NP, B-VP, B-PP, B-NP, B-SBAR, B-NP, B-VP, B-NP, I-NP, I-NP, B-PP, B-NP, I-NP, I-NP, B-PP, B-NP, I-NP, I-NP, B-NP, B-PP, B-NP, B-NP, B-NP, B-NP, I-NP, B-PP, B-NP, I-NP, B-NP, I-NP, O, B-NP, I-NP, I-NP, O]</td>
<td>[B-ORG, I-ORG, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, B-LOC, O, O, B-ORG, I-ORG, I-ORG, O]</td>
</tr>
<tr>
<th>6</th>
<td>3923</td>
<td>[Ghent, 3, Aalst, 2]</td>
<td>[NN, CD, NNP, CD]</td>
<td>[B-NP, I-NP, I-NP, I-NP]</td>
<td>[B-ORG, O, B-ORG, O]</td>
</tr>
<tr>
<th>7</th>
<td>2776</td>
<td>[The, separatists, ,, who, swept, into, Grozny, on, August, 6, ,, still, control, large, areas, of, the, centre, of, town, ,, and, Russian, soldiers, are, based, at, checkpoints, on, the, approach, roads, .]</td>
<td>[DT, NNS, ,, WP, VBD, IN, NNP, IN, NNP, CD, ,, RB, VBP, JJ, NNS, IN, DT, NN, IN, NN, ,, CC, JJ, NNS, VBP, VBN, IN, NNS, IN, DT, NN, NNS, .]</td>
<td>[B-NP, I-NP, O, B-NP, B-VP, B-PP, B-NP, B-PP, B-NP, I-NP, O, B-ADVP, B-VP, B-NP, I-NP, B-PP, B-NP, I-NP, B-PP, B-NP, O, O, B-NP, I-NP, B-VP, I-VP, B-PP, B-NP, B-NP, I-NP, I-NP, O]</td>
<td>[O, O, O, O, O, O, B-LOC, O, O, O, O, O, O, O, O, O, O, O, O, O, O, B-MISC, O, O, O, O, O, O, O, O, O, O]</td>
</tr>
<tr>
<th>8</th>
<td>1178</td>
<td>[Doctor, Masserigne, Ndiaye, said, medical, staff, were, overwhelmed, with, work, ., "]</td>
<td>[NNP, NNP, NNP, VBD, JJ, NN, VBD, VBN, IN, NN, ., "]</td>
<td>[B-NP, I-NP, I-NP, B-VP, B-NP, I-NP, B-VP, I-VP, B-PP, B-NP, O, O]</td>
<td>[O, B-PER, I-PER, O, O, O, O, O, O, O, O]</td>
</tr>
<tr>
<th>9</th>
<td>10988</td>
<td>[Reuters, historical, calendar, -, September, 4, .]</td>
<td>[NNP, JJ, NN, :, NNP, CD, .]</td>
<td>[B-NP, I-NP, I-NP, O, B-NP, I-NP, O]</td>
<td>[B-ORG, O, O, O, O, O, O]</td>
</tr>
</tbody>
</table>

## Preprocessing data

Before feeding the data into the model, we need to preprocess the data. The preprocessing tool is called `Tokenizer`. `Tokenizer` first tokenizes the input, then converts the tokens into the corresponding token ID required in the pre-model, and then converts them into the input format required by the model.

In order to achieve the purpose of data preprocessing, we use the `AutoTokenizer.from_pretrained` method to instantiate our tokenizer, which ensures:

- We get a tokenizer that corresponds to the pre-trained model one by one.

- When using the tokenizer corresponding to the specified model checkpoint, we also download the vocabulary required by the model, more precisely, the tokens vocabulary.

This downloaded tokens vocabulary will be cached so that it will not be downloaded again when used again.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

Note: The following codeThe code requires that the tokenizer must be of type transformers.PreTrainedTokenizerFast, because we need to use some special features of the fast tokenizer (such as multi-threaded fast tokenizer) during preprocessing.

Almost all tokenizers corresponding to models have corresponding fast tokenizers. We can view the features of tokenizers corresponding to all pre-trained models in the [model tokenizer corresponding table](https://huggingface.co/transformers/index.html#bigtable).

```python
import transformers
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
```

Check whether the model has a fast tokenizer in the [big table of models here](https://huggingface.co/transformers/index.html#bigtable).

Tokenizers can preprocess a single text or a pair of texts.Processing, the data obtained after tokenizer preprocessing meets the input format of the pre-trained model

```python
tokenizer("Hello, this is one sentence!")
```

{'input_ids': [101, 7592, 1010, 2023, 2003, 2028, 6251, 999, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

```python
tokenizer(["Hello", ",", "this", "is", "one", "sentence", "split", "into", "words", "."], is_split_into_words=True)
```

{'input_ids': [101, 7592, 1010, 2023, : [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}Note that transformer pre-training models usually use subwords during pre-training. If our text input has been segmented into words, these words will be further segmented by our tokenizer. For example:

```python
example = datasets["train"][4]
print(example["tokens"])
```

['Germany', "'s", 'representative', 'to', 'the', 'European', 'Union', "'s", 'veterinary', 'committee', 'Werner', 'Zwingmann', 'said', 'on', 'Wednesday', 'consumers', 'should', 'buy', 'sheepmeat', 'from', 'countries', 'other', 'than', 'Britain', 'until', 'the', 'scientific', 'advice', 'was', 'clearer', '.']

```python
tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokens)
```

['[CLS]', 'germany', "'", 's', 'representative', 'to', 'the', 'european', 'union', "'", 's', 'veterinary', 'committee', 'werner', 'z', '##wing', '##mann', 'said', 'on', 'wednesday', 'consumers', 'should', 'buy', 'sheep', '##me', '##at', 'from', 'countries', 'other', 'than', 'britain', 'until', 'the', 'scientific', 'advice', 'was', 'clearer', '.', '[SEP]']

The words "Zwingmann" and "sheepmeat" are further divided into 3 subtokens.

Since the labeled data is usually annotated at the word level, since the word is also divided into subtokens, it means that we also need to align the subtokens of the labeled data. At the same time, due to the requirements of the pre-trained model input format, some special symbols such as: `[CLS]` and `[SEP]` are often required.

```python
len(example[f"{task}_tags"]), len(tokenized_input["input_ids"])
```

(31, 39)

The tokenizer has a ` `word_ids` method that can help us solve this problem.

```python
print(tokenized_input.word_ids())
```

[None, 0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 11, 11, 12, 13, 14, 15, 16, 17, 18, 18, 18, 19, 20, 21,22, 23, 24, 25, 26, 27, 28, 29, 30, None]

We can see that word_ids corresponds each subtokens position to a word subscript. For example, the first position corresponds to the 0th word, and the second and third positions correspond to the first word. Special characters correspond to None. With this list, we can align subtokens with words and annotated labels.

```python
word_ids = tokenized_input.word_ids()
aligned_labels = [-100 if i is None else example[f"{task}_tags"][i] for i in word_ids]
print(len(aligned_labels), len(tokenized_input["input_ids"]))
```

39 39

We usually set the label of special characters to -100. In the model, -100 is usually ignored and loss is not calculated.

We have two ways to align labels:
- Align multiple subtokens to one word, align one label
- MultipleThe first subtoken of subtokens is aligned with word and a label, and other subtokens are directly assigned -100.

We provide these two methods, which can be switched by `label_all_tokens = True`.

```python
label_all_tokens = True
```

Finally, we combine all the content into our preprocessing function. `is_split_into_words=True` is already finished above.

```python
def tokenize_and_align_labels(examples):
tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

labels = []
for i, label in enumerate(examples[f"{task}_tags"]):
word_ids = tokenized_inputs.word_ids(batch_index=i)
previous_word_idx = None
label_ids = []
for word_idx in word_ids:
# Special tokens have a word id that is None. We set the label to -100 so they are automatically
# ignored in the loss function.
if word_idx is None:
label_ids.append(-100)
# We set the label for the first token of each word.
elif word_idx != previous_word_idx:
label_ids.append(label[word_idx])
# For the other tokens in a word, we set the label to either the current label or -100, depending on
# the label_all_tokens flag.
else:
label_ids.append(label[word_idx] if label_all_tokens else -100)
previous_word_idx = word_idx

labels.append(label_ids)

tokenized_inputs["labels"] = labels
return tokenized_inputs
```

The above preprocessing function can process one sample or multiple sample examples. If it processes multiple samples, it returns a list of results after multiple samples are preprocessed.

```python
tokenize_and_align_labels(datasets['train'][:5])
```: {'input_ids': [[101, 7327, 19164, 2446, 2655, 2000, 17757, 2329, 12559, 1012, 102], [101, 2848, 13934, 102], [101, 9371, 2727, 1011, 5511, 1011, 2570, 102], [101, 1996, 2647, 3222, 2056, 2006, 9432, 2009, 18335, 2007, 2446, 6040, 2000, 10390, 2000, 2000, 1996, 2647, 2586, 1005, 1055, 15651, 2837, 14121, 1062, 9328, 5804, 2056, 2006, 9317, 10390, 2321, 1, 1, 1, 1, 1, 1, 3, 4965, 8351, 4168, 4017, 2013, 3032, 2060, 2084, 3725, 2127, 1996, 4045, 6040, 2001, 24509, 1012, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, : 'labels': [[-100, 3, 0, 7, 0, 0, 7, 0, 0, -100], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'labels': [[-100, 3, 0, 7, 0, 0, 7, 0, 0, -100], [-100, 1, 2, -100], [-100,0, 0, 0, 3, 4, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 -100]]}

Next, preprocess all samples in the dataset datasets by using the map function to apply the preprocessing function prepare_train_features to all samples.

```python
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
```

Better yet, the returned results will be automatically cached to avoid recalculation the next time you process them (but be aware that if the input is changed, it may be affected by the cache!). daThe tasets library function will detect the input parameters to determine whether there are any changes. If there are no changes, the cached data will be used. If there are changes, the data will be reprocessed. However, if the input parameters do not change, it is best to clean the cache when you want to change the input. The way to clean is to use the `load_from_cache_file=False` parameter. In addition, the `batched=True` parameter used above is a feature of the tokenizer, because it uses multiple threads to process the input in parallel.

## Fine-tune the pre-trained model

Now that the data is ready, we now need to download and load our pre-trained model, and then fine-tune the pre-trained model. Since we are doing a seq2seq task, we need a model class that can solve this task. We use the `AutoModelForTokenClassification` class. Similar to the tokenizer, the `from_pretrained` method can also help us download and load the model, and it will also cache the model so that the model will not be downloaded repeatedly.

```python
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
```

Downloading: 0%| | 0.00/268M [00:00<?, ?B/s]

Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForTokenClassification: ['vocab_transform.weight', 'vocab_layer_norm.bias', 'vocab_projector.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_layer_norm.weight']
- This IS expected if you are initializing DistilBertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DistilBertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of DistilBertForTokenClassificationtion were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a downstream task to be able to use it for predictions and inference.

Since our fine-tuning task is the token classification task, and we load the pre-trained language model, it will prompt us that some unmatched neural network parameters were thrown away when loading the model (for example: the neural network head of the pre-trained language model was thrown away, and the neural network head of the token classification was randomly initialized).

In order to get a `Trainer` training tool, we also need 3 elements, the most important of which is the training settings/parameters [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments). This training setup contains all the properties that define the training process.

```python
args = TrainingArguments(
f"test-{task}",
evaluation_strategy = "epoch",
learning_rate=2e-5,
per_device_train_batch_size=batch_size,
per_device_eval_batch_size=batch_size,
num_train_epochs=3,
weight_decay=0.01,
)
```

The evaluation_strategy = "epoch" parameter above tells the training code that we will do a validation evaluation once per epoch.

The batch_size is defined above in this notebook.

Finally, we need a data collator to feed our processed input to the model.

```python
from transformers importDataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)
```

The last thing left to set up `Trainer` is to define the evaluation method. We use the [`seqeval`](https://github.com/chakki-works/seqeval) metric to complete the evaluation. We will also do some data post-processing before feeding the model predictions into the evaluation:

```python
metric = load_metric("seqeval")
```

The input to the evaluation is a list of predictions and labels

```python
labels = [label_list[i] for i in example[f"{task}_tags"]]
metric.compute(predictions=[labels], references=[labels])
```

{'LOC': {'f1': 1.0, 'number': 2, 'precision': 1.0, 'recall': 1.0},
'ORG': {'f1': 1.0, 'number': 1, 'precision': 1.0, 'recall': 1.0},
'PER': {'f1': 1.0, 'number': 1, 'precision': 1.0, 'recall': 1.0},
'overall_accuracy': 1.0,
'overall_f1': 1.0,
'overall_precision': 1.0,
'overall_recall': 1.0}

Do some post-processing on the model prediction results:
- Select the subscript with the maximum probability of the predicted classification
- Convert the subscript to label
- Ignore the -100

The following function combines the above steps.

```python
import numpy as np

def compute_metrics(p):
predictions, labels = p
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
for prediction, label in zip(predictions, labels)
]
true_labels = [
[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
return {
"precision": results["overall_precision"],"recall": results["overall_recall"],
"f1": results["overall_f1"],
"accuracy": results["overall_accuracy"],
}
```

We calculate the total precision/recall/f1 of all categories, so we will throw away the precision/recall/f1 of a single category

Just pass the data/model/parameters to `Trainer`

```python
trainer = Trainer(
model,
args,
train_dataset=tokenized_datasets["train"],
eval_dataset=tokenized_datasets["validation"],
data_collator=data_collator,
tokenizer=tokenizer,
compute_metrics=compute_metrics
)
```

Call the `train` method to start training

```pythonn
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

<progress value='2634' max='2634' style='width:300px; height:20px; vertical-align: middle;'></progress>[2634/2634 01:45, Epoch 3/3]
</div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: left;">
<th>Epoch</th>
<th>Training Loss</th>
<th>Validation Loss</th>
<th>Precision</th>
<th>Recall</th>
<th>F1</th>
<th>Accuracy</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>0.237721</td>
<td>0.068198</td>
<td>0.903148</td>
<td>0.921132</td>
<td>0.912051</td>
<td>0.979713</td></tr>
<tr>
<td>2</td>
<td>0.053160</td>
<td>0.059337</td>
<td>0.927697</td>
<td>0.932990</td>
<td>0.930336</td>
<td>0.983113</td>
</tr>
<tr>
<td>3</td>
<td>0.029850</td>
<td>0.059346</td>
<td>0.929267</td>
<td>0.939143</td>
<td>0.934179</td>
<td>0.984257</td>
</tr>
</tbody>
</table><p>

TrainOutput(global_step=2634, training_loss=0.08569671253227518)

We can use the `evaluate` method again to evaluate otherdata set.


```python
trainer.evaluate()
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

<progress value='408' max='204' style='width:300px; height:20px; vertical -align: middle;'></progress>
[204/204 00:05]
</div>{'eval_loss': 0.05934586375951767,
'eval_precision': 0.9292672127518264,
'eval_recall': 0.9391430808815304,
'eval_f1': 0.9341790463472988,
'eval_accuracy': 0.9842565968195466,
'epoch': 3.0}

If we want to get the precision/recall/f1 of a single category, we can directly input the results into the same evaluation function:

```python
predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
[label_list[p] for (p, l) inzip(prediction, label) if l != -100]
for prediction, label in zip(predictions, labels)
]
true_labels = [
[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
results
```

{'LOC': {'precision': 0.949718574108818,
'recall': 0.966768525592055,
'f1': 0.9581677077418134,
'number': 2618},
'MISC': {'precision': 0.8132387706855791,
'recall': 0.8383428107229894,
'f1': 0.82559999999999999,
'number': 1231},
'ORG': {'precision': 0.9055232558139535,
'recall': 0.9090466926070039,
'f1': 0.9072815533980583,
'number': 2056},
'PER': {'precision': 0.9759552042160737,
'recall': 0.9765985497692815,
'f1': 0.9762767710049424,
'number': 3034},
'overall_precision': 0.9292672127518264,
'overall_recall': 0.9391430808815304,
'overall_f1': 0.9341790463472988,
'overall_accuracy': 0.9842565968195466}

Finally, don't forget to upload the model to [🤗 Model Hub](https://huggingface.co/models) (click [here](https://huggingface.co/transformers/model_sharing.html) to see how to upload). Then you can use the model you uploaded directly by using the model name, just like at the beginning of this notebook.

```python

```