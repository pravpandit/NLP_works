The jupter notebook involved in this article is in the [Chapter 4 code base](https://github.com/datawhalechina/learn-nlp-with-transformers/tree/main/docs/%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1).

It is recommended to open this tutorial directly using google colab notebook to quickly download relevant datasets and models.
If you are opening this notebook in google colab, you may need to install the Transformers and 🤗Datasets libraries. Uncomment the following commands to install.

```python
# !pip install datasets transformers
```

# Fine-tune the transformer model on the machine question answering task

In this notebook, we will learn how to fine-tune the transformer model of [🤗 Transformers](https://github.com/huggingface/transformers)rmer model to solve machine question answering tasks. This article mainly solves the extractive question answering task: given a question and a text, find the text fragment (span) that can answer the question from the text. By using the `Trainer` API and the dataset package, we will easily load the dataset and then fine-tune the transformers. The figure below shows a simple example
![Widget inference representing the QA task](images/question_answering.png)

**Note:** Note: The question answering task in this article is to extract answers from text, not to generate answers directly!

The examples designed in this notebook can be used to solve any extractive question answering task similar to SQUAD 1 and SQUAD 2, and any model checkpoint in the [Model Library Model Hub](https://huggingface.co/models) can be used, as long as these models contain a token classification head and a fast tokenizer. For the correspondence between models and fast tokenizers, see: [this table](https://huggingface.co/transformers/index.html#bigtable).

If your dataset is different from this notebook, you can use this notebook directly with only minor adjustments. Of course, depending on your hardware (computer memory, graphics card size), you need to adjust the batch size reasonably to avoid out-of-memory errors.
Set those three parameters, then the rest of the notebook should run smoothly:

```python
# squad_v2 equals True or False to use SQUAD v1 or SQUAD v2 respectively.
# If you are using other datasets, True means that the model can answer "unanswerable" questions, that is, some questions are not answered, while False means that all questions must be answered.
squad_v2 = False
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
```

## Load the dataset

We will use [🤗 Datasets](https://github.com/huggingface/datasets) library to download the data and get the metrics we need (to compare with the benchmark).

These two tasks can be easily accomplished using the functions `load_dataset` and `load_metric`.

```python
from datasets import load_dataset, load_metric
```

For example, we will use the [SQUAD dataset](https://rajpurkar.github.io/SQuAD-explorer/) in this notebook. Similarly, this notebook is also compatible with all question answering datasets provided by the datasets library.

If you are using your own dataset (json or csv format), please check out the [Datasets documentation](https://huggingface.co/docs/datasets/loading_datasets.html#from-local-files) to learn how to load your own dataset. You may need to adjust the names used for each column.

```python
# Download the data (make sure you have network)
datasets = load_dataset("squad_v2" if squad_v2 else"squad")
```

This `datasets` object is a [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict) structure, and training, validation, and testing correspond to a key in this dict respectively.

```python
# View the following datasets and their properties
datasets
```

DatasetDict({
train: Dataset({
features: ['id', 'title', 'context', 'question', 'answers'],
num_rows: 87599
})
validation: Dataset({
features: ['id', 'title', 'context', 'question', 'answers'],
num_rows: 10570})
})

Whether it is a training set, a validation set or a test set, each Q&A data sample will have three keys: "context", "question" and "answers".

We can use a subscript to select a sample.

```python
datasets["train"][0]
# answers represent answers
# context represents text fragments
# question represents questions
```

{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christwith arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
'id': '5733be284776f41900661182',
'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
'title': 'University_of_Notre_Dame'}

Note the annotation of answers. In addition to the answer text in the text snippet, answers also gives the position of the answer (counted from the beginning of the character, the 515th in the above example).

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
show_random_elements(datasets["train"], num_examples=2)
```

<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>answers</th>
<th>context</th>
<th>id</th>
<th>question</th>
<th>title</th>
</tr>
</thead>
<tbody><tr>
<th>0</th>
<td>{'answer_start': [185], 'text': ['diesel fuel']}</td>
<td>In Alberta, five bitumen upgraders produce synthetic crude oil and a variety of other products: The Suncor Energy upgrader near Fort McMurray, Alberta produces synthetic crude oil plus diesel fuel; the Syncrude Canada, Canadian Natural Resources, and Nexen upgraders near Fort McMurray produce synthetic crude oil; and the Shell Scotford Upgrader near Edmonton produces synthetic crude oil plus an intermediate feedstock for the nearby Shell Oil Refinery. A sixth upgrader, under construction in 2015 near Redwater, Alberta, will upgrade half of its crude bitumen directly to diesel fuel, with the remainder of the output being sold as feedstock to nearby oil refineries and petrochemical plants.</td>
<td>571b074c9499d21900609be3</td>
<td>Besides crude oil, what does the Suncor Energy plant produce?</td>
<td>Asphalt</td>
</tr>
<tr>
<th>1</th>
<td>{'answer_start': [191], 'text': ['the GIOVE satellites for the Galileo system']}</td>
<td>Compass-M1 is an experimental satellite launched for signal testing and validation and for the frequency filing on 14 April 2007. The role of Compass-M1 for Compass is similar to the role of the GIOVE satellites for the Galileo system. The orbit of Compass-M1 is nearly circular, has an altitude of 21,150 km and an inclination of 55.5 degrees.</td>
<td>56e1161ccd28a01900c6757b</td>
<td>The purpose of theCompass-M1 satellite is similar to the purpose of what other satellite?</td>
<td>BeiDou_Navigation_Satellite_System</td>
</tr>
</tbody>
</table>

## Preprocessing the training data

Before feeding the data into the model, we need to preprocess the data. The preprocessing tool is called `Tokenizer`. `Tokenizer` first tokenizes the input, then converts the tokens into the corresponding token ID required in the pre-model, and then converts it into the input format required by the model.

In order to achieve the purpose of data preprocessing, we use the `AutoTokenizer.from_pretrained` method to instantiate our tokenizer, which ensures:

- We get a tokenizer that corresponds to the pre-trained model one by one.
- When using the tokenizer corresponding to the specified model checkpoint, we also download the vocabulary required by the model, more precisely, the tokens vocabulary.The downloaded tokens vocabulary will be cached so that it will not be downloaded again when used again.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

The following code requires that the tokenizer must be of type transformers.PreTrainedTokenizerFast, because we need to use some special features of fast tokenizer (such as multi-threaded fast tokenizer) during preprocessing.

Almost all tokenizers corresponding to models have corresponding fast tokenizers. We can view the features of all tokenizers corresponding to pre-trained models in the [Model Tokenizer Correspondence Table](https://huggingface.co/transformers/index.html#bigtable).

```python
import transformers
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
```

```python
# If we want to see the text format after tokenizer preprocessing, we only use the tokenizer's tokenize method. Add special tokens means adding special tokens required by the pre-trained model.
print("single text tokenize: {}".format(tokenizer.tokenize("What is your name?"), add_special_tokens=True))
print("2 text tokenize: {}".format(tokenizer.tokenize("My name is Sylvain.", add_special_tokens=True)))
# The input format of the pre-trained model requires token IDs and an attetnion mask. You can use the following method to get the input required by the pre-trained model format.
```

Single text tokenize: ['what', 'is', 'your', 'name', '?']
2 text tokenize: ['[CLS]','my', 'name', 'is', 'sy', '##lva', '##in', '.', '[SEP]']

Tokenizer can preprocess a single text or a pair of texts. The data obtained after tokenizer preprocessing meets the input format of the pretrained model

```python
# Preprocess a single text
tokenizer("What is your name?")
```

{'input_ids': [101, 2054, 2003, 2115, 2171, 1029, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}

```python
# Preprocess 2 texts. You can see that tokenizer adds 101 token ID at the beginning and 102 token in the middle. The ID distinguishes the two texts and ends with 102. These rules are designed by the pre-trained model.
tokenizer("What is your name?", "My name is Sylvain.")
```

{'input_ids': [101, 2054, 2003, 2115, 2171, 1029, 102, 2026, 2171, 2003, 25353, 22144, 2378, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

The token IDs or input_ids you see above generally vary with the names of the pre-trained models. The reason is that different pre-trained models set different rules during pre-training. But as long as the names of the tokenizer and the model are the same, the input format of the tokenizer preprocessing will meet the model requirements. For more information about pre-processing, refer to [this tutorial](https://huggingface.co/transformers/preprocessing.html)

Now we also need to think about how pre-trained machine question answering models handle very long texts. Generally speaking, there is a maximum length requirement for pre-trained model input, so we usually truncate overlong inputs. However, if we truncate the overlong context in the question-answer data triple <question, context, answer>, we may lose the answer (because we extract asmall snippet as the answer). To solve this problem, the following code finds an example of exceeding the length and then shows you how to handle it. We slice the overlong input into multiple shorter inputs, each of which must meet the model's maximum length input requirement. Since the answer may exist in the slice, we need to allow intersections between adjacent slices, which is controlled by the `doc_stride` parameter in the code.

Machine question answering pre-trained models usually concatenate question and context as input, and then let the model find the answer from the context.

```python
max_length = 384 # Maximum length of the input feature, after question and context are concatenated
doc_stride = 128 # Number of overlapping tokens between 2 slices.
```

The for loop iterates through the dataset and looks for an extremely long sample. The maximum input required by the notebook example model is 384 (512 is also commonly used)

```python
for i, example in enumerate(datasets["train"]):
if len(tokenizer(example["question"], example["context"])["input_ids"]) > 384:
break
example = datasets["train"][i]
```

If it is not truncated, the input length is 396

```python
len(tokenizer(example["question"], example["context"])["input_ids"])
```

396

Now if we truncate to the maximum length of 384, the information of the extra-long part will be lost

```python
len(tokenizer(example["question"], example["context"], max_length=max_length, truncation="only_second")["input_ids"])
```

384

Note that in general, we only slice the context, not the question. Since the context is concatenated after the question, corresponding to the second text, we use `only_second` to control it. The tokenizer uses `doc_stride` to control the overlap length between slices.

```python
tokenized_example = tokenizer(
example["question"],
example["context"],
max_length=max_length,
truncation="only_second",
return_overflowing_tokens=True,
stride=doc_stride
)
```

Due to slicing of the super long input, we get multiple inputs, the lengths of these input_ids are

```python
[len(x) for x in tokenized_example["input_ids"]]
```

[384, 157]

We can restore the preprocessed token IDs, input_ids to text format:

```python
for i, x in enumerate(tokenized_example["input_ids"][:2]):
print("Slice: {}".format(i))
print(tokenizer.decode(x))
```Slice: 0
[CLS] how many wins does the notre dame men's basketball team have? [SEP] the men's basketball team has over 1, 600 wins, one of only 12 schools who have reached that mark, and have appeared in 28 ncaa tournaments. former player austin carr holds the record for most points scored in a single game of the tournament with 61. although the team has never won the ncaa tournament, they were named by the helms athletic foundation as national champions twice. the team has orchestrated a number of upsets of number one ranked teams, the most notable of which was ending ucla's record 88 - game winning streak in 1974. the team has beaten an additional eight number - one teams, and those nine wins rank second, to ucla's 10, all - time in wins against the top team. the team plays in newly renovated purcell pavilion ( within the edmund p. joyce center ), which reopened for the beginning of the 2009 – 2010 season. the team is coached by mike brey, who, as of the 2014 – 15 season, his fifteenth at Notre Dame, has achieved a 332 - 165 record. in 2009 they were invited to the nit, where they advanced to the semifinals but were beaten by Penn State who went on and beat Baylor in the championship. the 2010 – 11 team concluded its regular season ranked number seven in the country, with a record of 25 – 5, Brey's fifth straight 20 - win season, and a second - place finish in the Big East. during the 2014 - 15 season, the team went 32 - 6 and won the ACC conference tournament, later advancing to the elite 8, where the fighting irish lost on a missed buzzer-beater against then undefeated kentucky. led by nba draft picks jerian grant and pat connaughton, the fighting irish beat the eventual national champion duke blue devils twice during the season. the 32 wins were [SEP]
slice: 1
[CLS] how many wins does the notre dame men's basketball team have? [SEP] championship. the 2010 – 11 team concluded its regular season ranked number seven in the country, with a record of 25– 5, brey's fifth straight 20-win season, and a second-place finish in the big east. during the 2014-15 season, the team went 32-6 and won the acc conference tournament, later advancing to the elite 8, where the fighting irish lost on a missed buzzer-beater against then undefeated kentucky. led by nba draft picks jerian grant and pat connaughton, the fighting irish beat the eventual national champion duke blue devils twice during the season. the 32 wins were the most by the fighting irishish team since 1908 - 09. [SEP]

Since we have sliced ​​the long text, we need to find the answer position again (relative to the beginning of each context). The machine question answering model will use the answer position (the start and end position of the answer, start and end) as the training label (instead of the token IDS of the answer). So the slice needs to have a corresponding relationship with the original input, the position of each token in the context after slicing and the position in the original long context. In the tokenizer, you can use the `return_offsets_mapping` parameter to get the map of this correspondence:

```python
tokenized_example = tokenizer(
example["question"],
example["context"],
max_length=max_length,
truncation="only_second",
return_overflowing_tokens=True,
return_offsets_mapping=True,
stride=doc_stride
)
# Print the correspondence between the position subscripts before and after the slice
print(tokenized_example["offset_mapping"][0][:100])
```

[(0, 0), (0, 3), (4, 8), (9, 13), (14, 18), (19, 22), (23, 28), (29, 33), (34, 37), (37, 38), (38, 39), (40, 50), (51, 55), (56, 60), (60, 61), (0, 0), (0, 3), (4, 7), (7, 8), (8, 9), (10, 20), (21, 25), (26, 29), (30, 34), (35, 36), (36, 37), (37, 40), (41, 45), (45, 46), (47, 50), (51, 53), (54, 58), (59, 61), (62, 69), (70, 73), (74, 78), (79, 86), (87, 91), (92, 96), (96, 97), (98, 101), (102, 106), (107, 115), (116, 118), (119, 121), (122, 126), (127, 138), (138, 139), (140, 146), (147, 153), (154, 160), (161, 165), (166, 171), (172, 175), (176, 182), (183, 186), (187, 191), (192, 198), (199, 205), (206, 208), (209, 210), (211, 217), (218, 222), (223, 225), (226, 229), (230, 240), (241, 245), (246, 248), (248, 249), (250, 258), (259, 262), (263, 267), (268, 271), (272, 277), (278, 281), (282, 285), (286, 290), (291, 301), (301, 302), (303, 307), (308, 312), (313, 318), (319, 321), (322, 325), (326, 330), (330, 331), (332, 340), (341, 351), (352, 354), (355, 363), (364, 373), (374, 379), (379, 380), (381, 384), (385, 389), (390, 393), (394, 406), (407, 408), (409, 415), (416, 418)]
[0, 0]

The above prints the positions of the first 100 tokens of the 0th slice of tokenized_example in the original context slice. Note that the first token is `[CLS]` set to (0, 0) because this token is not part of the question or answer. The start and end positions of the second token are 0 and 3. We can convert the corresponding token according to the token id after slicing; then use the `offset_mapping` parameter to map back to the token position before slicing to find the tokens at the original position. Since the question is concatenated before the context, we can just find it directly from the question according to the subscript.

```python
first_token_id = tokenized_example["input_ids"][0][1]
offsets = tokenized_example["offset_mapping"][0][1]
print(tokenizer.convert_ids_to_tokens([first_token_id])[0], example["question"][offsets[0]:offsets[1]])
```

how How

Therefore, we get the position correspondence before and after the slice. We also need to use the `sequence_ids` parameter to distinguish between question and context.

```python
sequence_ids = tokenized_example.sequence_ids()
print(sequence_ids)
```

[None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, None]

`None` corresponds to special tokens, and 0 or 1 represents the first and second texts. Since we pass qeustin as the first and context as the second, they correspond to question and context respectively. Finally, we can find the position of the annotated answer in the features after preprocessing:

```python
answers = example["answers"]
start_char = answers["answer_start"][0]
end_char = start_char + len(answers["text"][0])

# Find the Start token index of the current text.
token_start_index = 0
while sequence_ids[token_start_index] != 1:
token_start_index += 1

# Find the End token idnex of the current text.
token_end_index = len(tokenized_example["input_ids"][0]) - 1
while sequence_ids[token_end_index] != 1:
token_end_index -= 1

# Check if the answer is outside the text interval. In this case, it means that the data of the sample is marked at the CLS token position.
offsets = tokenized_example["offset_mapping"][0]
if (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
# Move token_start_index and token_end_index to both sides of the answer position.
# Note: The answer is at the end of the boundary condition.
while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
token_start_index += 1
start_position = token_start_index - 1while offsets[token_end_index][1] >= end_char:
token_end_index -= 1
end_position = token_end_index + 1
print("start_position: {}, end_position: {}".format(start_position, end_position))
else:
print("The answer is not in this feature.")
```

start_position: 23, end_position: 26

We need to verify the position of the answer. The verification method is: use the answer position index, get the corresponding token ID, then convert it into text, and then compare it with the original answer.

```python
print(tokenizer.decode(tokenized_example["input_ids"][0][start_position: end_position+1]))
print(answers["text"][0])
```

over 1, 600
over 1,600

Sometimes question is concatenated with context, and sometimes context is concatenated with question. Different models have different requirements, so we need to use the `padding_side` parameter to specify it.

```python
pad_on_right = tokenizer.padding_side == "right" #context is on the right
```

Now, merge all the steps together. For the case where there is no answer in the context, we directly place the start and end positions of the annotated answer at the subscript of CLS. If the `allow_impossible_answers` parameter is `False`, then these unanswered samples will be thrown away. For simplicity, we throw them away first.

```python
def prepare_train_features(examples):
# We need to truncate and pad the examples and keep all the information, so we need to use the slicing method.
# Each super long text example will be sliced ​​into multiple inputs, withThere will be intersections.
tokenized_examples = tokenizer(
examples["question" if pad_on_right else "context"],
examples["context" if pad_on_right else "question"],
truncation="only_second" if pad_on_right else "only_first",
max_length=max_length,
stride=doc_stride,
return_overflowing_tokens=True,
return_offsets_mapping=True,
padding="max_length",
)

# We use the overflow_to_sample_mapping parameter to map the slice ID to the original ID.
# For example, if 2 examples are cut into 4 slices, then the corresponding ones are [0, 0, 1, 1], the first two pieces correspond to the original first example.
sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
# offset_mapping also corresponds to 4 pieces
# The offset_mapping parameter helps us map to the original input. Since the answer is marked on the original input, it helps us find the starting and ending positions of the answer.
offset_mapping = tokenized_examples.pop("offset_mapping")

# Re-label the data
tokenized_examples["start_positions"] = []
tokenized_examples["end_positions"] = []

for i, offsets in enumerate(offset_mapping):
# Process each piece
# Mark the samples without answers on CLS
input_ids = tokenized_examples["input_ids"][i]
cls_index = input_ids.index(tokenizer.cls_token_id)

# Distinguish question and context
sequence_ids = tokenized_examples.sequence_ids(i)

# Get the original example index.
sample_index = sample_mapping[i]
answers = examples["answers"][sample_index]
# If there is no answer, use the position of CLS as the answer.
if len(answers["answer_start"]) == 0:
tokenized_examples["start_positions"].append(cls_index)
tokenized_examples["end_positions"].append(cls_index)
else:
# Character level start/end position of the answer.
start_char = answers["answer_start"][0]
end_char = start_char + len(answers["text"][0])

# Find the token level index start.
token_start_index = 0
while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
token_start_index += 1

# Find the token level index end.
token_end_index = len(input_ids) - 1
while sequence_ids[token_end_index] != (1 if pad_on_rightt else 0):
token_end_index -= 1

# Check if the answer exceeds the text length. If it exceeds, CLS index is also used as a marker.
if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
tokenized_examples["start_positions"].append(cls_index)
tokenized_examples["end_positions"].append(cls_index)
else:
# If it does not exceed, find the start and end positions of the answer token. .
# Note: we could go after the last offset if the answer is the lastword (edge ​​case).
while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
token_start_index += 1
tokenized_examples["start_positions"].append(token_start_index - 1)
while offsets[token_end_index][1] >= end_char:
token_end_index -= 1
tokenized_examples["end_positions"].append(token_end_index + 1)

return tokenized_examples
```

The above preprocessing function can process one sample or multiple samples.les. If multiple samples are processed, the returned result list is the result list after multiple samples are preprocessed.

```python
features = prepare_train_features(datasets['train'][:5])
# Process 5 samples
```

Next, all samples in the dataset datasets are preprocessed by using the `map` function to apply the preprocessing function `prepare_train_features` to all samples.

```python
tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)
```

Better yet, the returned result is automatically cached to avoid recalculation the next time it is processed (but also note that if the input is changed, it may be affected by the cache!). The datasets library function will detect the input parameters to determine whether there is a change. If there is no change, the cached data will be used. If there is a change, it will be reprocessed. However, if the input parameters remain unchanged, it is best to clear the cache when you want to change the input. The way to clear is to use `load_from_cache_file=False` parameter. In addition, the `batched=True` parameter used above is a feature of the tokenizer, because it uses multiple threads to process the input in parallel.

## Fine-tuning the model

At present, we have preprocessed the data required for training/fine-tuning, and now we download the pre-trained model. Since we are doing machine question answering tasks, we use this class `AutoModelForQuestionAnswering`. Similar to the tokenizer, the model is also loaded using the `from_pretrained` method.

```python
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```

Downloading: 100%|██████████| 268M/268M [00:46<00:00, 5.79MB/s]
Someweights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForQuestionAnswering: ['vocab_transform.weight', 'vocab_transform.bias', 'vocab_layer_norm.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_projector.bias']
- This IS expected if you are initializing DistilBertForQuestionAnswering from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model froma BertForPreTraining model).
- This IS NOT expected if you are initializing DistilBertForQuestionAnswering from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of DistilBertForQuestionAnswering were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['qa_outputs.weight', 'qa_outputs.bias']
You should probably TRAIN this model on a downstream task to be able to use it for predictions and inference.

Since our fine-tuning task is a machine question answering task, and we load a pre-trained language model, it will prompt us that some unmatched neural network parameters were thrown away when loading the model (the neural network head of the pre-trained language model was thrown away, and the neural network head of the machine question answering was randomly initialized).

Because of these randomly initialized parameters, we have to re-fine-tune our model on the new dataset.

In order to get a `Trainer` training tool, we need 3 more elements, the most important of which is the training settings/parameters [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments). This training setting contains all the properties that can define the training process. At the same time, it requires a folder name. This folder will be used to save the model and other model configurations.

```python
args = TrainingArgumentsments(
f"test-squad",
evaluation_strategy = "epoch",
learning_rate=2e-5, #learning rate
per_device_train_batch_size=batch_size,
per_device_eval_batch_size=batch_size,
num_train_epochs=3, # training epochs
weight_decay=0.01,
)
```

The `evaluation_strategy = "epoch"` parameter above tells the training code that we will do a validation evaluation per epcoh.

The `batch_size` above is defined before this notebook.

We use a default_data_collator to feed the preprocessed data to the model.

```python
from transformers import default_data_collator

data_collator = default_data_collator
```

During training, we will only calculate the loss. According to the evaluationThe evaluation model will be put in the next section.

Just pass the model, training parameters, data, previously used tokenizer, and data delivery tool default_data_collator to Trainer.

```python
trainer = Trainer(
model,
args,
train_dataset=tokenized_datasets["train"],
eval_dataset=tokenized_datasets["validation"],
data_collator=data_collator,
tokenizer=tokenizer,
)
```

Call the `train` method to start training

```python
trainer.train()
```

Because the training time is very long, if it is trained on a local mac, each epcoh takes about 2 seconds to disappear, so save the following model after each training.

```python
trainer.save_model("test-squad-trained")
```

## Evaluation

Model evaluation will be slightly more complicated. We need to save the output of the modelPost-processed into the text format we need. The model itself predicts the logits at the start/end of the answer. If we feed the model a batch during evaluation, the output is as follows:

```python
import torch

for batch in trainer.get_eval_dataloader():
break
batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
with torch.no_grad():
output = trainer.model(**batch)
output.keys()
```

The output of the model is a dict-like data structure, which contains loss (because the label is provided, all loss), answer start and end logits. When we output the prediction results, we don't need to look at the loss, just look at the logits directly.

```python
output.start_logits.shape, output.end_logits.shape
```

(torch.Size([16, 384]), torch.Size([16, 384]))

Each token in each feature will have a logit. The simplest way to predict the answer is to select the largest subscript in the start logits as the actual position of the answer, and the largest subscript in the end logits as the end position of the answer.

```python
output.start_logits.argmax(dim=-1), output.end_logits.argmax(dim=-1)
```

(tensor([ 46, 57, 78, 43, 118, 15, 72, 35, 15, 34, 73, 41, 80, 91,
156, 35], device='cuda:0'),
tensor([ 47, 58, 81, 55, 118, 110, 75, 37, 110, 36, 76, 53, 83, 94,
158, 35], device='cuda:0'))

The above strategy is good in most cases. However,If our input tells us that we can't find the answer: for example, the position of start is larger than the position of end, or the positions of start and end point to the question.

At this time, the simple method is that we need to continue to choose the second best prediction as our answer. If it doesn't work, look at the third best prediction, and so on.

Since the above method is not easy to find a feasible answer, we need to think of a more reasonable method. We add the logits of start and end to get a new score, and then look at the best `n_best_size` start and end pairs. From the `n_best_size` start and end pairs, the corresponding answer is derived, and then the answer is checked to see if it is valid. Finally, they are sorted according to the score, and the highest score is selected as the answer.

```python
n_best_size = 20
```

```python
import numpy as np

start_logits = output.start_logits[0].cpu().numpy()
end_logits = output.end_logits[0].cpu().numpy()
# Collect the best start and end logits positions:
start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
valid_answers = []
for start_index in start_indexes:
for end_index in end_indexes:
if start_index <= end_index: # If start is small rain, then it is reasonable
valid_answers.append(
{
"score": start_logits[start_index] + end_logits[end_index],
"text":"" # The answer needs to be found according to the token index
}
)
```

Then weSort the valid_answers by score and find the best one. The last step is to check if the text at the start and end positions is in the context instead of the question.

To accomplish this, we need to add the following two pieces of information to the validation features:
- The ID of the example that generated the feature. Since each example may generate multiple features, each feature/slice of features needs to know the example they correspond to.
- Offset mapping: Map the position of the tokens of each slice to the original text based on the character subscript position.

So we reprocessed the following validation set. It is slightly different from the `prepare_train_features` when processing training.

```python

def prepare_validation_features(examples):
# Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
# in one example possible giving several features when a context is long, each of those features having a
# context that overlaps a bit the context of the previous feature.
tokenized_examples = tokenizer(
examples["question" if pad_on_right else "context"],
examples["context" if pad_on_right else "question"],
truncation="only_second" if pad_on_right else "only_first",
max_length=max_length,
stride=doc_stride,
return_overflowing_tokens=True,
return_offsets_mapping=True,
padding="max_length",
)

# Since one example might give us several features if it has a long context, we need a map from a feature to
# its corresponding example. This key gives us just that.
sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

# We keep the example_id that gave us this feature and we will store the offset mappings.
tokenized_examples["example_id"] = []

for iin range(len(tokenized_examples["input_ids"])):
# Grab the sequence corresponding to that example (to know what is the context and what is the question).
sequence_ids = tokenized_examples.sequence_ids(i)
context_index = 1 if pad_on_right else 0

# One example can give several spans, this is the index of the example containing this span of text.
sample_index = sample_mapping[i]
tokenized_examples["example_id"].append(examples["id"][sample_index])# Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
# position is part of the context or not.
tokenized_examples["offset_mapping"][i] = [
(o if sequence_ids[k] == context_index else None)
for k, o in enumerate(tokenized_examples["offset_mapping"][i])
]

return tokenized_examples
```

Apply the `prepare_validation_features` function to each validation set example as before.

```python
validation_features = datasets["validation_features"].on"].map(
prepare_validation_features,
batched=True,
remove_columns=datasets["validation"].column_names
)
```

HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))

Use `Trainer.predict` method to get all prediction results

```python
raw_predictions = trainer.predict(validation_features)
```

This `Trainer` *hides* some attributes that are not used during model training (here are `example_id` and `offset_mapping`, which will be used in post-processing), so we need to set them back:

```python
validation_features.set_format(type=validation_features.format["type"], columns=list(vvalidation_features.features.keys()))
```

When a token position corresponds to the question part, the `prepare_validation_features` function sets the offset mappings to `None`, so we can easily determine whether the token is in the context based on the offset mapping. We also avoid throwing away very long answers.

```python
max_answer_length = 30
```

```python
start_logits = output.start_logits[0].cpu().numpy()
end_logits = output.end_logits[0].cpu().numpy()
offset_mapping = validation_features[0]["offset_mapping"]
# The first feature comes from the first example. For the more general case, we will need to be matchthe example_id to
# an example index
context = datasets["validation"][0]["context"]

# Gather the indices the best start/end logits:
start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
valid_answers = []
for start_index in start_indexes:
for end_index in end_indexes:
# Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
# to part of the input_ids that are not in the context.
if (
start_index >= len(offset_mapping)
or end_index >= len(offset_mapping)
or offset_mapping[start_index] is None
or offset_mapping[end_index] is None
):
continue
# Don't consider answers with a length that is either < 0 or > max_answer_length.
if end_index < start_index or end_index - start_index + 1 > max_answer_length:
continue
if start_index <= end_index: # We need to refine that test to check the answer is inside the context
start_char = offset_mapping[start_index][0]
end_char = offset_mapping[end_index][1]
valid_answers.append(
{
"score": start_logits[start_index] + end_logits[end_index],
"text": context[start_char: end_char]
}
)

valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]
valid_answers
```

[{'score': 16.706663, 'text': 'Denver Broncos'},
{'score': 14.635585,
'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
{'score': 13.234194, 'text': 'Carolina Panthers'},
{'score': 12.468662, 'text': 'Broncos'},
{'score': 11.709289, 'text': 'Denver'},
{'score': 10.397583,
'text': 'Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},{'score': 10.104669,
'text': 'American Football Conference (AFC) champion Denver Broncos'},
{'score': 9.721636,
'text': 'The American Football Conference (AFC) champion Denver Broncos'},
{'score': 9.007437,
'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10'},
{'score': 8.834958,
'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina'},
{'score': 8.38701,
'text': 'Denver Broncos defeated the National Football Conference (NFC)'},
{'score': 8.143825,
'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title.'},
{'score': 8.03359,
'text': 'American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
{'score': 7.832466,
'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title.'},
{'score': 8.03359,
'text': 'American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},l Football Conference (NFC'},
{'score': 7.650557,
'text': 'The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'},
{'score': 7.6060467, 'text': 'Carolina Panthers 24–10'},
{'score': 7.5795317,
'text': 'Denver Broncos defeated the National Football Conference'},
{'score': 7.433568, 'text': 'Carolina'},
{'score': 6.742434,
'text': 'Carolina Panthers 24–10 to earn their third Super Bowl title.'},
{'score': 6.71136,
'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24'}]

```python
Compare the predicted answer with the true answer:
```

```python
datasets["validation"][0]["answers"]
```

{'answer_start': [177, 177, 177],
'text': ['Denver Broncos', 'Denver Broncos', 'Denver Broncos']}

You can see that the model did it right!

As mentioned in the example above, since the first feature must come from the first example, it is relatively easy. For other fearures, we need a mapping map between features and examples. Similarly, since an example may be sliced ​​into multiple features,So we also need to collect all the answers in all features. The following code maps the index of example and the index of features.

```python
import collections

examples = datasets["validation"]
features = validation_features

example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
features_per_example = collections.defaultdict(list)
for i, feature in enumerate(features):
features_per_example[example_id_to_index[feature["example_id"]]].append(i)
```

The post-processing process is basically complete. The last thing is how to solve the situation where there is no answer (when squad_v2=True). The above code only considers the answers in the context, so we also need to map the answers without answers.Collect the prediction scores of the answers (the start and end of the CLStoken corresponding to the prediction of no answer). If an example sample has multiple features, then we also need to predict whether there is no answer in multiple features. So the final score of no answer is the one with the smallest no answer score of all features.

As long as the final score of no answer is higher than the scores of all other answers, then the question is unanswered.

Put everything together:

```python
from tqdm.auto import tqdm

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
all_start_logits, all_end_logits = raw_predictions
# Build a map example to its corresponding features.
example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
features_per_example = collections.defaultdict(list)
for i, feature in enumerate(features):
features_per_example[example_id_to_index[feature["example_id"]]].append(i)

# The dictionaries we have to fill.
predictions = collections.OrderedDict()

# Logging.
print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

# Let's loop over all the examples!
for example_index, example in enumerate(tqdm(examples)):# Those are the indices of the features associated to the current example.
feature_indices = features_per_example[example_index]

min_null_score = None # Only used if squad_v2 is True.
valid_answers = []

context = example["context"]
# Looping through all the features associated to the current example.
for feature_index in feature_indices:
# We grab the predictions of the model for this feature.
start_logits= all_start_logits[feature_index]
end_logits = all_end_logits[feature_index]
# This is what will allow us to map some of the positions in our logits to span of texts in the original
# context.
offset_mapping = features[feature_index]["offset_mapping"]

# Update minimum null prediction.
cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
feature_null_score = start_logits[cls_index] + end_logits[cls_index]
if min_null_score is None or min_null_score < feature_null_score:
min_null_score = feature_null_score

# Go through all possibilities for the `n_best_size` greater start and end logits.
start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
for start_index in start_indexes:
for end_index in end_indexes:
# Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
# to part of the input_ids that are not in the context.
if (
start_index >= len(offset_mapping)
or end_index >= len(offset_mapping)
or offset_mapping[start_index] is None
or offset_mapping[end_index] is None
):continue
# Don't consider answers with a length that is either < 0 or > max_answer_length.
if end_index < start_index or end_index - start_index + 1 > max_answer_length:
continue
start_char = offset_mapping[start_index][0]
end_char = offset_mapping[end_index][1]
valid_answers.append(
{
"score": start_logits[start_index] + end_logits[end_index],
"text": context[start_char: end_char]
}
)

if len(valid_answers) > 0:
best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
else:
# In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
# failure.
best_answer = {"text": "", "score": 0.0}

# Let's pick our final answer: the best one or the null answer (only for squad_v2)
if not squad_v2:
predictions[example["id"]] = best_answer["text"]
else:
answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
predictions[example["id"]] = answer

return predictions
```

Apply the postprocessing function to the original predictions:

```python

```

```python
final_predictions = postprocess_qa_predictions(datasets["validation"], validation_features, raw_predictions.predictions)
```

Post-processing 10570 example predictions split into 10784 features.

HBox(children=(FloatProgress(value=0.0, max=10570.0), HTML(value='')))

Then we load the metrics:

```python
metric = load_metric("squad_v2" if squad_v2 else "squad")
```

Then we calculate the metrics based on the predictions and annotations. For a reasonable comparison, we need to format the predictions and annotations. For squad2, the metrics also need the `no_answer_probability` parameter (since there is no answer, it is directly set to an empty string, so this parameter is directly set to 0.0 here)

```python
if squad_v2:
formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()]
else:
formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
metric.compute(predictions=formatted_predictions, references=references)
```

{'exact_match': 76.74550614947965, 'f1': 85.13412652023338}

Finally, don't forget to [check how to upload the model](https://huggingface.co/transformers/model_sharing.html) and upload the model to [🤗 Model Hub](https://huggingface.co/models). You can then use your model by name, just like at the beginning of this notebook.

```python

```