The reference material for this article is the course under Resources on the [Hugging Face homepage](https://huggingface.co/), which excerpts and annotates (in bold italics), and also adds an introduction to the main parameters of Trainer and args. Interested students can check out the [original text](https://huggingface.co/course/chapter1).
****
The main content of this chapter includes two parts:
- Pipeline tool demonstrates NLP task processing
- Build Trainer fine-tuning model<br>

Contents
- [1. Introduction](#1--Introduction)
- [History of Transformers](#History of Transformers)
- [Architectures and checkpoints](#architectures and checkpoints)
- [The Inference API](#the-inference-api)
- [2. Using pipeline to deal with NLP problems](#2-Using pipeline to deal with nlp problems)
- [3. Behind the pipeline](#3-behind-the-pipeline)
- [Tokenizer pre-processing](#3-behind-the-pipeline)=Processing](#tokenizer preprocessing)
- [Select a model](#Select a model)
- [Model heads](#model-heads)
- [Post-processing](#post-processing)
- [4. Build Trainer API to fine-tune the pre-trained model](#4-Build trainer-api to fine-tune the pre-trained model)
- [Download dataset from Hub](#Download dataset from Hub)
- [Preprocessing dataset](#Preprocessing dataset)
- [Fine-tune in PyTorch using Trainer API](#Fine-tune in PyTorch using -trainer-api-)
- [Training](#Training)
- [Evaluation function](#Evaluation function)
- [5. Supplementary section](#5-Supplementary section)
- [Why does the fourth chapter of the tutorial use Trainer to fine-tune the model? ](#Why does the fourth chapter of the tutorial use trainer to fine-tune the model)
- [TrainingArguments main parameters](#trainingarguments main parameters)
- [Different model loading methods](#Different model loading methods)
- [Dynamic padding——dynamic padding technology](#dynamic-paddingdynamic padding technology)

## 1. Introduction
This chapter will use [the library in the Hugging Face ecosystem](https://github.com/huggingface)——🤗 Transformers to perform natural language processing (NLP).
### History of Transformers
Here are some reference points in the (brief) history of the Transformer model:
![transformers_chrono](https://img-blog.csdnimg.cn/3ba51fe4f21d4d528ca7b0f2fd78aee4.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
[Transformer architecture](https://arxiv.org/abs/1706.03762) was published in June 2017The original research focused on translation tasks. Several influential models followed, including:

- June 2018: [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), the first pre-trained Transformer model that was fine-tuned for various NLP tasks and achieved state-of-the-art results

- October 2018: [BERT](https://arxiv.org/abs/1810.04805), another large pre-trained model designed to generate better sentence summaries (more on this in the next chapter!)

- February 2019: [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), an improved (and larger) version of GPT that was not immediately released publicly due to ethical concerns

- October 2019: [BERT](https://arxiv.org/abs/1810.04805), another large pre-trained model that was designed to generate better sentence summaries (more on this in the next chapter!) Month: [DistilBERT](https://arxiv.org/abs/1910.01108), a distilled version of BERT that is 60% faster and uses 40% less memory, but still retains 97% of BERT’s performance
- October 2019: [BART](https://arxiv.org/abs/1910.13461) and [T5](https://arxiv.org/abs/1910.10683), two large pre-trained models that use the same architecture as the original Transformer model (the first to do so)
- May 2020: [GPT-3](https://arxiv.org/abs/2005.14165), a larger version of GPT-2 that performs well on a variety of tasks without fine-tuning (called zero-shot learning)

This list is not exhaustive, but is meant to highlight some of the different types of Transformer models. Generally speaking, they can be divided into three categories:

- GPT (only using the transformer-decoder part, autoregressive transformer model)
- BERT (only using the transformer-encoder part, autoencoding transformer model)
- BART/T5 (Transformer-encoder-decoder model）

### Architectures and checkpoints
In the study of Transformer models, some terms will appear: architecture Architecture and checkpoint checkpoint and model. These terms have slightly different meanings:

Architecture: defines the basic structure and basic operations of the model

Checkpoint: a training state of the model, loading this checkpoint will load the weights at this time. (You can choose to automatically save the checkpoint during training)

Model: This is a general term, not as precise as "architecture" or "checkpoint", it can mean both at the same time. When it is necessary to reduce ambiguity, this course will specify architecture or checkpoint. <br>
For example, BERT is an architecture, and bert-base-cased (a set of weights trained by the Google team for the first version of BERT) is a checkpoint. However, you can say "the BERT model" and "the bert-base-cased model".

***The concept of checkpoint is more mentioned in big data. When training, the model can be set to automatically save at a certain point in time (for example, the model has trained for one epoch)., updated the parameters, and saved the model in this state as a checkpoint. )
So each checkpoint corresponds to a state of the model and a set of weights. In big data, a checkpoint is a database event, and its fundamental purpose is to reduce crash time. That is, to reduce the time to recover after a database crash due to an unexpected situation. ***
### The Inference API
[Model Hub](https://huggingface.co/models) contains checkpoints for multilingual models. You can refine your search for models by clicking on the language label and then select a model that generates text in another language. 

After clicking to select a model, you will see a widget - Inference API (supports online trial). That is, you can use various models directly on this page, and you can see the results of the model processing the input data by entering custom text. In this way, you can quickly test the model's functionality before downloading it.
![DistilBERT base model (uncased)](https://img-blog.csdnimg.cn/0edebca3ab8248b4b2bac88f88ab79c0.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
## 2. Using pipelines for NLP problems
In this section, we’ll look at what Transformer models can do and use the first tool in the 🤗 Transformers library: pipelines.

>🤗 The [Transformers library](https://github.com/huggingface/transformers) provides the ability to create and use shared models. [Model Hub](https://huggingface.co/models) contains thousands of pre-trained models that anyone can download and use. You can also upload your own models to the Hub!

🤗 The most basic object in the Transformers library is the pipeline. It connects the model with its necessary pre- and post-processing steps, allowing us to directly input any text and get an understandable answer:
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```
```python
[{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```
We can even pass in a few sentences!
```python
classifier([
"I've been waiting for a HuggingFace course my whole life.", 
"I hate this so much!"
])
```
```python
[{'label': 'POSITIVE', 'score': 0.9598047137260437},
{'label': 'NEGATIVE', 'score': 0.9994558095932007}]
```
By default, this pipeline selects a specific pre-trained model that has been fine-tuned for English sentiment analysis. When creating a classifier object, pass the followingDownload and cache the model. If you rerun the command, the cached model will be used and there is no need to download the model again.

There are three main steps involved when passing some text to a pipeline:

1. Preprocessing: The text is preprocessed into a format that the model can understand.

2. Input model: The model is built and the preprocessed input is passed to the model.

3. Postprocessing: The predictions of the model are postprocessed so you can understand them.

Some of the pipelines currently available are:

- feature-extraction (get vector representation of text)

- fill-mask fills in the blanks in a given text (cloze)

- ner (named entity recognition) part-of-speech tagging

- question-answering

- sentiment-analysis

- summarization

- text-generation

- translation

- zero-shot-classification

You can also select pipelines for specific models from the Hub for specific tasks, for example, text generation. Go to [Model Hub](https://huggingface.co/models) and click on the corresponding tab on the left, the page will only show the text generation taskThe model supported by the service.
(***In addition to matching the model to the task, one of the further considerations is that the dataset used when the pre-trained model is trained should be as close as possible to the dataset contained in the task you need to handle. The closer the two datasets are, the better.***) 

The Transformers pipeline API can handle different NLP tasks. You can use the full architecture or just the encoder or decoder, depending on the type of task you want to solve. The following table summarizes this:

Model | Example | Task
-------- | ----- |----- 
Encoder | ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa | Sentence classification, named entity recognition, extractive question answering
Decoder | CTRL, GPT, GPT-2, Transformer XL | Text generation
Encoder-decoder | BART, T5, Marian, mBART | Summary generation, translation, generative question answering

The pipelines shown above are mainly for demonstration purposes. They are programmed for specific tasks and cannot execute their variants. In the next section, you will learn what is inside the pipeline and how to customize its behavior.
>Simple examples of the above pipelines can be viewed at -—[Hugging Face Homepage Course 1: Transformer models](https://blog.csdn.net/qq_56591814/article/details/120124306).
>Or click [Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/chapter1/section3.ipynb) to open a Google Colab notebook containing other pipeline application code examples.
If you want to run the examples locally, we recommend that you check out [Settings](https://huggingface.co/course/chapter0).
## 3. Behind the pipeline
>Code for this section: [Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/chapter2/section2_pt.ipynb) (PyTorch)<br>
>YouTube video: [what happend inside the pipeline function](https://youtu.be/1pedAIvTWXk)

Let’s start with a complete example and see what happens behind the scenes when we execute the following code in Section 1:
```python

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier([
"I've been waiting for a HuggingFace course my whole life.", 
"I hate this so much!",
])
```
```python
[{'label': 'POSITIVE', 'score': 0.9598047137260437},
{'label': 'NEGATIVE', 'score': 0.9994558095932007}]
```
As we saw in Chapter 1, this pipeline combines three steps: preprocessing, passing the input through the model, and postprocessing:
![full_nlp_pipeline ](https://img-blog.csdnimg.cn/7f19b775bfe94fa0bb9e35d883567e16.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

Let’s take a quick look at these.
### Tokenizer Preprocessing
Like other neural networks, Transformer models cannot process raw text directly, so the first step in our pipeline is to convert the text input into numbers that the model can understand. To do this, we use a tokenizer, which will be responsible for:

- Splitting the input into words, subwords, or symbols (such as punctuation marks) called tokens
- Mapping each token to an integer
- Adding other inputs that may be useful to the model

Using the AutoTokenizer class and its from_pretrained method, we can ensure that all this preprocessing is done in exactly the same way as when the model was pretrained.Complete. Set the checkpoint name of the model, which will automatically fetch the data associated with the model's Tokenizer and cache it (so it is only downloaded the first time you run the code below).

Since the default checkpoint for the sentiment analysis pipeline is [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english), we can run the following command to get the tokenizer we need:
```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
```python
raw_inputs = [
"I've been waiting for a HuggingFace course my whole life.", 
"I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
#return_tensors="pt" means returning Pytorch tensors. After the text is converted to numbers, it must be converted to tensors before it can be input into the model.
#padding=True means padding the input sequence to the maximum length, truncation=True means that the long sequence is truncated

print(inputs)

```
The following is the result of PyTorch tensor:

```python
{
'input_ids': tensor([
[ 101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102],
[ 101, 1045, 5223, 2023, 2061, 2172, 999, 102,0, 0, 0, 0, 0, 0, 0, 0]
]), 
'attention_mask': tensor([
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
])
}
```
### Choosing a model
We can download our pre-trained model just like we did with the tokenizer. 🤗 Transformers provides an AutoModel class, which also has a from_pretrained method:
```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```
The AutoModel class and all its related classes are actually the various available models in the library.A simple wrapper around the AutoModel class. It automatically guesses a suitable model architecture for your checkpoint and then instantiates a model using that architecture. (***The AutoModel class can instantiate any model from a checkpoint, and this is a better way to instantiate a model. There is another way to build a model, which is at the end of the article.***)

In this code snippet, we download the same checkpoint we used before in the pipeline (it should actually have been cached) and instantiate a model with it. But this architecture only contains the basic Transformer module: given some input, it outputs what we call hidden states. While these hidden states are useful in themselves, they are usually input to another part of the model (the model head).
### Model heads
We can use the same model architecture for different tasks, but each task has different model heads associated with it.

Model heads: take as input the high-dimensional vectors of hidden states (aka logits vectors) and project them onto different dimensions. They usually consist of one or several linear layers:
![transformer_and_head](https://img-blog.csdnimg.cn/9156c4e09d184e7c88732e60ae59e05e.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
In this diagram, the model is represented by its embeddings layer and subsequent layers. The input data passes through the embeddings layer and outputs the logits vector to produce the final representation of the sentence.

🤗 There are many different architectures available in Transformers, each designed around handling specific tasks. Some of the Model heads are listed below:
* Model (retrieve the hidden states)
* ForCausalLM
* ForMaskedLM
* ForMultipleChoice
* ForQuestionAnswering
* ForSequenceClassification
* ForTokenClassification
* and others 🤗

Take sentiment classification as an example, we needA Model head with sequence classification (able to classify sentences as positive or negative). Therefore, we will not actually use the AutoModel class, but AutoModelForSequenceClassification:

(***That is to say, the model = AutoModel.from_pretrained(checkpoint) written earlier cannot get the results of the sentiment classification task because the Model head is not loaded***)
```python
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
```
The model head takes the high-dimensional vector we saw before as input and outputs a vector containing two values ​​(one for each label):
```python 
print(outputs.logits.shape)
```
```python
torch.Size([2, 2])
```
Since we only have two sentences and two labels, the result we get from the model is of shape 2 x 2.
### Post-processing
The values ​​we get as outputs from the model don’t necessarily make sense on their own. Let’s take a look:
```python 
print(outputs.logits)
```
```python 
tensor([[-1.5607, 1.6123],
[ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
```
Our model predicted the first sentence result [-1.5607, 1.6123] and the second sentence result [4.1692, -3.3464]. These are not probabilities, but logits, the raw, unnormalized scores output by the last layer of the model. To convert them to probabilities, they need to go through a SoftMax layer. All 🤗 Transformers models output logits, because the training loss function generally combines the last activation function (such as SoftMax) with the actual cross entropy loss function. <br>
(***Supplement: In Pytorch, cross entropy loss CEloss is not the mathematical cross entropy loss (NLLLoss). Pytorch's CrossEntropyLoss is to merge Softmax–Log–NLLLoss into one step. For details, please refer to the Zhihu article ["How to understand NLLLoss?"] (https://zhuanlan.zhihu.com/p/30187567)***）

```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```
```python
tensor([[4.0195e-02, 9.5980e-01],
[9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
```
This time the output is a recognizable probability score.

To get the label corresponding to each position, we can check the id2label property of the model configuration:
```python
model.config.id2label
```
```python
{0:'NEGATIVE', 1: 'POSITIVE'}
```
Now we can conclude that the model predicts the following:

First sentence: NEGATIVE: 0.0402, POSITIVE: 0.9598<br>
Second sentence: NEGATIVE: 0.9995, POSITIVE: 0.0005

## 4. Build Trainer API to fine-tune the pre-trained model
>Code for this section: [Open in Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/chapter3/section3.ipynb) (PyTorch), it is recommended to click here for testing. Colab loads datasets very quickly, and training is also relatively fast after setting up GPU.
>After opening, select the "Modify" tab in the upper left corner, click Notebook Settings-Hardware Accelerator None to GPU.

In Section 3, we explored how to use the word segmenter and pre-trained model for prediction. But what if you want to fine-tune a pre-trained model for your own dataset? That’s what this chapter is about! You’ll learn:

- How to prepare large datasets from Model Hub
- How to use the high-level TrainerAPI to fine-tune the model
- How to use a custom training loop
- How to leverage the 🤗 Accelerate library to easily run the custom training loop on any distributed setup
### Download the dataset from the Hub
>Youtube video: [Hugging Face Datasets Overview](https://youtu.be/_BZearw7f0w) (pytorch)

The Hub contains more than just models; it also contains multiple [datasets](https://huggingface.co/datasets) in many different languages. We recommend that you try loading and processing new datasets after completing this section (see the [docs](https://huggingface.co/docs/datasets/loading_datasets.html#from-the-huggingface-hub)). 

The MRPC dataset is one of the 10 datasets that make up the [GLUE benchmark](https://gluebenchmark.com/). The GLUE benchmark is an academic benchmark that measures the performance of ML models on 10 different languages.Performance in this classification task.

🤗 The Datasets library provides a very simple command to download and cache datasets on the Hub. We can download the MRPC dataset like this:

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

```python
DatasetDict({
train: Dataset({
features: ['sentence1', 'sentence2', 'label', 'idx'],
num_rows: 3668
})
validation: Dataset({
features: ['sentence1', 'sentence2', 'label', 'idx'],
num_rows: 408
})
test: Dataset({
features: ['sentence1', 'sentence2', 'label', 'idx'],
num_rows: 1725
})
})
```
This will give you a DatasetDict object containing the training set, validation set, and test set. There are 3,668 sentence pairs in the training set, 408 pairs in the validation set, and 1,725 ​​pairs in the test set. Each sentence pair contains four columns of data: 'sentence1', 'sentence2', 'label', and 'idx'.

load_dataset method, you can build datasets from different places<br>
- from the HuggingFace Hub,
- from local files, such as CSV/JSON/text/pandas files
- from in-memory data like python dict or a pandas dataframe.

For example: datasets = load_dataset("text", data_files={"train": path_to_train.txt, "validation": path_to_validation.txt} For details, please refer to the [document](https://link.zhihu.com/?target=https%3A//huggingface.co/docs/datasets/loading_datasets.html%23from-local-files)

The load_dataset command downloads and caches the dataset, by default in ~/.cache/huggingface/dataset . You can customize the cache folder by setting the HF_HOME environment variable.

Like dictionaries, raw_datasets can access sentence pairs by index:

```python
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
```

```python
{'idx': 0,
'label': 1,
'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .'}
```
```python
import pandas as pd
validation=pd.DataFrame(raw_datasets['validation'])
validation
```
![validation](https://img-blog.csdnimg.cn/df0adbee66ba40dc862e64f5df5e0022.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
It can be seen that the label is already an integer and no further preprocessing is required. The type of each column can be known through the features attribute of raw_train_dataset:
```python
raw_train_dataset.features
```

```python
{'sentence1': Value(dtype='string', id=None),
'sentence2': Value(dtype='string', id=None),
'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
'idx': Value(dtype='int32', id=None)}
```
label is of ClassLabel type, label=1 means that the pair of sentences are paraphrases to each other, label=0 means that the sentence pair has inconsistent meanings.

### Dataset preprocessing
>YouTube video [《Preprocessing sentence pairs》](https://youtu.be/0u3ioSwev3s)

The tokenizer can be used to convert text into numbers that the model can understand.

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
Let's look at an example:
```python
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs
```

```python
{ 'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```
So by passing the list of sentence pairs to the tokenizer, we can perform tokenization on the entire dataset. Therefore, one way to preprocess the training dataset is:```python
tokenized_dataset = tokenizer(
raw_datasets["train"]["sentence1"],
raw_datasets["train"]["sentence2"],
padding=True,
truncation=True,
)
```
This processing method is ok, but the disadvantage is that after processing, tokenized_dataset is no longer in a dataset format, but returns a dictionary (with our keys: input_ids, attention_mask and token_type_ids, and the corresponding key-value pairs). And once our dataset is too large to fit in memory, this approach will cause an Out of Memory exception. (🤗 The datasets in the Datasets library are Apache Arrow files stored on disk, so the samples requested to be loaded are all stored in memory).

In order to keep our data in the format of the dataset, we will use the more flexible Dataset.map method. This method can complete more preprocessing than just tokenization. The map method is to apply each element in the dataset to the map.We use the same function, so let's define a function to tokenize the input:

```python
def tokenize_function(example):
return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```
This function takes a dictionary (like the items in our dataset) and returns a dictionary (with three keys: input_ids, attention_mask, and token_type_ids). 

The padding parameter is omitted in the tokenization function because it is more efficient to pad to the maximum length in the batch than to pad all sequences to the maximum sequence length of the entire dataset. This can save a lot of time and processing power when the input sequence lengths are very inconsistent!

Below is the tokenization method applied to the entire dataset. We used batched=True in the map call, so the function is applied to the entire batch of elements of the dataset at once, rather than to each element separately. This makes the preprocessing faster (because 🤗 T in the Tokenizers libraryThe okenizer is written in Rust and can be very fast when processing a lot of input at once).

```python
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```
🤗 The way the Datasets library applies this processing is by adding a new field to the dataset, like this:
```python
DatasetDict({
train: Dataset({
features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
num_rows: 3668
})
validation: Dataset({
features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
num_rows: 3668
})s'],
num_rows: 408
})
test: Dataset({
features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
num_rows: 1725
})
})
```
>If you are not using the fast tokenizer supported by this library, you can set the num_proc parameter for multi-threaded processing when preprocessing the Dataset.map function to speed up preprocessing.

Finally, when we batch the input sequence, we need to pad all input sequences to the length of the longest sequence in this batch - we call it dynamic padding (dynamic padding: padding the input sequence of each batch to the same length. The specific content is at the end).
### Fine-tuning in PyTorch using the Trainer API
Since PyTorch does not provide a wrapped training loop, the 🤗 Transformers library has written a transformers.Trainer API, which is a simple butA full-featured PyTorch training and evaluation loop optimized for 🤗 Transformers, with many training options and built-in functions, and also supports multi-GPU/TPU distributed training and mixed precision. That is, the Trainer API is a packaged trainer (a small framework built into the Transformers library, or TFTrainer for Tensorflow).

However, Trainer did not exist at the beginning (it did not exist in early versions), and because a lot of parameters are required to start training, and each NLP task has many common parameters, these are abstracted out by Trainer. For a more specific understanding, you can look at the original version of [Trainer code](https://github.com/huggingface/transformers/blob/master/examples/pytorch/multiple-choice/run_swag.py) written by Duoduo. Trainer just merges the parameters needed before training starts.

After the data is preprocessed, it only takes a few simple steps to define the parameters of the Trainer and then you can perform the basic training loop of the model (otherwise, you have to load and preprocess the data from scratch, set various parameters, and write the training loop step by step. The content of custom training loop is at the end of this section).The hardest part of Trainer is probably preparing the environment to run Trainer.train, as it runs very slowly on a CPU. (If you don’t have a GPU set up, you can access a free GPU or TPU on Google Colab)

The main parameters of trainer are:
- Model: The model to use for training, evaluation, or prediction
- args (TrainingArguments): The parameters to adjust for training. If not provided, it will default to a base instance of TrainingArguments
- data_collator (DataCollator, optional) – A function used to batch train_dataset or eval_dataset
- train_dataset: The training set
- eval_dataset: The validation set
- compute_metrics: A function used to compute evaluation metrics. EvalPrediction must be passed in and will return a dictionary with the key-value pairs being the metric and its value.
- callbacks (callback function, optional): A list of callbacks for customizing the training loop (List of TrainerCallback)
- optimizers: A tuple containing optimizers and learning rate adjustersgroup, the default optimizer is AdamW, and the default learning rate is a linear learning rate from 5e-5 to 0

In addition to the above main parameters, there are some parameters and properties (there must be dozens of them, you can take your time to read them. The complete Trainer documentation can be found here](https://huggingface.co/transformers/main_classes/trainer.html?highlight=trainer#transformers.Trainer))

The following code examples assume that you have executed the examples in the previous section:

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")#MRPC determines whether two sentences are paraphrases to each other
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)# Dynamic padding, that is, padding the input sequence of each batch to the same length
```

#### Training
The first parameter of Trainer is the TrainingArguments class, which is a subset of parameters related to the training loop itself, including all hyperparameters used for training and evaluation in Trainer. The only parameter that must be provided is: the directory where the model or checkpoint is saved, and other parameters can select the default value (such as the default training of 3 epochs, etc.) (TrainingArguments also has dozens of parameters, common parameters are written at the end of the article, and the complete document is included in the Trainer document mentioned above)

```python
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
```
Step 2: Define the model
As in the previous section, we will use the AutoModelForSequenceClassification class with two labels:
(***Actually, select the task head according to your own task for fine-tuning***)
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)#The number of labels is 2, which is binary classification
```
A warning will be reported after instantiating this pre-trained model. This is because BERT has not been pre-trained in sentence pair classification, so the head of the pre-trained model has been discarded, and a new head suitable for sequence classification has been added. The warning indicates that some weights are not used (corresponding to the discarded pre-trained head part), while some other weights are randomly initialized(new head section), and finally you are encouraged to train the model.

Once you have a model, you can define a trainer Trainer and pass it all the objects built so far to fine-tune the model. These objects include: model, training_args, training and validation datasets, data_collator and tokenizer. (These are all parameters of Trainer):

```python
from transformers import Trainer

trainer = Trainer(
model,
training_args,
train_dataset=tokenized_datasets["train"],
eval_dataset=tokenized_datasets["validation"],
data_collator=data_collator,
tokenizer=tokenizer,
)
```
When passing the tokenizer like above, the parameter data_collator is the dynamic padding DataCollatorWithPadding defined previously, so data_collator=data in this callThe ta_collator line can be skipped. (But it was still important to show you this part of the processing in section 2!)

To fine-tune the model on our dataset, we just need to call the train method of Trainer:

```python
trainer.train()
```
Fine-tuning starts (about 6 minutes on colab with GPU), and the training is complete:

```python
The following columns in the training set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence1, sentence2, idx.
***** Running training *****
Num examples = 3668
Num Epochs = 3
Instantaneous batch size per device = 8
Total train batch size (w. parallel, distributed & accumulation) = 8
Gradient Accumulation steps = 1
Total optimization steps = 1377

Step Training Loss
500 0.544700
1000 0.326500

TrainOutput(global_step=1377, training_loss=0.3773723704795865, metrics={'train_runtime': 379.1704, 'train_samples_per_second': 29.021, 'train_steps_per_second': 3.632, 'total_flos': 405470580750720.0, 'train_loss': 0.3773723704795865, 'epoch': 3.0})
#Only the results of 500 steps and 1000 steps are displayed during the run, and the final result is 1377 stepss, the final loss is 0.377
```

We can first look at the structure of the validation set after preprocessing:
```python
tokenized_datasets["validation"]
```
```python
Dataset({
features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
num_rows: 408
})
```
We can use the Trainer.predict command to obtain the prediction results of the model:
```python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```

```python
(408, 2) (408,)
```
predict The method outputs a tuple with three fields.
- predictions: predicted values, shape: [batch_size, num_labels], which is logits instead of the result after softmax
- label_ids: the real label id
- metrics: evaluation indicators, the default is training loss, and some time metrics (total time and average time required for prediction). But once we pass the compute_metrics function to Trainer, the return value of the function will also be output

![mrpc](https://img-blog.csdnimg.cn/7a920b0dddf147cf87b38fb18a0ad0a8.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
```python
metrics={'test_loss': 0.6269022822380066, 'test_runtime': 4.0653, 'test_samples_per_second': 100.362, 'test_steps_per_second': 12.545})
```
predictions is a 2D array with a shape of 408 x 2 (408 validation sets, two labels). To compare the predictions with the labels, we need to take the index of the maximum value on the second axis of predictions:
```python
import numpy as np
preds = np.argmax(predictions.predictions, axis=-1)
```
At the same time, from the training process above, you can see that the model reports the training loss every 500 steps. However, it does not tell you how well the model performs. This is because:
1. The evaluation_strategy parameter is not set to tell the model how many "steps" (eval_steps) or "epochs" to evaluate the loss once.
2. Trainer's compute_metrics can calculate the values ​​of specific evaluation indicators during training (such as acc, F1 score, etc.). Without compute_metrics, only training loss is displayed, which is not an intuitive number.

If we set compute_metricsOnce the function is written and passed to the Trainer, the metrics field will also contain the metric values ​​returned by compute_metrics.
#### Evaluation Function
Now let's see how to construct the compute_metrics function. This function:
- Must pass in the EvalPrediction parameter. EvalPrediction is a tuple with a predictions field and a label_ids field.
- Returns a dictionary with key-value pairs of key: metric name (string type), value: metric value (float type).

***That is what [Tutorial 4.1](https://datawhalechina.github.io/learn-nlp-with-transformers/#/./%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1/4.1-%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB) said: directly call the compute method of metric and pass in labels and predictionsTo get the value of the metric. Only by doing this can we get the results such as acc and F1 during training (the specific indicators are determined according to different tasks)***

To build our compute_metric function, we will rely on the metric in the 🤗 Datasets library. With the load_metric function, we can load the metric associated with the MRPC dataset as easily as loading a dataset. The object returned has a compute method we can use to do the metric calculation:

```python
from datasets import load_metric

metric = load_metric("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```

```python
{'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}#The accuracy of the model on the validation set is 85.78%, and the F1 score is 89.97
```
The random initialization of the model head during each training may change the final metric value, so the final results here may be different from what you run. acc and F1 are two metrics used to evaluate the MRPC dataset results of the GLUE benchmark. The table in the BERT paper reports an F1 score of 88.9 for the base model. That is the un-cased model, and we are currently using the cased model, which explains the better results. (cased means case-sensitive English)

Putting the above together, we get the compute_metrics function:

```python
def compute_metrics(eval_preds):
metric = load_metric("glue", "mrpc")
logits, labels = eval_preds
predictions = np.argmax(logits, axis=-1)
return metric.compute(predictions=predictions, references=labels)
```
Then set each epoch to check the validation evaluation once. So here is how we set compuTrainer after te_metrics parameter:

```python
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```
```python
trainer = Trainer(
model,
training_args,
train_dataset=tokenized_datasets["train"],
eval_dataset=tokenized_datasets["validation"],
data_collator=data_collator,
tokenizer=tokenizer,
compute_metrics=compute_metrics
)
```
Note that we created a new TrainingArguments with evaluationion_strategy is set to "epoch" and a new model - otherwise, we will just continue training the model we have already trained. To start a new training run, we execute:

```python
trainer.train()
```
The training ended up taking 6 minutes and 33 seconds, a little longer than the last time. The final run results are:
```python
The following columns in the training set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence1, sentence2, idx.
***** Running training *****
Num examples = 3668
Num Epochs = 3
Instantaneous batch size per device = 8
Total train batch size (w. parallel, distributed & accumulation) = 8
Gradient Accumulation steps = 1
Total optimization steps = 1377

Epoch Training Loss Validation Loss Accuracy F1
1 No log 0.557327 0.806373 0.872375
2 0.552700 0.458040 0.862745 0.903448
3 0.333900 0.560826 0.867647 0.907850
TrainOutput(global_step=1377, training_loss=0.37862846690325436, metrics={'train_runtime': 393.5652, 'train_samples_per_second': 27.96, 'train_steps_per_second': 3.499, 'total_flos': 405470580750720.0, 'train_loss': 0.37862846690325436, 'epoch': 3.0})
```
This time, the model will report validation loss and metrics at the end of each epoch in addition to training loss. Again, due to the random task head initialization of the model, the accuracy/F1 score you achieve may be slightly different from what we found, but it should be in the same range.

Trainer supports multiple GPUs/TPUs by default, and also supports mixed precision training, which can be set in the training configuration TrainingArguments by setting fp16 = True.

Using Trainer is convenient, but the high-level wrapper API also has its drawbacks, that is, you cannot perform many customized operations. So we can use the regular pytorch training method and customize the training loop. You can also choose to use the Accelerate library for distributed training (the previous examples all use a single GPU/CPU). This part of the content is not required. If you are interested, you can check the original text [《A full training》](https://huggingface.co/course/chapter3/4?fw=pt), or translate [《Fine-tuning the pre-trained model》](https://blog.csdn.net/qq_56591814/article/details/120147114).

## 5. Supplementary section
### Why does the fourth chapter of the tutorial use Trainer to fine-tune the model?

There are two ways to use the pre-trained model:

- Feature extraction (pre-trained models are not trained later and weights are not adjusted)

- Fine-tuning (simply train a few epochs according to downstream tasks and adjust the weights of the pre-trained model)

The fifth part (experiments) of the BERT paper says that although BERT has two methods for NLP tasks, it is not recommended to directly output results for prediction without training the model. In addition, the author of Hugging Face also recommends that you use Trainer to train the model.

In practice, the effect of fine-tuning will be significantly better than feature extraction (unless you are stubborn and follow the feature extraction with a very complex model).

As for why Trainer is used for fine-tuning, it has been mentioned before: Trainer is a PyTorch training and evaluation loop API written specifically for Transformers, which is relatively simple to use. Otherwise, you need to customize the training loop.

This short paragraph is my understanding and is not in the HF homepage course.

### TrainingArguments main parameters
There are dozens of TrainingArguments parameters, and the main ones used in the following chapters are:
- output_dir (str): The directory where model predictions and checkpoints are saved. The saved model can be loaded using pipelines and used in the next prediction. For details, see [《Using huggingface transformers to achieve one-stop BERT training and prediction》](https://zhuanlan.zhihu.com/p/344767513?utm_source=wechat_session&utm_medium=social&utm_oi=1400823417357139968&utm_campaign=shareopn)
- evaluation_strategy : There are three options
- "no": no evaluation during training
- "step": complete (and record) evaluation for each eval_steps
- "epoch": evaluate at the end of each epoch.
- learning_rate (float, optional) – AdamW optimizer learning rate, defaults to 5e-5
- weight_decay (float, optional, defaults to 0): If not 0, the weight decay applied to all layers, except for all biases and LayerNorm weights in the AdamW optimizer. About weFor details about ight decay, please refer to the Zhihu article [It's 9102, don't use Adam + L2 regularization anymore](https://zhuanlan.zhihu.com/p/63982470).
- save_strategy (str or IntervalStrategy, optional, defaults to "steps"): Checkpoint saving strategy used during training. Possible values ​​are:
- "no": no saving during training
- "epoch": save at the end of each epoch
- "steps": save once per step.
- fp16 (bool, optional, defaults to False) – Whether to use 16-bit (mixed) precision training instead of 32-bit training.
- metric_for_best_model (str, optional): Used in conjunction with load_best_model_at_end to specify the metric used to compare two different models. Must be the name of the metric returned by the evaluation, with or without the prefix "eval_".
- num_train_epochs (float, optional, default is 3) – number of epochs to train
- load_best_model_at_end (bool, optional, defaults to False) : Whether to load the best model found during training at the end of training.

### Different ways to load models
The AutoModel class and all its related classes are actually simple wrappers around the various models available in the library. It can automatically guess the appropriate model architecture for your checkpoint and then instantiate a model with that architecture.

However, if you know the type of model you want to use, you can directly use the class that defines its architecture. Let's see how it works with a BERT model.

The first thing you need to do to initialize a BERT model is to load the configuration object:
```python
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)
```
The config configuration contains a number of properties for building a model:
```python
print(config)
```
```python
BertConfig {
[...]
"hidden_size": 768, #The size of the hidden_states vector
"intermediate_size": 3072, #The number of neurons in the first layer of FFN, that is, the attention layer will expand the dimension by 4 times when it is passed to the first layer of full connection
"max_position_embeddings": 512,#Maximum sequence length 512
"num_attention_heads": 12,
"num_hidden_layers": 12,
[...]
}
```
hidden_size: The size of the hidden_states vector<br>
num_hidden_layers: The number of layers of the Transformer model

Creating a model from the default configuration will initialize it with random values:
```python
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)

# The model has been randomly initialized
```
The model can be used in this state, but it will output garbled characters; it needs to be trained first. We can train the model from scratch based on the task at hand, which will take a long time andA lot of data.

Use the from_pretrained method to load a pre-trained Transformer model:
```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```
As you saw earlier, we can replace BertModel with the AutoModel class and the effect is the same. We will use the AutoModel class later. The advantage of this is that the part that sets the model structure does not affect the checkpoint. If your code works for one checkpoint, it can also work for another checkpoint. Even if the model structure is different, as long as the checkpoint is trained for a similar task, it also works.

***Using the AutoModel class, passing in different checkpoints, you can implement different models to handle the task (as long as the output of this model can handle this task). If you choose BertModel, the model structure is fixed. ***

In the code example above, we did not use BertConfig (BertConfig is an initialized model without any training), but instead passed the identifier ""bert-base-cased" loads a checkpoint of a pre-trained model, trained by the authors of BERT themselves. You can find more details about it in its [model card](https://huggingface.co/bert-base-cased).

The model is now initialized with all the weights of the checkpoint. It can be used directly for inference on the task it was trained on, or fine-tuned on a new task.

The weights are downloaded and cached in a cache folder (so future calls to the from_pretrained method will not re-download them), which defaults to ~/.cache/huggingface/transformers. You can customize the cache folder by setting the HF_HOME environment variable.

The identifier used to load the model can be the identifier of any model on Model Hub, as long as it is compatible with the BERT architecture. A full list of BERT checkpoints can be found [here](https://huggingface.co/models?filter=bert).
### Dynamic padding——Dynamic padding technology
>youtube video: [《what is Dynamic padding》](https://youtu.be/7q5NyFT8REg)

In PyTorch, DataLoader has a parameter - collate function. It is responsible for putting a batch of samples together. By default, it is a function, so it is called a collation function. It converts your samples into PyTorch tensors for connection (recursion if your elements are lists, tuples, or dictionaries).

Since the input sequences we have are of different lengths, the input sequences need to be padded (as input to the model, the tensors of the same batch must be of the same length). As mentioned earlier, padding to the maximum length in the batch is more efficient than padding all sequences to the maximum sequence length of the entire dataset.
Note: If you use TPU, you still need to pad to the max length of the model, because TPU is more efficient this way.
To do this in practice, we must define a collate function that will apply the correct amount of padding to the batched data. (For different batches of data, different lengths of padding are used.) 🤗 The Transformers library provides us with such functionality through DataCollatorWithPadding. When you instantiate it, it requires a tokenizer (to know which one to use)padding tokens, and whether the model wants padding on the left or right side of the input), and it will do everything you need:

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
For testing, we pick the samples from the training set that we want to batch together. Here we need to remove the idx, sentence1, and sentence2 columns because they are not needed and they contain strings (which cannot create tensors). Check the length of each input in the batch:

```python
samples = tokenized_datasets["train"][:8]
samples = {
k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]
}
[len(x) for x in samples["input_ids"]]
```

```python
[50, 59, 47, 67, 59, 50,62, 32]
```
We got sequences of different lengths. Dynamic padding means that all sequences in the batch should be padded to a length of 67. Without dynamic padding, all samples must be padded to the maximum length in the entire dataset, or the maximum length that the model can accept. Let's double-check that our data_collator is padding the batch correctly:

```python
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
```

```python
{'attention_mask': torch.Size([8, 67]),
'input_ids': torch.Size([8, 67]),
'token_type_ids': torch.Size([8, 67]),
'labels': torch.Size([8])}
```

Tips: The bold italic words above are the author's comments, which are the interpretation of part of the original text. If you find any problems in this tutorial, please give us feedback in time. Thank you.