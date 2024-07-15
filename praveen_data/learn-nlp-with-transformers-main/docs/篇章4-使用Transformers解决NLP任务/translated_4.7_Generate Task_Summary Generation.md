The jupter notebook involved in this article is in the [Chapter 4 code base](https://github.com/datawhalechina/learn-nlp-with-transformers/tree/main/docs/%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1).

It is recommended to open this tutorial directly using google colab notebook to quickly download relevant datasets and models.
If you are opening this notebook in google colab, you may need to install the Transformers and 🤗Datasets libraries. Uncomment the following commands to install.

```python
! pip install datasets transformers rouge-score nltk
```

For distributed training, please see [here](https://github.com/huggingface/transformers/tree/master/examples/seq2seq).

Fine-tune transformerModel to solve the summary generation task

In this notebook, we will show how to fine-tune the pre-trained model in [🤗 Transformers](https://github.com/huggingface/transformers) to solve the summary generation task. We use the [XSum dataset](https://arxiv.org/pdf/1808.08745.pdf) dataset. This dataset contains BBC articles and a corresponding summary. Here is an example:

![Widget inference on a summarization task](https://github.com/huggingface/notebooks/blob/master/examples/images/summarization.png?raw=1)

For the summary generation task, we will show how to use a simple loading dataset and fine-tune the model for the corresponding Trainer interface in transformer.

```python
model_checkpoint = "t5-small"
```

As long as the pre-trained transformer model contains the head layer of the seq2seq structure, this notebookThe notebook can theoretically use a variety of transformer models to solve any summary generation task. Here, we use the [`t5-small`](https://huggingface.co/t5-small) model checkpoint.

## Loading data

We will use the [🤗 Datasets](https://github.com/huggingface/datasets) library to load data and corresponding evaluation methods. Data loading and evaluation method loading only need to use load_dataset and load_metric.

```python
from datasets import load_dataset, load_metric

raw_datasets = load_dataset("xsum")

metric = load_metric("rouge")
```

This datasets object itself is a [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict) data structure. ForFor training set, validation set and test set, you only need to use the corresponding key (train, validation, test) to get the corresponding data.

```python
raw_datasets
```

DatasetDict({
train: Dataset({
features: ['document', 'summary', 'id'],
num_rows: 204045
})
validation: Dataset({
features: ['document', 'summary', 'id'],
num_rows: 11332
})
test: Dataset({
features: ['document', 'summary', 'id'],
num_rows: 11334
})
})

Given a data segmentation key (train, validation, test)ion or test) and subscript to view the data:

```python
raw_datasets["train"][0]
```

{'document': 'Recent reports have linked some France-based players with returns to Wales.\n"I\'ve always felt - and this is with my rugby hat on now; this is not region or WRU - I\'d rather spend that money on keeping players in Wales," said Davies.\nThe WRU provides £2m to the fund and £1.3m comes from the regions.\nFormer Wales and British and Irish Lions fly-half Davies became WRU chairman on Tuesday 21 October, succeedingdeposed David Pickering following governing body elections.\nHe is now serving a notice period to leave his role as Newport Gwent Dragons chief executive after being voted on to the WRU board in September.\nDavies was among the leading figures among Dragons, Ospreys, Scarlets and Cardiff Blues officials who were embroiled in a protracted dispute with the WRU that ended in a £60m deal in August this year.\nIn the wake of that deal being done, Davies said the £3.3m should be spent on ensuring currencyOther Wales-based stars remain there.\nIn recent weeks, Racing Metro flanker Dan Lydiate was linked with returning to Wales.\nLikewise the Paris club's scrum-half Mike Phillips and centre Jamie Roberts were also touted for possible returns.\nWales coach Warren Gatland has said: "We haven't instigated contact with the players.\n"But we are aware that one or two of them are keen to return to Wales sooner rather than later."\nSpeaking to Scrum V on BBC Radio Wales, Davies re-iterated his stance, sakeeping players such as Scarlets full-back Liam Williams and Ospreys flanker Justin Tipuric in Wales should take precedence.\n"It\'s obviously a limited amount of money [available]. The union are contributing 60% of that contract and the regions are putting £1.3m in.\n"So it\'s a total pot of just over £3m and if you look at the sorts of salaries that the... guys... have been tempted to go overseas for [are] significant amounts of money.\n"So if we were to bring the players back, we\'d probWe can definitely get five or six players.\n"And I\'ve always felt - and this is with my rugby hat on now; this is not region or WRU - I\'d rather spend that money on keeping players in Wales.\n"There are players coming out of contract, perhaps in the next year or so… you\'re looking at your Liam Williams\' of the world; Justin Tipuric for example - we need to keep these guys in Wales.\n"We actually want them there. They are the ones who are going to impress the young kids, for example.\n"They are the sort off heroes that our young kids want to emulate.\n"So I would start off [by saying] with the limited pot of money, we have to retain players in Wales.\n"Now, if that can be done and there\'s some spare monies available at the end, yes, let\'s look to bring players back.\n"But it\'s a cruel world, isn\'t it?\n"It\'s fine to take the buck and go, but great if you can get them back as well, provided there\'s enough money."\nBritish and Irish Lions centre Roberts has insisted he will see out his RacingMetro contract.\nHe and Phillips also earlier dismissed the idea of ​​leaving Paris.\nRoberts also admitted being hurt by comments in French Newspaper L\'Equipe attributed to Racing Coach Laurent Labit questioning their effectiveness.\nCentre Roberts and flanker Lydiate joined Racing ahead of the 2013-14 season while scrum-half Phillips moved there in December 2013 after being dismissed for disciplinary reasons by former club Bayonne.',
'id': '29750031',
'summary': 'New Welsh Rugby Union chairman Gareth Davies believes a joint £3.3m WRU-regions fund should be used to retain home-based talent such as Liam Williams, not bring back exiled stars.'}

To further understand what the data looks like, the following function will randomly select a few examples from the dataset to display.

```python
import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=2):
assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
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
show_random_elements(raw_datasets["train"])
```

<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>document</th>
<th>id</th>
<th>summary</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Media playback is not supported on this device\nThe 31-year-old Portugal captain has the chance to win the first international title of his glittering career when his side take on the tournament hosts at Stade de France.\n"Everyone in the country will be against him, but he will thrive off"That hostility, and off their fear," Ferdinand told BBC Sport.\n"The French know Cristiano is the player capable of destroying their dream because he has produced magic moments in huge matches right through his career.\n"Part of the reason he is a superstar is because he is not fazed by the big occasion - quite the opposite in fact.\n"Superstars like him relish these situations - the pressure that goes with it brings the best out of him, when other players falter."\nRonaldo helped Portugal reachthe final when they hosted Euro 2004, but was left crying on the pitch after defeat by Greece, and they have not reached this stage of a major tournament since.\n"He was only 19 then, so just a kid," added Ferdinand, who played alongside Ronaldo at Old Trafford between 2003 and 2009 and is working in France as a BBC pundit.\n"I remember his reaction but I think he was a bit too young to take it all in. At that age, you expect you will get back to another final soon to rectify what happened buthe has had to wait 12 years for his chance.\n"I think he is very aware this is his last opportunity to win something with his country and, knowing him like I do, that makes him even more dangerous. He will be so desperate not to miss out again.\n"Cristiano has produced great performances for Portugal when it matters before, for example his hat-trick against Sweden in the play-off for the 2014 World Cup.\n"So France will know that it is not just in a Real Madrid or Manchester United shirt that heis capable of great things, especially because it was his moment of brilliance that helped decide Portugal's semi-final against Wales."\nRonaldo is now the joint highest scorer in European Championship history with nine goals, and holds the record for most headed goals with five.\nTwo of them have come in France, most notably when he soared to nod Portugal in front against Wales, but Ferdinand says that part of Ronaldo's game is nothing new.\n"He has always been amazing on the ball but even whenn he first joined United in 2003 he was great in the air too," he said.\n"Early in his career it was a part of his game that was quite undervalued but he always scored a lot of headers and, the way he does it, he is the closest thing in football to basketball legend Michael Jordan.\n"The way he jumps and hangs in the air is the same as Jordan and he has got the ability to stay up there, assess the situation and then put the ball where he wants to, with power.\n"I will always remember the headerhe scored for United away at Roma in the Champions League quarter-finals in April 2008.\n"He more or less jumped on the edge of the box to meet a cross that Paul Scholes put over but he met the ball a good way inside the area. If you watch TV footage of that game, he just appears from nowhere and smashes it into the bottom corner.\n"Just like his goal against Wales, it was an unbelievable jump and he generated incredible power. I was on the pitch that night, and it was amazing to see.\n"Cristiano's heading ability will be a huge threat in the final too, along with Nani - another former United team-mate of mine.\n"France have struggled to defend crosses for most of the tournament and, although they were better at it in the semi-final, Germany did not have anyone to aim at in the box."\nEighty-six players at Euro 2016 have completed more dribbles than the three Ronaldo has managed in six matches.\n"He was always well known for his brilliant runs forward but his game is not about that anymore," said Ferdinand, who was in the same United team as Ronaldo when he scored 42 goals in the 2007-08 season.\n"Before, he used to exert a lot of energy trying to take people on from deep areas, running at goal from 30 or 40 yards, or even further out.\n"Now, he is very clever in where he tries to receive the ball. It has to be in good positions and, when he gets it, he finds a yard of space and hits it - either a shot or a cross.\n"Part of the reason he has been able to reinvent himself isbecause of how hard he works - right from the start of his career, when we were together at Old Trafford, he was totally committed to improving every part of his game.\n"But to be able to re-evaluate his game and change it is also down to his football intelligence.\n"Clearly he is clever - you do not score 50 goals a season, six seasons running, for Real Madrid if you are not.\n"But his extra intelligence has allowed him to evolve as a player, understand his body, where it can take him and how toften.\n"He has become a much more efficient player, but is still an extremely effective one."\nSunday offers Ronaldo the opportunity for personal glory too, with the chance to get one over Lionel Messi in the battle to be viewed as the best player in the world.\nMessi has never won a major tournament for Argentina and announced his international retirement last month after they lost to Chile in the final of the Copa America.\n"If Ronaldo wins the European Championship, it will be massive for him," said Ferdinand, 37.\n"I don't think it will give him the edge over Messi in terms of who deserves the accolade of the best in the world, but it is a huge achievement.\n"And it will matter to both of them, because there is a definite battle between them in their own minds about who has done what for club and country.\n"It is far from a given that Ronaldo will manage it, of course. France are looking very good and they have a game-changer in Antoine Griezmann.\n"Even if Ronaldo is at his best,it is a difficult ask for them and I think they are going to have to play ugly, like they have done all the way through the tournament, to win.\n"I have known him a long time and I would love to see him do it, but it is awkward for me because I have friends in both camps - Nani and Cristiano for Portugal, and Paul Pogba and Patrice Evra for France.\n"So I am not really bothered about the result, I just want to see a good game. I would love to see Ronaldo and Griezmann perform to their potentialand finish off this tournament on a high note."</td>
<td>36749142</td>
<td>Cristiano Ronaldo will relish taking on the whole of France in Sunday's Euro 2016 final, according to his former Manchester United team-mate Rio Ferdinand.</td>
</tr>
<tr>
<th>1</th>
<td>Media playback is unsupported on your device\n18 December 2014 Last updated at 10:28 GMT\nMalaysia has successfully tackled poverty over the last four decades by drawing on its rich natural resources.\nAccoAccording to the World Bank, some 49% of Malaysians in 1970 were extremely poor, and that figure has been reduced to 1% today. However, the government's next challenge is to help the lower income group to move up to the middle class, the bank says.\nUlrich Zahau, the World Bank's Southeast Asia director, spoke to the BBC's Jennifer Pak.</td>
<td>30530533</td>
<td>In Malaysia, the "aspirational" low-income part of the population is helping to drive economic growth through consumption, accounting for 50% of the population.rding to the World Bank.</td>
</tr>
</tbody>
</table>

The metric is an instance of the [`datasets.Metric`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Metric) class. See metric and examples of usage:

```python
metric
```

Metric(name: "rouge", features: {'predictions': Value(dtype='string', id='sequence'), 'references': Value(dtype='string', id='sequence')}, usage: """
Calculates average rouge scores for a list of hypotheses and references
Args:
predictions: list of predictions to score. Each predictions
should be a string with tokens separated by spaces.
references: list of reference for each prediction. Each
reference should be a string with tokens separated by spaces.
rouge_types: A list of rouge types to calculate.
Valid names:
`"rouge{n}"` (e.g. `"rouge1"`, `"rouge2"`) where: {n} is the n-gram based scoring,
`"rougeL"`: Longest common subsequence based scoring.
`"rougeLSum"`: rougeLsum splits text using `"
"`.
See details in https://github.com/huggingface/datasets/issues/617
use_stemmer: Bool indicating whether Porter stemmer should be used to strip word suffixes.
use_agregator: Return aggregates if this is set to True
Returns:
rouge1: rouge_1 (precision, recall, f1),
rouge2: rouge_2 (precision, recall, f1),
rougeL: rouge_l (precision, recall, f1),
rougeLsum: rouge_lsum (precision, recall, f1)
Examples:

>>> rouge = datasets.load_metric('rouge')
>>> predictions = ["hello there", "general kenobi"]
>>> references = ["hello there", "general kenobi"]
>>> results = rouge.compute(predictions=predictions, references=references)
>>> print(list(results.keys()))
['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
>>> print(results["rouge1"])
AggregateScore(low=Score(precision=1.0, recall=1.0, fmeasure=1.0), mid=Score(precision=1.0, recall=1.0, fmeasure=1.0), high=Score(precision=1.0, recall=1.0, fmeasure=1.0))
>>> print(results["rouge1"].mid.fmeasure)
1.0
""", stored examples: 0)

We use the `compute` method to compare predictions and labels to calculate the score. Both predictions and labels need to be a list. See the example below for the specific format:

```python
fake_preds = ["hello there", "general kenobi"]
fake_labels = ["hello there", "general kenobi"]
metric.compute(predictions=fake_preds, references=fake_labels)
```

{'rouge1': AggregateScore(low=Score(precision=1.0, recall=1.0, fmeasure=1.0))l=1.0, fmeasure=1.0), mid=Score(precision=1.0, recall=1.0, fmeasure=1.0), high=Score(precision=1.0, recall=1.0, fmeasure=1.0)),
'rouge2': AggregateScore(low=Score(precision=1.0, recall=1.0, fmeasure=1.0), mid=Score(precision=1.0, recall=1.0, fmeasure=1.0), high=Score(precision=1.0, recall=1.0, fmeasure=1.0)),
'rougeL': AggregateScore(low=Score(precision=1.0, recall=1.0, fmeasure=1.0), mid=Score(precision=1.0, recall=1.0, fmeasure=1.0), high=Score(precision=1.0, recall=1.0, fmeasure=1.0)),
'rougeLsum': AggregateScore(low=Score(precision=1.0, recall=1.0, fmeasure=1.0), mid=Score(precision=1.0, recall=1.0, fmeasure=1.0), high=Score(precision=1.0, recall=1.0, fmeasure=1.0))}

## Data preprocessing

Before feeding the data into the model, we need to preprocess the data. The preprocessing tool is called Tokenizer. Tokenizer first tokenizes the input, then converts the tokens into the corresponding token ID required in the pre-model, and then converts it into the input format required by the model.

In order to achieve the purpose of data preprocessing, we use the AutoTokenizer.from_pretrained method to instantiate our tokenizer, which ensures:

- We get a tokenizer that corresponds to the pre-trained model one by one.
- When using the tokenizer corresponding to the specified model checkpoint, we also download the vocabulary required by the model, to be precise, the tokens vocabulary.

The downloaded tokens vocabulary will be cached so that it will not be downloaded again when used again.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

```

By default, the call above will use one of the fast tokenizers (backed by Rust) from the 🤗 Tokenizers library.

Tokenizer can preprocess a single text or a pair of texts. The data obtained after tokenizer preprocessing meets the input format of the pretrained model

```python
tokenizer("Hello, this one sentence!")
```

{'input_ids': [8774, 6, 48, 80, 7142, 55, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}

The token IDs or input_ids seen above generally vary with the names of the pre-trained models. The reason is that different pre-trained models set different rules during pre-training. But as long as the names of the tokenizer and the model are the same, the input format of the tokenizer preprocessing will meet the model requirements. For more information about preprocessing, refer to [this tutorial](https://huggingface.co/transformers/preprocessing.html)

In addition to tokenizing a sentence, we can also tokenize a list of sentences.

```python
tokenizer(["Hello, this one sentence!", "This is another sentence."])
```

{'input_ids': [[8774, 6, 48, 80, 7142, 55, 1], [100, 19, 430, 7142, 5, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}

Note: In order to give the model accuratePrepare the translation targets. We use `as_target_tokenizer` to control the special tokens corresponding to the targets:

```python
with tokenizer.as_target_tokenizer():
print(tokenizer(["Hello, this one sentence!", "This is another sentence."]))
```

{'input_ids': [[8774, 6, 48, 80, 7142, 55, 1], [100, 19, 430, 7142, 5, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}

If you are using the checkpoints of the T5 pre-trained model, you need to check for special prefixes. T5 uses special prefixes to tell the model what tasks to do. Examples of specific prefixes are as follows:

```python
if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
prefix = "summarize: "
else:
prefix = ""
```

Now we can put everything together to form our preprocessing function. When we preprocess the samples, we also use the parameter `truncation=True` to ensure that our long sentences are truncated. By default, we automatically pad for shorter sentences.

```python
max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
inputs = [prefix + doc for doc in examples["document"]]
model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

# Setup the tokenizer for targets
with tokenizer.as_target_tokenizer():
labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

model_inputs["labels"] = labels["input_ids"]
return model_inputs
```

The above preprocessing function can process one sample or multiple sample examples. If it processes multiple samples, the result list after multiple samples are preprocessed is returned.

```python
preprocess_function(raw_datasets['train'][:2])
```

{'input_ids': [[21603, 10, 17716, 2279, 43, 5229, 128, 1410, 18, 390, 1508, 28, 5146, 12, 10256, 5, 96, 196, 31, 162, 373, 1800, 3, 18, 11, 48, 19, 28, 82, 22209, 3, 547, 30, 230, 117, 48, 19, 59, 1719, 42, 549, 8503, 3, 18, 27, 31, 26, 1066, 1492, 24, 540, 30, 2627, 1508, 16, 10256, 976, 243, 28571, 5, 37, 549, 8503, 795, 17586, 51, 12, 8, 3069, 11, 3996, 13606, 51, 639, 45, 8, 6266, 5, 18263, 10256, 11, 2390, 11, 7262, 10371, 7, 3971, 18, 17114, 28571, 1632, 549, 8503, 13404, 30, 2818, 1401, 1797, 6, 7229, 53, 20, 12151, 1955, 8356, 49, 53, 826, 3, 19585, 643, 9768, 5, 216, 19, 230, 3122, 3, 9, 2103, 1059, 12, 1175, 112, 1075, 38, 24260, 350, 16103, 10282, 7, 5752, 4297, 227, 271, 3, 11060, 30, 12, 8, 549, 8503, 1476, 16, 1600, 5, 28571, 47, 859, 8, 1374, 5638, 859, 10282, 7, 6, 411, 7, 2026, 63, 7, 6, 14586, 7677, 11, 26911, 2419, 7, 4298, 113, 130, 10960, 52, 26786, 16, 3, 9, 813, 11674, 11044, 28, 8, 549, 8503, 24, 3492, 16, 3, 9, 3996, 3328, 51, 1154, 16, 1660, 48, 215, 5, 86, 8, 7178, 13, 24, 1154, 271, 612, 6, 28571, 243, 8, 3996, 19660, 51, 225, 36, 1869, 30, 3, 5833, 750, 10256, 18, 390, 4811, 2367, 132, 5, 86, 1100, 1274, 6, 16046, 10730, 24397, 49, 2744, 31914, 17, 15,47, 5229, 28, 7646, 12, 10256, 5, 3, 21322, 8, 1919, 1886, 31, 7, 14667, 440, 18, 17114, 4794, 16202, 7, 11, 2050, 17845, 2715, 7, 130, 92, 2633, 26, 21, 487, 5146, 5, 10256, 3763, 16700, 2776, 17, 40, 232, 65, 243, 10, 96, 1326, 43, 29, 31, 17, 16, 7, 2880, 920, 574, 28, 8, 1508, 5, 96, 11836, 62, 33, 2718, 24, 80, 42, 192, 13, 135, 33, 9805, 12, 1205, 12, 10256, 14159, 1066, 145, 865, 535, 14734, 12, 4712, 2781, 584, 30, 9938, 5061, 10256, 6, 28571, 3, 60, 18, 155, 15, 4094, 112, 3, 8389, 6,2145, 2627, 1508, 224, 38, 14586, 7677, 423, 18, 1549, 1414, 265, 6060, 11, 411, 7, 2026, 63, 7, 24397, 49, 12446, 2262, 3791, 447, 16, 10256, 225, 240, 20799, 1433, 5, 96, 196, 17, 31, 7, 6865, 3, 9, 1643, 866, 13, 540, 784, 28843, 4275, 37, 7021, 33, 12932, 15436, 13, 24, 1696, 11, 8, 6266, 33, 3, 3131, 3996, 13606, 51, 16, 5, 96, 5231, 34, 31, 7, 3, 9, 792, 815, 13, 131, 147, 23395, 51, 11, 3, 99, 25, 320, 44, 8, 10549, 13, 21331, 24, 8, 233, 3413, 233, 43, 118, 3, 22765, 12, 281, 10055, 21,784, 355, 908, 1516, 6201, 13, 540, 5, 96, 5231, 3, 99, 62, 130, 12, 830, 8, 1508, 223, 6, 62, 31, 26, 1077, 129, 874, 42, 1296, 1508, 5, 96, 7175, 27, 31, 162, 373, 1800, 3, 18, 11, 48, 19, 28, 82, 22209, 3, 547, 30, 230, 117, 48, 19, 59, 1719, 42, 549, 8503, 3, 18, 27, 31, 26, 1066, 1492, 24, 540, 30, 2627, 1508, 16, 10256, 5, 96, 7238, 33, 1508, 1107, 91, 13, 1696, 6, 2361, 16, 8, 416, 215, 42, 78, 233, 25, 31, 60, 479, 44, 39, 1414, 265, 6060, 31, 13, 8, 296, 117, 12446, 2262, 3791, 447, 21,677, 3, 18, 62, 174, 12, 453, 175, 3413, 16, 10256, 5, 96, 1326, 700, 241, 135, 132, 5, 328, 33, 8, 2102, 113, 33, 352, 12, 18514, 8, 1021, 1082, 6, 21, 677, 5, 96, 10273, 33, 8, 1843, 13, 17736, 24, 69, 1021, 1082, 241, 12, 29953, 5, 96, 5231, 27, 133, 456, 326, 784, 969, 2145, 908, 28, 8, 1643, 815, 13, 540, 6, 62, 43, 12, 7365, 1508, 16, 10256, 5, 96, 17527, 6, 3, 99, 24, 54, 36, 612, 11, 132, 31, 7, 128, 8179, 3, 26413, 347, 44, 8, 414, 6, 4273, 6, 752, 31, 7, 320, 12, 830, 1508, 223, 5, 96, 11836, 34, 31, 7, 3, 9, 23958, 296, 6, 19, 29, 31, 17, 34, 58, 96, 196, 17, 31, 7, 1399, 12, 240, 8, 3, 13863, 11, 281, 6, 68, 248, 3, 99, 25, 54, 129, 135, 223, 38, 168, 6, 937, 132, 31, 7, 631, 540, 535, 2390, 11, 7262, 10371, 7, 2050, 2715, 7, 65, 16, 15777, 3, 88, 56, 217, 91, 112, 16046, 10730, 1696, 5, 216, 11, 16202, 7, 92, 2283, 19664, 8, 800, 13, 3140, 1919, 5, 2715, 7, 92, 10246, 271, 4781, 57, 2622, 16, 2379, 29494, 301, 31, 427, 23067, 15, 3, 20923, 12, 16046, 9493, 9906, 17, 325,2360, 822, 53, 70, 9570, 5, 2969, 2715, 7, 11, 24397, 49, 31914, 17, 15, 3311, 16046, 2177, 13, 8, 2038, 11590, 774, 298, 14667, 440, 18, 17114, 16202, 7, 2301, 132, 16, 1882, 2038, 227, 271, 19664, 21, 3, 15471, 2081, 57, 1798, 1886, 2474, 4006, 91, 5, 5, 2409, 7, 2, 130, 16645, 326, 11, 2117, 12355, 1054, 38, 3, 9, 6478, 16813, 47, 4006, 91, 5, 37, 12787, 6,261, 57, 1932, 27874, 5220, 31195, 3230, 6, 43, 118, 7774, 3, 9, 381, 13, 648, 5, 1377, 1310, 6, 17602, 6417, 6032, 130, 4006, 91, 30, 8, 6036, 30, 12096, 8348, 16, 1186, 11, 932, 5, 37, 6032, 1553, 826, 3, 9, 27874, 896, 2063, 2902, 16, 1882, 1673, 18395, 53, 8, 7070, 13, 8, 7021, 5692, 44, 8, 896, 2501, 5, 1193, 1778, 29, 53, 8, 1251, 3534, 9, 226, 6, 11529, 283, 4569, 4409, 5225, 8692, 243, 10, 96, 196, 17, 19, 3, 9, 2261, 5415, 21, 8, 415, 616, 6, 34, 4110, 2261, 17879, 6, 34, 10762, 151, 3112, 103, 70, 613, 11, 830, 174, 151, 12, 4831, 535, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 333, 535, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 333, 535, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, ... 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 59, 830, 223, 1214, 699, 26, 34, 261, 12, 7365, 234, 18, 390, 3683, 224, 38, 1414, 265, 6060, 4, 6, 223, 1215, 699, 26, 34, 261, 12, 7365, 234, 18, 390, 3683, 224, 38, 1414, 265, 6060, 4, 6, 223, 1214, 699, 26, 4811, 5, 1], [71, 21641, 2642, 646, 1067, 46, 11529, 3450, 828, 16, 5727, 27874, 65, 118, 10126, 3, 9, 3534, 9, 226, 5, 1]]}

Next, all samples in the datasets are preprocessed by using the map function to apply the preprocessing function prepare_train_features to all samples.

```python
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
```

Better yet, the returned results are automatically cached to avoid recalculation the next time you process them (but be aware that if the input is changed, it may be affected by the cache!). The datasets library function will check the input parameters to see if there are any changes. If there are no changes, the cached data will be used. If there are changes, the data will be reprocessed. However, if the input parameters do not change, it is best to clear the cache when you want to change the input. The way to clear is to use the `load_from_cache_file=False` parameter. In addition, the `batched=`True` is a tokenizer feature, because it uses multiple threads to process the input in parallel.

## Fine-tune the model

Now that the data is ready, we need to download and load our pre-trained model, and then fine-tune the pre-trained model. Since we are doing seq2seq tasks, we need a model class that can solve this task. We use the `AutoModelForSeq2SeqLM` class. Similar to the tokenizer, the `from_pretrained` method can also help us download and load the model, and it will also cache the model so that we don't download the model repeatedly.

```python
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

Since our fine-tuning task is a seq2seq task, and we load a pre-trained seq2seq model, we will not be prompted that some unmatched neural network parameters were thrown away when loading the model.number (for example: the neural network head of the pre-trained language model is thrown away, and the neural network head of the machine translation is randomly initialized).

In order to get a `Seq2SeqTrainer` training tool, we need 3 more elements, the most important of which is the training settings/parameters [`Seq2SeqTrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Seq2SeqTrainingArguments). This training setting contains all the properties that can define the training process

```python
batch_size = 16
args = Seq2SeqTrainingArguments(
"test-summarization",
evaluation_strategy = "epoch",
learning_rate=2e-5,
per_device_train_batch_size=batch_size,
per_device_eval_batch_size=batch_size,
weight_decay=0.01,
save_total_limit=3,
num_train_epochs=1,
predict_with_generate=True,
fp16=True,
)
```

The evaluation_strategy = "epoch" parameter above tells the training code that we will do a validation evaluation once per epoch.

The batch_size above was defined before this notebook.

Since our dataset is large and `Seq2SeqTrainer` will keep saving models, we need to tell it to save at most `save_total_limit=3` models.

Finally, we need a data collator data to feed our processed input to the model.

```python
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

The last thing left after setting up `Seq2SeqTrainer` is that we need to define the evaluation method. We use `metric` to complete the evaluation. Before sending the model predictions to evaluation, we will also do some data post-processing:

```python
import nltk
import numpy as np

def compute_metrics(eval_pred):
predictions, labels = eval_pred
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
# Replace -100 in the labels as we can't decode them.
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

# Rouge expects a newline after each sentence
decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
# Extract a few results
result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

# Add mean generated length
prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
result["gen_lensen"] = np.mean(prediction_lens)

return {k: round(v, 4) for k, v in result.items()}
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

<div>
<style>
/* Turns off somestyling */
progress {
/* gets rid of default border in Firefox and Opera. */
border: none;
/* Needs to be in here for Safari polyfill so background images work as expected. */
background-size: auto;
}
</style>

<progress value='12753' max='12753' style='width:300px; height:20px; vertical-align: middle;'></progress>
[12753/12753 1:21:48, Epoch 1/1]
</div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: left;">
<th>Epoch</th>
<th>Training Loss</th>
<th>Validation Loss</th>
<th>Rouge1</th>
<th>Rouge2</th>
<th>Rougel</th>
<th>Rougelsum</th>
<th>Gen Len</th>
<th>Runtime</th>
<th>Samples Per Second</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>2.721100</td>
<td>2.479327</td>
<td>28.300900</td>
<td>7.721100</td>
<td>22.243000</td>
<td>22.249600</td>
<td>18.822500</td>
<td>326.333800</td>
<td>34.725000</td>
</tr>
</tbody>
</table><p>

TrainOutput(global_step=12753, training_loss=2.7692033505520146, metrics={'train_runtime': 4909.3835, 'train_samples_per_second': 2.598, 'total_flos': 7.774481450954342e+16, 'epoch': 1.0, 'init_mem_cpu_alloc_delta': 335248, 'init_mem_gpu_alloc_delta': 242026496, 'init_mem_cpu_peaked_delta': 18306, 'init_mem_gpu_peaked_delta': 0, 'train_mem_cpu_alloc_delta': 2637782, 'train_mem_gpu_alloc_delta': 728138240, 'train_mem_cpu_peaked_delta': 138226182, 'train_mem_gpu_peaked_delta': 14677017088})

Finally, don't forget to check how to upload the model and upload the model to [🤗 Model Hub](https://huggingface.co/models). Then you can use your model directly by using the model name as in the beginning of this notebook.

```python

```