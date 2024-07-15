# Chapter 6 Evaluation

Evaluation is a key step in testing the quality of language model question answering. Evaluation can test the question answering effect of the language model on different documents and find out its weaknesses. It can also select the best system by comparing different models. In addition, regular evaluation can also check the decay of model quality. Evaluation usually has two purposes:
- Check whether the LLM application meets the acceptance criteria
- Analyze the impact of changes on the performance of LLM applications

The basic idea is to use the language model itself and the chain itself to assist in the evaluation of other language models, chains and applications. Let's take the document question answering application in the previous chapter as an example to discuss how to handle and consider the content of evaluation in LangChain in this chapter.

## 1. Create an LLM application
First, build an LLM document question and answer application in the langchain chain mode

```python
from langchain.chains import RetrievalQA #Retrieval QA chain, search on documents
from langchain.chat_models import ChatOpenAI #openai model
from langchain.document_loaders import CSVLoader #Document loader, stored in csv format
from langchain.indexes import VectorstoreIndexCreator #Import vector store index creator
from langchain.vectorstores import DocArrayInMemorySearch #Vector store
#Load Chinese data
file = '../data/product_data.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

#View data
import pandas as pd
test_data = pd.read_csv(file,skiprows=0)
display(test_data.head())
```

<div>
<style scoped>
.dataframe tbody tr th:only-of-type {
vertical-align: middle;
}

.dataframe tbody tr th {
vertical-align: top;
}

.dataframe theadth {
text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>product_name</th>
<th>description</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Fully automatic coffee machine</td>
<td>Specifications:\nLarge - Size: 13.8'' x 17.3''. \nMedium - Size: 11.5'' ...</td>
</tr>
<tr>
<th>1</th>
<td>Electric toothbrush</td>
<td>Specifications:\nGeneral size - Height: 9.5'', Width: 1''. \n\nWhy we love it:\nOur...</td>
</tr>
<tr>
<th>2</th>
<td>Orange Flavored Vitamin C Effervescent Tablets</td>
<td>Specifications:\nEach box contains 20 tablets. \n\nWhy we love it:\nOur Orange Flavored Vitamin C Effervescent Tablets are a quick way to replenish vitamin...</td>
</tr>
<tr>
<th>3</th>
<td>Wireless Bluetooth Headset</td>
<td>Specifications:\nSingle Headset Size: 1.5'' x 1.3''. \n\nWhy we love it:\nThis wireless Bluetooth...</td>
</tr>
<tr>
<th>4</th>
<td>Yoga Mat</td>
<td>Specifications:\nSize: 24'' x 68''. \n\nWhy we love it:\nOur yoga mats have excellent...</td>
</tr>
</tbody>
</table>
</div>

```python
# Will specify the vector storage class, once created, we will call it from the loader, loading through the document recorder list

index = VectorstoreIndexCreator(
vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

#Create a retrieval QA chain by specifying the language model, chain type, retriever, and the level of detail we want to print
llm = ChatOpenAI(temperature = 0.0)
qa = RetrievalQA.from_chain_type(
llm=llm, 
chain_type="stuff", 
retriever=index.vectorstore.as_retriever(), 
verbose=True,
chain_type_kwargs = {
"document_separator": "<<<<>>>>>"
}
)
```

The main functions and effects of the above code have been explained in the previous section, so I will not repeat them here

### 1.1 Set up the test data
Let's take a look at the information in the data generated after loading by the file loader CSVLoad. Here we extract data The ninth and tenth data in the table, take a look at their main contents:

Ninth data:

```python
data[10]
```

Document(page_content="product_name: HDTV\ndescription: Specifications:\nSize: 50''.\n\nWhy we love it:\nOur HDTVs offer an immersive viewing experience with outstanding picture quality and powerful sound.\n\nMaterial and Care:\nClean with a dry cloth.\n\nConstruction:\nMade of plastic, metal and electronic components.\n\nOther Features:\nSupports network connection for online video viewing.\nComes with remote control.\nMade in Korea.\n\nHave questions? Feel free to contact our customer service team who will answer all your questions.", metadata={'source': '../data/product_data.csv', 'row': 10})

Tenth data:

```python
data[11]
```

Document(page_content="product_name: Travel backpack\ndescription: Specifications:\nSize: 18'' x 12'' x 6''. \n\nWhy we love it:\nOur travel backpacks are ideal for short trips, with multiple practical interior and exterior pockets to easily fit your essentials. \n\nMaterial & Care:\nHand wash and air dry. \n\nConstruction:\nMade from water-resistant nylonMade of Dragon. \n\nAdditional Features: \nComes with adjustable strap and safety lock. \nMade in China. \n\nQuestions? Feel free to contact our customer service team who will answer all your questions. ", metadata={'source': '../data/product_data.csv', 'row': 11})

Look at the first document above, there is a high-definition TV, and the second document has a travel backpack. From these details, we can create some example queries and answers

### 1.2 Manually create test data

It should be noted that our document here is a csv file, so the document loader we use is CSVLoader. CSVLoader will split each row of data in the csv file, so the data[10] and data[11] you see here are the contents of the 10th and 11th data in the csv file. Next, we manually set two "question-answer pairs" based on these two data. Each "question-answer pair" contains a query and an answer:

```python
examples = [
{
"query": "How to care for a high-definition TV?",
"answer": "Clean with a dry cloth. "
},{
"query": "Does the travel backpack have inner and outer pockets?",
"answer": "Yes."
}
]
```

### 1.3 Generate test cases through LLM

In the previous content, we used manual methods to build test data sets. For example, we manually created 10 questions and 10 answers, and then asked LLM to answer these 10 questions, and then compared the answers given by LLM with the answers we prepared, and finally gave LLM a score. This is the evaluation process. But there is a problem here, that is, we need to manually create all the question sets and answer sets, which will be very time-consuming and labor-intensive. Is there a way to automatically create a large number of question-answer test sets? Of course there is. Today we will introduce the method provided by Langchain: `QAGenerateChain`. We can use `QAGenerateChain` to automatically create a Q&A set for our document:

Since the `PROMPT` used in the `QAGenerateChain` class is in English, we inherit the `QAGenerateChain` class and add "Please use Chinese output" to `PROMPT`. The following is the `QAGenerateChain` class in the `generate_chain.py` fileSource code

```python
from langchain.evaluation.qa import QAGenerateChain #Import the QA generation chain, which will receive documents and create a question-answer pair from each document

# Below is the source code in langchain.evaluation.qa.generate_prompt, add "Please use Chinese output" at the end of template
from langchain.output_parsers.regex import RegexParser
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
from typing import Any

template = """You are a teacher coming up with questions to ask on a quiz. 
Given the following document, please generate a question and answer based on that document.

Example Format:
<Begin Document>
...
<End Document>
QUESTION: question here
ANSWER: answer here

These questions should be detailed and be based explicitly on information in the document. Begin!

<Begin Document>
{doc}
<End Document>
Please use Chinese output
"""
output_parser = RegexParser(
regex=r"QUESTION: (.*?)\nANSWER: (.*)", output_keys=["query", "answer"]
)
PROMPT = PromptTemplate(
input_variables=["doc"], template=template, output_parser=output_parser
)

# Inherit QAGenerateChain
class ChineseQAGenerateChain(QAGenerateChain):
"""LLM Chain specifically for generating examples for question answering."""

@classmethod
def from_llm(cls, llm: BaseLanguageModel, **kwargs: Any) -> QAGenerateChain:
"""Load QA Generate Chain from LLM."""
return cls(llm=llm, prompt=PROMPT, **kwargs)

example_gen_chain = ChineseQAGenerateChain.from_llm(ChatOpenAI())#Create this chain by passing the chat open AI language model
new_examples = example_gen_chain.apply([{"doc": t} for t in data[:5]])

#See the use caseData
new_examples 
```

[{'qa_pairs': {'query': 'What are the dimensions of this fully automatic coffee machine? ',
'answer': "Large size is 13.8'' x 17.3'', Medium size is 11.5'' x 15.2''."}},
{'qa_pairs': {'query': 'What are the specifications of this electric toothbrush? ', 'answer': "General size - Height: 9.5'', Width: 1''."}},
{'qa_pairs': {'query': 'What is the name of this product? ', 'answer': 'The name of this product is Orange Flavored Vitamin C Effervescent Tablets. '}},
{'qa_pairs': {'query': 'What are the dimensions of this wireless Bluetooth headset? ',
'answer': "The dimensions of this wireless Bluetooth headset are 1.5'' x 1.3''."}},
{'qa_pairs': {'query': 'What are the dimensions of this yoga mat? ', 'answer': "The dimensions of this yoga mat are 24'' x 68''."}}]

In the above code, we create a `QAGenerateChain`, and then we applied the apply method of `QAGenerateChain` to create 5 "question-answer pairs" for the first 5 pieces of data. Since the creation of question-answer sets is automatically completed by LLM, it will involve the issue of token cost, so for the purpose of demonstration, we only create question-answer sets for the first 5 pieces of data in data.

```python
new_examples[0]
```

{'qa_pairs': {'query': 'What are the dimensions of this fully automatic coffee machine? ',
'answer': "Large is 13.8'' x 17.3'' and Medium is 11.5'' x 15.2''."}}

Source data:

```python
data[0]
```

Document(page_content="product_name: Fully automatic coffee machine\ndescription: Specifications:\nLarge - Dimensions: 13.8'' x 17.3''.\nMedium - Dimensions: 11.5'' x 15.2''.\n\nWhy we love it:\nThis fully automatic coffee machine is the perfect choice for coffee lovers. With one click, you can grind beans and brew your favorite coffee. ItIts durability and consistency make it an ideal choice for home and office. \n\nMaterial & Care:\nJust wipe gently to clean. \n\nConstruction:\nMade of high-quality stainless steel. \n\nOther Features:\nBuilt-in grinder and filter. \nPreset multiple coffee modes. \nMade in China. \n\nQuestions? Feel free to contact our customer service team who will answer all your questions. ", metadata={'source': '../data/product_data.csv', 'row': 0})

### 1.4 Integrate test set

Remember the two Q&A sets we created manually before? Now we need to merge the Q&A sets created manually into the Q&A set created by `QAGenerateChain`, so that there are both manually created examples and examples automatically created by llm in the answer set, which will make our test set more complete.

Next, we need to let the document Q&A chain `qa` created previously answer the questions in this test set. Let's see how LLM answers:

```python
examples += [ v for item in new_examples for k,v in item.items()]
qa.run(examples[0]["query"])
```> Entering new RetrievalQA chain...

> Finished chain.

'It is very easy to care for a HDTV. You only need to use a dry cloth to clean it. Avoid using a wet cloth or chemical cleaners to avoid damaging the surface of the TV. '

Here we see that `qa` answered the 0th question: "How to care for a HDTV?" The 0th question here is the first question we created manually before, and the answer we created manually is: "Use a dry cloth to clean it." Here we find that the Q&A chain `qa` also answered "You only need to use a dry cloth to clean it", but it has an additional explanation than our answer: "It is very easy to care for a HDTV. You only need to use a dry cloth to clean it. Avoid using a wet cloth or chemical cleaners to avoid damaging the surface of the TV.".

## 2. Manual evaluation

Do you want to know how `qa` finds the answer to the question? Let's turn on `debug` and see how `qa` finds the answer to the question!

```python
import langchain
langchain.debug = True

# Rerun the same example as above and you can see it starts printing more information
qa.run(examples[0]["query"])
```

[chain/start] [1:chain:RetrievalQA] Entering Chain run with input:
{
"query": "How to care for HDTV? "
}
[chain/start] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain] Entering Chain run with input:
[inputs]
[chain/start] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain] Entering Chain run with input:
{
"question": "How to care for HDTV? ",
"context": "product_name: HDTV\ndescription: Specifications:\nSize: 50''.\n\nWhy we love it:\nOur HDTVs have excellent picture quality and powerful soundeffect, bringing an immersive viewing experience. \n\nMaterial & Care:\nClean with a dry cloth. \n\nConstruction:\nMade of plastic, metal and electronic components. \n\nOther Features:\nSupports network connection to watch videos online. \nEquipped with remote control. \nMade in Korea. \n\nQuestions? Feel free to contact our customer service team, who will answer all your questions. <<<<>>>>>product_name: Air Purifier\ndescription: Specifications:\nDimensions: 15'' x 15'' x 20''. \n\nWhy we love it:\nOur air purifiers use advanced HEPA filtration technology to effectively remove particles and odors from the air, providing you with a fresh indoor environment. \n\nMaterial & Care:\nWipe with a dry cloth when cleaning. \n\nConstruction:\nMade of plastic and electronic components. \n\nOther Features:\nThree wind speeds with a timer function. \nMade in Germany. \n\nQuestions? Please feel free to contact our customer service team who will answer all your questions. <<<<>>>>>product_name: Automatic Pet Feeder\ndescription: Specifications:\nDimensions: 14'' x 9'' x 15''.\n\nWhy we love it:\nOur automatic pet feeder can deliver food at a fixed time, allowing you to ensure your pet's diet whether you are at home or away.food. \n\nMaterial and Care:\nCan be cleaned with a damp cloth. \n\nConstruction:\nMade of plastic and electronic components. \n\nOther Features:\nEquipped with an LCD screen for easy operation. \nCan be set for multiple feedings. \nMade in the USA. \n\nHave a question? Feel free to contact our customer service team, who will answer all your questions. <<<<>>>>>product_name: Glass Protector\ndescription: Specifications:\nSuitable for mobile phone screens of all sizes. \n\nWhy we love it:\nOur glass protector can effectively prevent mobile phone screens from scratches and cracks without affecting the sensitivity of touch. \n\nMaterial and Care:\nWipe with a dry cloth. \n\nConstruction:\nMade of high-strength glass material. \n\nOther Features:\nEasy to install, suitable for self-installation. \nMade in Japan. \n\nHave a question? Feel free to contact our customer service team, who will answer all your questions. "
}
[llm/start] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain > 5:llm:ChatOpenAI] Entering LLM run with input:
{
"prompts": [
"System: Use the following pieces of context to answer the users question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\nproduct_name: HDTV\ndescription: Specifications:\nDimensions: 50''. \n\nWhy we love it:\nOur HDTVs offer an immersive viewing experience with excellent picture quality and powerful sound. \n\nMaterial & Care:\nClean with a dry cloth. \n\nConstruction:\nMade of plastic, metal, and electronic components. \n\nOther Features:\nSupports network connection for online video viewing. \nComes with remote control. \nMade in Korea. \n\nHave questions? Feel free to contact our customer service team who will answer all your questions. <<<<>>>>>product_name: Air Purifier\ndescription: Specifications:\nDimensions: 15'' x 15'' x 20''. \n\nWhy we love it:\nOur air purifiers use advanced HEPA filtration technology to effectively remove particles and odors from the air, providing you with a fresh indoor environment. \n\nMaterial & Care:\nWipe with a dry cloth when cleaning. \n\nConstruction:\nMade of plastic and electronic components. \n\nOther Features:\nThree wind speeds with a timer. \nMade in Germany. \n\nHave questions? Feel free to contact our customer service team, who will answer all your questions. <<<<>>>>>product_name: Automatic Pet Feeder\ndescription: Specifications:\nDimensions: 14'' x 9'' x 15''. \n\nWhy we love it:\nOur automatic pet feeder can deliver food at a fixed time, allowing you to ensure your pet's diet whether you are at home or away. \n\nMaterial & Care:\nCan be cleaned with a damp cloth. \n\nConstruction:\nMade of plastic and electronic components. \n\nOther Features:\nEquipped with an LCD screen for easy operation. \nCan be set to deliver multiple times. \nMade in the USA. \n\nHave questions? Feel free to contact our customer service team who will answer all your questions. <<<<>>>>>product_name: Glass Protective Film\ndescription: Specifications:\nSuitable for mobile phone screens of all sizes. \n\nWhy we love it:\nOur glass protective film can effectively prevent your mobile phone screen from scratches and crackscracks, and does not affect the sensitivity of the touch. \n\nMaterial and Care:\nWipe with a dry cloth. \n\nConstruction:\nMade of high-strength glass material. \n\nOther Features:\nEasy to install, suitable for self-installation. \nMade in Japan. \n\nHave questions? Please feel free to contact our customer service team, they will answer all your questions. \nHuman: How to care for HDTV? "
]
}
[llm/end] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain > 5:llm:ChatOpenAI] [2.86s] Exiting LLM run with output:
{
"generations": [
[
{
"text": "HDTV care is very simple. You only need to use a dry cloth to clean it. Avoid using a wet cloth or chemical cleaners to avoid damaging the surface of the TV. ",
"generation_info": {
"finish_reason": "stop"},
"message": {
"lc": 1,
"type": "constructor",
"id": [
"langchain",
"schema",
"messages",
"AIMessage"
],
"kwargs": {
"content": "HDTV care is very simple. You only need to use a dry cloth to clean it. Avoid using a wet cloth or chemical cleaners to avoid damaging the surface of the TV.",
"additional_kwargs": {}
}
}
}
]
],
"llm_output": {
"token_usage": {"prompt_tokens": 823,
"completion_tokens": 58,
"total_tokens": 881
},
"model_name": "gpt-3.5-turbo"
},
"run": null
}
[chain/end] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain] [2.86s] Exiting Chain run with output:
{
"text": "The care of the HDTV is very simple. You only need to use a dry cloth to clean it. Avoid using a wet cloth or chemical cleaners to avoid damaging the surface of the TV."
}
[chain/end] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain] [2.87s] Exiting Chain run with output:
{"output_text": "HDTV care is very simple. You only need to use a dry cloth to clean it. Avoid using wet cloths or chemical cleaners to avoid damaging the surface of the TV."
}
[chain/end] [1:chain:RetrievalQA] [3.26s] Exiting Chain run with output:
{
"result": "HDTV care is very simple. You only need to use a dry cloth to clean it. Avoid using wet cloths or chemical cleaners to avoid damaging the surface of the TV."
}

'HDTV care is very simple. You only need to use a dry cloth to clean it. Avoid using wet cloths or chemical cleaners to avoid damaging the surface of the TV. '

We can see that it first goes deep into the retrieval QA chain, and then it goes into some document chains. As mentioned above, we are using the stuff method, and now we are passing this context, and you can see that this context is created by the different documents we retrieved. So when doing question answering, when an error result is returned, it is usually not the language model itself that is wrong, but actually the retrieval step that is wrong, and looking closely at the exact content and context of the question can help debug why it went wrong. 

We can then go down one level further and look at exactly what goes into the language model, andOpenAI itself, here we can see the full prompt passed, we have a system message with a description of the prompt used, this is the prompt used by the question answering chain, we can see the prompt printed out, answering the user's question using the following context snippet.

If you don't know the answer, just say you don't know, don't try to make up the answer. Then we see a bunch of context that was inserted before, and we can also see more information about the actual return type. We don't just return an answer, but also the usage of tokens, which can understand the usage of the number of tokens

Since this is a relatively simple chain, we can now see the final response, which is returned to the user through the chain. In this part, we mainly explain how to view and debug a single input to the chain.

## 3. Evaluation Example through LLM

Let's briefly sort out the process of question answering evaluation:

- First, we used LLM to automatically build a question answering test set, including questions and standard answers.

- Then, the same LLM tried to answer all the questions in the test set and got a response.

- Next, we need to evaluate whether the language model's answer is correct. The wonderful thing here is that we use another LLM chain to make a judgment, so LLM is both a "player" and a "referee".

Specifically, the first language model is responsible for answering questions. The second language model chain is used to determine the answer. Finally, we can collect the judgment results, get the effect score of the language model on this task. It should be noted that the language model chain for answering questions and the answer judgment chain are separate and have different responsibilities. This avoids the subjective judgment of the same model on its own results.

In short, the language model can automatically complete the whole process of building test sets, answering questions and judging answers, making the evaluation process more intelligent and automated. We only need to provide documents and parse the final results.

```python
langchain.debug = False

#Create predictions for all different examples
predictions = qa.apply(examples)

#Evaluate the predicted results, import QA question answering, evaluation chain, create this chain through the language model
from langchain.evaluation.qa import QAEvalChain #Import QA question answering, evaluation chain

#Evaluate by calling chatGPT
llm = ChatOpenAI(temperature=0)
eval_chain = QAEvalChain.from_llm(llm)

#Call evaluate on this chain for evaluation
graded_outputs = eval_chain.evaluate(examples, predictions)
```

> Entering new RetrievalQA chain...

> Finished chain.

> Entering new RetrievalQA chain...

> Finished chain.

> Entering new RetrievalQA chain...

> Finished chain.

> Entering new RetrievalQA chain...

> Finished chain.

> Entering new RetrievalQA chain...

> Finished chain.

> Entering new RetrievalQA chain...

> Finished chain.

> Entering new RetrievalQA chain...

> Finished chain.

```python
# We'll pass in examples and predictions, get a bunch of graded outputs, loop through them and print the answers
for i, eg in enumerate(examples):
print(f"Example {i}:")
print("Question: " + predictions[i]['query'])
print("Real Answer: " + predictions[i]['answer'])
print("Predicted Answer: " + predictions[i]['result'])
print("Predicted Grade: " + graded_outputs[i]['results'])
print()
```

Example 0:
Question: How do you take care of your HDTV?
Real Answer: Use a dry cloth to clean it.
Predicted Answer: It's very easy to take care of your HDTV. You just need to use a dry clothJust clean it. Avoid using damp cloths or chemical cleaners to avoid damaging the surface of the TV.
Predicted Grade: CORRECT

Example 1:
Question: Does the travel backpack have inner and outer pockets?
Real Answer: Yes.
Predicted Answer: Yes, the travel backpack has multiple practical inner and outer pockets to easily fit your essentials.
Predicted Grade: CORRECT

Example 2:
Question: What are the dimensions of this fully automatic coffee machine?
Real Answer: The large size is 13.8'' x 17.3'' and the medium size is 11.5'' x 15.2''.
Predicted Answer: This fully automatic coffee machine is available in two sizes:
- The large size is 13.8'' x 17.3''.
- The medium size is 11.5'' x 15.2''.
Predicted Grade: CORRECT

Example 3:
Question: What are the specifications of this electric toothbrush?
Real Answerer: General size - Height: 9.5'', Width: 1''.
Predicted Answer: The specifications of this electric toothbrush are: Height 9.5 inches, Width 1 inch.
Predicted Grade: CORRECT

Example 4:
Question: What is the name of this product?
Real Answer: The name of this product is Orange Flavored Vitamin C Effervescent Tablets.
Predicted Answer: The name of this product is Children's Educational Toys.
Predicted Grade: INCORRECT

Example 5:
Question: What are the dimensions of this wireless Bluetooth headset?
Real Answer: The dimensions of this wireless Bluetooth headset are 1.5'' x 1.3''.
Predicted Answer: The dimensions of this wireless Bluetooth headset are 1.5'' x 1.3''.
Predicted Grade: CORRECT

Example 6:
Question: What are the dimensions of this yoga mat?
Real Answer: The dimensions of this yoga mat areThe dimensions of this yoga mat are 24'' x 68''.
Predicted Answer: The dimensions of this yoga mat are 24'' x 68''.
Predicted Grade: CORRECT

From the above return results, we can see that each question in the evaluation results contains four groups of content: `Question`, `Real Answer`, `Predicted Answer` and `Predicted Grade`. Among them, `Real Answer` is the answer in the question and answer test set created by the previous `QAGenerateChain`, and `Predicted Answer` is the answer given by our `qa` chain. The last `Predicted Grade` is answered by `QAEvalChain` in the above code. 

In this chapter, we learned how to use the LangChain framework to realize the automatic evaluation of LLM question and answer effects. Unlike the traditional methods of manually preparing evaluation sets and judging questions one by one, LangChain automates the entire evaluation process. It can automatically build a test set containing question-answering samples, then use the language model to automatically generate responses to the test set, and finally automatically judge the accuracy of each answer through another model chain. **This fully automatic evaluation method greatly simplifies the question-answering system.The system evaluation and optimization process does not require developers to manually prepare test cases or judge the correctness one by one, which greatly improves work efficiency**.

With the help of LangChain's automatic evaluation function, we can quickly evaluate the question-answering effect of the language model on different document sets, and can continuously tune the model without manual intervention. This automated evaluation method frees our hands and allows us to iteratively optimize the performance of the question-answering system more efficiently.

In short, automatic evaluation is a major advantage of the LangChain framework. It will greatly reduce the threshold for question-answering system development and enable anyone to easily train a powerful question-answering model.

## English version tips

**1. Create an LLM application**

```python
from langchain.chains import RetrievalQA 
from langchain.chat_models import ChatOpenAI 
from langchain.document_loaders import CSVLoader 
from langchain.indexes import VectorstoreIndexCreator 
from langchain.vectorstores import DocArrayInMemorySearch 
from langchain.evaluation.qa import QAGenerateChain 
import pandas as pd

file = '../data/OutdoorClothingCatalog_1000.csv'

loader = CSVLoader(file_path=file)

data = loader.load()

test_data = pd.read_csv(file,skiprows=0,usecols=[1,2])

display(test_data.head())

llm = ChatOpenAI(temperature = 0.0)

index = VectorstoreIndexCreator(
vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

qa = RetrievalQA.from_chain_type(
llm=llm, 
chain_type="stuff", 
retriever=index.vectorstore.as_retriever(), 
verbose=True,
chain_type_kwargs = {
"document_separator": "<<<<>>>>>"
}
)

print(data[10],"\n")

print(data[11],"\n")

examples = [
{
"query": "Do the Cozy Comfort Pullover Set have side pockets?",
"answer": "Yes"
},
{
"query": "What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?",
"answer": "The DownTek collection"
}
]

example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI())

from langchain.evaluation.qa import QAGenerateChain #Import the QA generation chain, which will receive documents and create a question answer pair from each document
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI())#Create this chain by passing the chat open AI language model
new_examples = example_gen_chain.apply([{"doc": t} for t in data[:5]])

#View the use case data
print(new_examples)

examples += [ v for item in new_examples for k,v in item.items()]
qa.run(examples[0]["query"])
```

<div>
<style scoped>
.dataframe tbody tr th:only-of-type {
vertical-align: middle;
}

.dataframe tbody tr th {
vertical-align: top;
}

.dataframe thead th {
text-align: right;
}

</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>name</th>
<th>description</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Women's Campside Oxfords</td>
<td>This ultracomfortable lace-to-toe Oxford boasts...</td>
</tr>
<tr>
<th>1</th>
<td>Recycled Waterhog Dog Mat, Chevron Weave</td>
<td>Protect your floors from spills and splashing ...</td>
</tr>
<tr>
<th>2</th>
<td>Infant and Toddler Girls' Coastal Chill Swimsu...</td>
<td>She'll love the bright colors, ruffles and exc...</td>
</tr>
<tr>
<th>3</th>
<td>Refresh Swimwear, V-Neck Tankini Contrasts</td>
<td>Whether you're going for a swim or heading out...</td>
</tr>
<tr>
<th>4</th>
<td>EcoFlex 3L Storm Pants</td>
<td>Our new TEK O2 technology makes our four-seaso...</td>
</tr>
</tbody>
</table>
</div>

page_content=": 10\nname: Cozy Comfort Pullover Set, Stripe\ndescription: Perfect for lounging, this striped knit set lives up to its name. We used ultrasoft fabric and an easy design that's as comfortable at bedtime as it is when we have to make a quick run out.\n\nSize & Fit\n- Pants are Favorite Fit: Sits lower on the waist.\n- Relaxed Fit: Our most generous fit sits farthest from the body.\n\nFabric & Care\n- In the softest blend of 63% polyester, 35% rayon and 2% spandex.\n\nAdditional Features\n- Relaxed fit top with raglan sleeves and rounded hem.\n- Pull-on pants have a wide elastic waistband and drawstring, side pockets and a modern slim leg.\n\nImported." metadata={'source': '../data/OutdoorClothingCatalog_1000.csv', 'row': 10} 

page_content=': 11\nname: Ultra-Lofty 850 Stretch Down Hooded Jacket\ndescription: This technical stretch down jacket from our DownTek collection is sureto keep you warm and comfortable with its full-stretch construction providing exceptional range of motion. With a slightly fitted style that falls at the hip and best with a midweight layer, this jacket is suitable for light activity up to 20° and moderate activity up to -30°. The soft and durable 100% polyester shell offers complete windproof protection and is insulated with warm, lofty goose down. Other features include welded baffles for a no-stitch construction and excellent stretch, an adjusable hood, an interior media port and mesh stash pocket and a hem drawcord. Machine wash and dry. Imported.' metadata={'source': '../data/OutdoorClothingCatalog_1000.csv', 'row': 11} 

[{'qa_pairs': {'query': "What is the description of the Women's Campside Oxfords?", 'answer': "The description of the Women's Campside Oxfords is that they are an ultracomfortable lace-to-toe Oxford made of super-soft canvas. They have thick cushioning and quality construction, providing a broken-in feel from the first time they are worn."}}, {'qa_pairs': {'query': 'What are the dimensions of the small and medium sizes of the Recycled Waterhog Dog Mat, Chevron Weave?', 'answer': 'The dimensions of the small size of the Recycled Waterhog Dog Mat, Chevron Weave are 18" x 28". The dimensions of the medium size are 22.5" x 34.5".'}}, {'qa_pairs': {'query': "What are the features of the Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece?", 'answer': "The swimsuit has bright colors, ruffles, and exclusive whimsical prints. It is made of four-way-stretch and chlorine-resistant fabric, which keeps its shape and resists snags. The fabric is UPF 50+ rated, providing the highest rated sun protection possible by blocking 98% of the sun's harmful rays. The swimsuit also has crossover no-slip straps and a fully lined bottom for a secure fit and maximum coverage."}}, {'qa_pairs': {'query': 'What is the fabric composition of the Refresh Swimwear, V-Neck Tankini Contrasts?', 'answer': 'The Refresh Swimwear, V-Neck Tankini Contrasts is made of 82% recycled nylon and 18% Lycra® spandex for the body, and 90% recycled nylon with 10% Lycra® spandex for the lining.'}}, {'qa_pairs': {'query': 'What is the fabric composition of the EcoFlex 3L Storm Pants?', 'answer': 'The EcoFlex 3L Storm Pants are made of 100% nylon, exclusive of trim.'}}]

> Entering new RetrievalQA chain...

> Finished chain.

'Yes, the Cozy Comfort Pullover Set does have side pockets.'

**2. Manual evaluation**

```python
import langchain
langchain.debug = True

# Rerun the same example as above and you can see that it starts to print out more information
qa.run(examples[0]["query"])

langchain.debug = False
```

[chain/start] [1:chain:RetrievalQA] Entering Chain run with input:
{
"query": "Do the Cozy Comfort Pullover Set have side pockets?"
}
[chain/start] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain] Entering Chain run with input:
[inputs]
[chain/start] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain] Entering Chain run with input:
{
"question": "Do the Cozy Comfort Pullover Set have side pockets?",
"context": ": 10\nname: Cozy Comfort Pullover Set, Stripe\ndescription: Perfect for lounging, this striped knit set lives up to its name. We used ultrasoft fabric and an easy design that's as comfortable at bedtime as it is when we have to make a quick run out.\n\nSize & Fit\n- Pants are Favorite Fit: Sits lower on the waist.\n- Relaxed Fit: Our most generous fit sits farthest from the body.\n\nFabric & Care\n- In the softest blend of 63% polyester, 35% rayon and 2% spandex.\n\nAdditional Features\n- Relaxed fit top with raglan sleeves and rounded hem.\n- Pull-on pants have a wide elastic waistband and drawstring, side pockets and a modern slim leg.\n\nImported.<<<<>>>>>: 73\nname: Cozy Cuddles Knit Pullover Set\ndescription: Perfect for lounging, this knit set lives up to its name. We used ultrasoft fabric and an easy design that's as comfortable at bedtime as itis when we have to make a quick run out. \n\nSize & Fit \nPants are Favorite Fit: Sits lower on the waist. \nRelaxed Fit: Our most generous fit sits farthest from the body. \n\nFabric & Care \nIn the softest blend of 63% polyester, 35% rayon and 2% spandex.\n\nAdditional Features \nRelaxed fit top with raglan sleeves and rounded hem. \nPull-on pants have a wide elastic waistband and drawstring, side pockets and a modern slim leg. \nImported.<<<<>>>>>: 151\nname: Cozy Quilted Sweatshirt\ndescription: Our sweatshirt is an instant classic with its great quilted texture and versatile weight that easily transitions between seasons. With a traditional fit that is relaxed through the chest, sleeve, and waist, this pullover is lightweight enough to be worn most months of the year. The cotton blend fabric is super soft and comfortable, making it the perfect casual layer. To make dressing easy, this sweatshirt also features a snap placket and a heritage-inspired Mt. Katahdin logo patch. For care, machine wash and dry. Imported.<<<<>>>>>: 265\nname: Cozy Workout Vest\ndescription: For serious warmth that won't weigh you down, reach for this fleece-lined vest, which provides you with layering options whether you're inside or outdoors.\nSize & Fit\nRelaxed Fit. Falls at hip.\nFabric & Care\nSoft, textured fleece lining. Nylon shell. Machine wash and dry. \nAdditional Features \nTwo handwarmer pockets. Knit side panels stretch for a more flattering fit. Shell fabric is treated to resist waater and stains. Imported."
}
[llm/start] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain > 5:llm:ChatOpenAI] Entering LLM run with input:
{
"prompts": [
"System: Use the following pieces of context to answer the users question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n: 10\nname: Cozy Comfort Pullover Set, Stripe\ndescription: Perfect for lounging, this striped knit set livesup to its name. We used ultrasoft fabric and an easy design that's as comfortable at bedtime as it is when we have to make a quick run out.\n\nSize & Fit\n- Pants are Favorite Fit: Sits lower on the waist.\n- Relaxed Fit: Our most generous fit sits farthest from the body.\n\nFabric & Care\n- In the softest blend of 63% polyester, 35% rayon and 2% spandex.\n\nAdditional Features\n- Relaxed fit top with raglan sleeves and rounded hem.\n- Pull-on pants have a wide elastic waistband and drawstring,side pockets and a modern slim leg.\n\nImported.<<<<>>>>>: 73\nname: Cozy Cuddles Knit Pullover Set\ndescription: Perfect for lounging, this knit set lives up to its name. We used ultrasoft fabric and an easy design that's as comfortable at bedtime as it is when we have to make a quick run out. \n\nSize & Fit \nPants are Favorite Fit: Sits lower on the waist. \nRelaxed Fit: Our most generous fit sits farthest from the body. \n\nFabric & Care \nIn the softest blend of 63% polyester, 35% rayon and2% spandex.\n\nAdditional Features \nRelaxed fit top with raglan sleeves and rounded hem. \nPull-on pants have a wide elastic waistband and drawstring, side pockets and a modern slim leg. \nImported.<<<<>>>>>: 151\nname: Cozy Quilted Sweatshirt\ndescription: Our sweatshirt is an instant classic with its great quilted texture and versatile weight that easily transitions between seasons. With a traditional fit that is relaxed through the chest, sleeve, and waist, this pullover is lightweight enough to comfortably fit your style.gh to be worn most months of the year. The cotton blend fabric is super soft and comfortable, making it the perfect casual layer. To make dressing easy, this sweatshirt also features a snap placket and a heritage-inspired Mt. Katahdin logo patch. For care, machine wash and dry. Imported.<<<<>>>>>: 265\nname: Cozy Workout Vest\ndescription: For serious warmth that won't weigh you down, reach for this fleece-lined vest, which provides you with layering options whether you're inside or outdoors.\nSize & Fit\nRelaxed Fit. Falls at hip.\nFabric & Care\nSoft, textured fleece lining. Nylon shell. Machine wash and dry. \nAdditional Features \nTwo handwarmer pockets. Knit side panels stretch for a more flattering fit. Shell fabric is treated to resist water and stains. Imported.\nHuman: Do the Cozy Comfort Pullover Set have side pockets?"
]
}
[llm/end] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain > 5:llm:ChatOpenAI] [879.746ms] Exiting LLM run with output:{
"generations": [
[
{
"text": "Yes, the Cozy Comfort Pullover Set does have side pockets.",
"generation_info": {
"finish_reason": "stop"
},
"message": {
"lc": 1,
"type": "constructor",
"id": [
"langchain",
"schema",
"messages",
"AIMessage"
],
"kwargs": {
"content": "Yes, the Cozy Comfort Pullover Set does have side pockets.",
"additional_kwargs": {}
}
}
}
]
],
"llm_output": {
"token_usage": {
"prompt_tokens": 626,
"completion_tokens": 14,
"total_tokens": 640
},
"model_name": "gpt-3.5-turbo"
},
"run": null
}
[chain/end] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain > 4:chain:LLMChain] [880.5269999999999ms] Exiting Chain run with output:
{
"text": "Yes, the Cozy Comfort Pullover Set does have side pockets."
}
[chain/end] [1:chain:RetrievalQA > 3:chain:StuffDocumentsChain] [881.44999999999999ms] Exiting Chain run with output:
{
"output_text": "Yes, the Cozy Comfort Pullover Set does have side pockets."
}
[chain/end] [1:chain:RetrievalQA] [1.21s] Exiting Chain run with output:
{
"result": "Yes, the Cozy Comfort Pullover Set does have side pockets."}

**3. Evaluation example through LLM**

```python
langchain.debug = False

#Create predictions for all different examples
predictions = qa.apply(examples)

#Evaluate the predicted results, import QA question answering, evaluation chain, create this chain through language model
from langchain.evaluation.qa import QAEvalChain #Import QA question answering, evaluation chain

#Evaluate by calling chatGPT
llm = ChatOpenAI(temperature=0)
eval_chain = QAEvalChain.from_llm(llm)

#Call evaluate on this chain for evaluation
graded_outputs = eval_chain.evaluate(examples, predictions)

#We will pass in examples and predictions, get a bunch of graded outputs, loop through them and print the answers
for i, eg in enumerate(examples):
print(f"Example {i}:")
print("Question: " +predictions[i]['query'])
print("Real Answer: " + predictions[i]['answer'])
print("Predicted Answer: " + predictions[i]['result'])
print("Predicted Grade: " + graded_outputs[i]['results'])
print()
```

> Entering new RetrievalQA chain...

> Finished chain.

> Entering new RetrievalQA chain...

> Finished chain.

> Entering new RetrievalQA chain...

> Finished chain.

> Entering new RetrievalQA chain...> Finished chain.

> Entering new RetrievalQA chain...

> Finished chain.

> Entering new RetrievalQA chain...

> Finished chain.

> Entering new RetrievalQA chain...

> Finished chain.

Example 0:
Question: Do the Cozy Comfort Pullover Set have side pockets?
Real Answer: Yes
Predicted Answer: Yes, the Cozy Comfort Pullover Set does have side pockets.
Predicted Grade: CORRECT

Example 1:
Questionion: What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?
Real Answer: The DownTek collection
Predicted Answer: The Ultra-Lofty 850 Stretch Down Hooded Jacket is from the DownTek collection.
Predicted Grade: CORRECT

Example 2:
Question: What is the description of the Women's Campside Oxfords?
Real Answer: The description of the Women's Campside Oxfords is that they are an ultracomfortable lace-to-toe Oxford made of super-soft canvas. They have thickcushioning and quality construction, providing a broken-in feel from the first time they are worn.
Predicted Answer: The description of the Women's Campside Oxfords is: "This ultracomfortable lace-to-toe Oxford boasts a super-soft canvas, thick cushioning, and quality construction for a broken-in feel from the first time you put them on."
Predicted Grade: CORRECT

Example 3:
Question: What are the dimensions of the small and medium sizes of the Recycled Waterhog Dog Mat, Chevron Weave?
Real Answer: The dimensions of the small size of the Recycled Waterhog Dog Mat, Chevron Weave are 18" x 28". The dimensions of the medium size are 22.5" x 34.5".
Predicted Answer: The dimensions of the small size of the Recycled Waterhog Dog Mat, Chevron Weave are 18" x 28". The dimensions of the medium size are 22.5" x 34.5".
Predicted Grade: CORRECT

Example 4:
Question: What are the features of the Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece?Real Answer: The swimsuit has bright colors, ruffles, and exclusive whimsical prints. It is made of four-way-stretch and chlorine-resistant fabric, which keeps its shape and resists snags. The fabric is UPF 50+ rated, providing the highest rated sun protection possible by blocking 98% of the sun's harmful rays. The swimsuit also has crossover no-slip straps and a fully lined bottom for a secure fit and maximum coverage.
Predicted Answer: The features of the Infant and Toddler Girls' CoaStal Chill Swimsuit, Two-Piece are:

- Bright colors and ruffles
- Exclusive whimsical prints
- Four-way-stretch and chlorine-resistant fabric
- UPF 50+ rated fabric for sun protection
- Crossover no-slip straps
- Fully lined bottom for a secure fit and maximum coverage
- Machine washable and line dry for best results
- Imported
Predicted Grade: CORRECT

Example 5:
Question: What is the fabric composition of the Refresh Swimwear, V-Neck TankiniContrasts?
Real Answer: The Refresh Swimwear, V-Neck Tankini Contrasts is made of 82% recycled nylon and 18% Lycra® spandex for the body, and 90% recycled nylon with 10% Lycra® spandex for the lining.
Predicted Answer: The fabric composition of the Refresh Swimwear, V-Neck Tankini Contrasts is 82% recycled nylon with 18% Lycra® spandex for the body, and 90% recycled nylon with 10% Lycra® spandex for the lining.
Predicted Grade: CORRECT

Example 6:
Question: What is the fabric composition of the Refresh Swimwear, V-Neck Tankini Contrasts?abric composition of the EcoFlex 3L Storm Pants?
Real Answer: The EcoFlex 3L Storm Pants are made of 100% nylon, exclusive of trim.
Predicted Answer: The fabric composition of the EcoFlex 3L Storm Pants is 100% nylon, exclusive of trim.
Predicted Grade: CORRECT