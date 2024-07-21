# Data Processing

The source code for this article is here (https://github.com/datawhalechina/llm-universe/blob/main/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93/3.%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86.ipynb). If you need to reproduce, you can download and run the source code.

To build our local knowledge base, we need to process local documents stored in various types, read local documents, and convert the content of local documents into word vectors through the Embedding method described above to build a vector database. In this section, we start with some practical examples to explain how to process local documents.
## 1. Source document selection
We use some classic open source courses from Datawhale as examples, including:
* [《Machine Learning Formula Explanation》PDF version](https://github.com/datawhalechina/pumpkin-book/releases)
* [《LLM Introduction Tutorial for Developers, Part 1 Prompt Engineering》md version](https://github.com/datawhalechina/llm-cookbook) 
We put the source data of the knowledge base in the ../data_base/knowledge_db directory.
## 2. Data reading
### 1. PDF document
We can use LangChain's PyMuPDFLoader to read the PDF file of the knowledge base. PyMuPDFLoader is the fastest PDF parser. The result will contain detailed metadata of the PDF and its pages, and return one document per page.

```python
from langchain.document_loaders.pdf import PyMuPDFLoader

# Create a PyMuPDFLoader Class instance, input as the pdf document path to be loaded
loader = PyMuPDFLoader("../../data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")

# Call the function load of PyMuPDFLoader Class to load the pdf file
pdf_pages = loader.load()
```

After the document is loadedStored in the `pages` variable:
- The variable type of `page` is `List`
- Print the length of `pages` to see how many pages the pdf contains

```python
print(f"The variable type after loading is: {type(pdf_pages)}，", f"The PDF contains a total of {len(pdf_pages)} pages")
```

The variable type after loading is: <class 'list'>, The PDF contains a total of 196 pages

Each element in `page` is a document, the variable type is `langchain_core.documents.base.Document`, the document variable type contains two attributes
- `page_content` contains the content of the document.
- `meta_data` is the descriptive data related to the document.

```python
pdf_page = pdf_pages[1]
print(f"The type of each element: {type(pdf_page)}.", 
f"Descriptive data of the document: {pdf_page.metadata}", 
f"View the content of the document:\n{pdf_page.page_content}", 
sep="\n------\n")
```

The type of each element: <class 'langchain_core.documents.base.Document'>.
------
Descriptive data of the document: {'source': './data_base/knowledge_db/pumkin_book/pumpkin_book.pdf', 'file_path': './data_base/knowledge_db/pumkin_book/pumpkin_book.pdf', 'page': 1, 'total_pages': 196, 'format': 'PDF 1.5', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'creator': 'LaTeX with hyperref', 'producer': 'xdvipdfmx (20200315)', 'creationDate': "D:20230303170709-00'00'", 'modDate': '', 'trapped': ''}
------
View the content of this document:
Preface
"Zhou Zhihua's "Machine Learning" (Xigua Book) is one of the classic introductory textbooks in the field of machine learning. In order to make as many readers as possible understand machine learning through Xigua Book, Zhou did not elaborate on the derivation details of some formulas in the book, but this may be "not very friendly" to those readers who want to delve into the details of formula derivation. This book aims to analyze the formulas that are more difficult to understand in Xigua Book, and to supplement some formulas with specific derivation details.
"
Reading this, you may wonder why the previous paragraph is quoted, because this is just our initial reverie. Later we learned that the real reason why Zhou omitted these derivation details is that he himself believes that "sophomore students with a solid foundation in science and engineering mathematics should have no difficulty with the derivation details in Xigua Book. The key points are all in the book, and the omitted details should be able to be supplemented by imagination or practice"
. So... This Pumpkin Book can only be regarded as the notes taken down by me and other math idiots when I was studying on my own. I hope it can help everyone become a qualified "sophomore student with a solid foundation in science and engineering mathematics". Instructions for use: • All the contents of the Pumpkin Book are based on the contents of the Watermelon Book.The best way to use the Pumpkin Book is to use the Watermelon Book as the main thread, and then consult the Pumpkin Book when you encounter a formula that you cannot derive or understand;
• For beginners in machine learning, it is strongly recommended not to delve into the formulas in Chapter 1 and Chapter 2 of the Watermelon Book. Just go through them briefly, and you can come back to read them when you are a little more advanced;
• We strive to explain the analysis and derivation of each formula from the perspective of undergraduate mathematics foundation, so we usually provide the mathematical knowledge that is beyond the syllabus in the form of appendices and references. Interested students can continue to study in depth based on the materials we provide;
• If the Pumpkin Book does not have the formula you want to check,
or you find an error in the Pumpkin Book,
please do not hesitate to go to our GitHub
Issues (address: https://github.com/datawhalechina/pumpkin-book/issues) for feedback, and submit the formula number or errata information you want to add in the corresponding section. We usually respond to you within 24 hours. If it exceeds 24 hours, we will reply to you. If you haven’t responded within 1 hour, please contact us on WeChat (WeChat ID: at-Sm1les)
;
Video tutorial: https://www.bilibili.com/video/BV1Mh411e7VU
Online reading address: https://datawhalechina.github.io/pumpkin-book (only for the first edition)
Latest PDF version: https://github.com/datawhalechina/pumpkin-book/releases
Editorial Board
Editor-in-Chief: Sm1les, archwalker, jbb0523
Editorial Board: juxiao, Majingmin, MrBigFan, shanry, Ye980226
Cover design: Concept - Sm1les, Creation - Lin Wang Maosheng
Acknowledgements
Special thanks to awyd234,
feijuan,
Ggmatch,
Heitao5200,
huaqing89,
LongJH,
LilRachel,
LeoLRH,
Nono17,
spareribs, sunchaothu, StevenLzq for their contributions to Pumpkin Book in the earliest days.
Scan the QR code below and reply with the keyword "Pumpkin Book", you can join the "Pumpkin Book Reader Exchange Group"
Copyright Statement
This work is licensed under the Creative Commons Attribution-Noncommercial-Share Alike 4.0 International License.

### 2. MD Document
We can read in markdown documents in almost the same way:

```python
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("../../data_base/knowledge_db/prompt_engineering/1. Introduction.md")
md_pages = loader.load()
```

The object read is exactly the same as the PDF document:

```python
print(f"The variable type after loading is: {type(md_pages)}，", f"The Markdown contains a total of {len(md_pages)} pages")
```

The variable type after loading is: <class 'list'>, the Markdown contains 1 page in total

```python
md_page = md_pages[0]
print(f"The type of each element: {type(md_page)}.", 
f"Descriptive data of the document: {md_page.metadata}", 
f"View the content of the document:\n{md_page.page_content[0:][:200]}", 
sep="\n------\n")
```

The type of each element: <class 'langchain_core.documents.base.Document'>.
------
Descriptive data of the document: {'source': './data_base/knowledge_db/prompt_engineering/1. Introduction Introduction.md'}
------
View the content of the document:
Chapter 1 Introduction

Welcome to the prompt engineering section for developers. This section is based on the "Prompt Engineering for Developer" course by Andrew Ng. Prompt EnGineering for Developer" is taught by Professor Andrew Ng and Isa Fulford, a member of the OpenAI technical team. Isa has developed the popular ChatGPT retrieval plug-in and is teaching LLM (Larg

## 3. Data cleaning
We expect the data in the knowledge base to be as orderly, high-quality, and concise as possible, so we need to delete low-quality text data that even affects understanding. 
It can be seen that the PDF file read above not only adds a line break `\n` to a sentence according to the line breaks of the original text, but also inserts `\n` between the original two symbols. We can use regular expressions to match and delete `\n`.

```python
import re
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content) print(pdf_page.page_content) ```Preface
"Zhou Zhihua's "Machine Learning" (Xigua Book) is one of the classic introductory textbooks in the field of machine learning. In order to make as many readers as possible understand machine learning through Xigua Book, Zhou did not elaborate on the derivation details of some formulas in the book, but this may be "not very friendly" to those readers who want to delve into the details of formula derivation. This book aims to analyze the formulas that are more difficult to understand in Xigua Book, and to supplement the specific derivation details of some formulas."
After reading this, you may wonder why the previous paragraph is quoted, because this is just our initial reverie. Later we learned that the real reason why Zhou omitted these derivation details is that he himself believes that "sophomore students with a solid foundation in science and engineering mathematics should have no difficulty with the derivation details in Xigua Book. The key points are all in the book, and the omitted details should be able to be supplemented or practiced." So... This Pumpkin Book can only be regarded as the notes taken by math losers like me when I was studying by myself. I hope it can help everyone become a qualified "sophomore student with a solid foundation in science and engineering mathematics."
Instructions for use
• All the contents of the Pumpkin Book are expressed based on the contents of the Watermelon Book as the prerequisite knowledge, so the best way to use the Pumpkin Book is to use the Watermelon Book as the main thread, and then consult the Pumpkin Book when you encounter a formula that you cannot derive or understand; • ForFor beginners of machine learning, it is strongly recommended not to study the formulas in Chapter 1 and Chapter 2 of the Watermelon Book in depth. Just go through them briefly. You can come back to study them when you are a little more advanced. • We strive to explain the analysis and derivation of each formula from the perspective of undergraduate mathematics foundation, so we usually provide the mathematical knowledge beyond the syllabus in the form of appendices and references. Interested students can continue to study in depth based on the materials we provide. • If the formula you want to check is not available in the Pumpkin Book, or if you find an error in the Pumpkin Book, please do not hesitate to go to our GitHub Issues (address: https://github.com/datawhalechina/pumpkin-book/issues) for feedback, and submit the formula number or errata information you want to add in the corresponding section. We usually respond to you within 24 hours. If you do not respond within 24 hours, you can contact us on WeChat (WeChat ID: at-Sm1les); Supporting video tutorial: https://www.bilibili.com/video/BV1Mh411e7VU
Online reading address: https://datawhalechina.github.io/pumpkin-book (only for the 1st edition)
The latest PDF version can be obtained at: https://github.com/datawhalechina/pumpkin-book/releases
Editorial Board
Editors-in-Chief: Sm1les, archwalker, jbb0523
Editorial Board: juxiao, Majingmin, MrBigFan, shanry, Ye980226
Cover Design: Concept - Sm1les, Creation - Lin Wang Maosheng
Acknowledgements
Special thanks to awyd234, feijuan, Ggmatch, Heitao5200, huaqing89, LongJH, LilRachel, LeoLRH, Nono17, spareribs, sunchaothu, StevenLzq for their contributions to Pumpkin Book in the earliest days.
Scan the QR code below and reply with the keyword "Pumpkin Book" to join the "Pumpkin Book Reader Exchange Group"
Copyright Statement
This work is licensed under the Creative Commons Attribution-Noncommercial-Share Alike 4.0 International License.

Further analyzing the data, we found that there are still a lot of `•` and spaces in the data, and our simple and practical replace method can be used.

```pythonn
pdf_page.page_content = pdf_page.page_content.replace('•', '')
pdf_page.page_content = pdf_page.page_content.replace(' ', '')
print(pdf_page.page_content)
```

Preface
"Zhou Zhihua's Machine Learning (Xigua Book) is one of the classic introductory textbooks in the field of machine learning. In order to make as many readers as possible understand machine learning through Xigua Book, Zhou did not elaborate on the derivation details of some formulas in the book, but this may be "not very friendly" to those readers who want to delve into the details of formula derivation. This book aims to analyze the formulas that are more difficult to understand in Xigua Book, and to supplement some formulas with specific derivation details."

Reading this, you may wonder why the previous paragraph is quoted, because this is just our initial reverie. Later we learned that the real reason why Zhou omitted these derivation details is that he himself believes that "sophomore students with a solid foundation in science and engineering mathematics should have a good understanding of Xigua Book.
The derivation details in the book should not be difficult, the key points are all in the book, and the omitted details should be supplemented by imagination or exercises. So... this pumpkin book can only be regarded asI
and other math idiots took down notes when I was studying on my own, hoping to help everyone become a qualified "sophomore
second year student with a solid foundation in science and engineering mathematics".
Instructions for use
All the contents of Pumpkin Book are expressed with the contents of Watermelon Book as the prerequisite knowledge, so the best way to use Pumpkin Book is to use Watermelon Book
as the main line, and then consult Pumpkin Book when you encounter a formula that you cannot derive or understand; for beginners of machine learning, it is strongly recommended not to study the formulas in Chapter 1 and Chapter 2 of Watermelon Book in depth, just go through them briefly, and come back to read them when you are a little bit drifting; we strive to explain the analysis and derivation of each formula from the perspective of undergraduate mathematics foundation, so we usually give the mathematical knowledge beyond the syllabus in the form of appendices and references, and interested students can continue to study in depth according to the materials we provide; if the formula you want to check is not in Pumpkin Book,
or you find an error in Pumpkin Book,
please do not hesitate to go to our GitHub
Issues (address: https://github.com/datawhalechina/pumpkin-book/issues) for feedback, and submit the formula number or errata information you want to add in the corresponding section, and we will usually reply to you within 24 hours, if you haven’t responded for more than 24 hours
, you can contact us on WeChat (WeChat ID: at-Sm1les);

Supporting video tutorial: https://www.bilibili.com/video/BV1Mh411e7VU

Online reading address: https://datawhalechina.github.io/pumpkin-book (only for the first edition)

Latest PDF access address: https://github.com/datawhalechina/pumpkin-book/releases

Editorial Board

Editor-in-chief: Sm1les, archwalker, jbb0523

Editorial Board: juxiao, Majingmin, MrBigFan, shanry, Ye980226

Cover design: Conception-Sm1les, Creation-Lin Wang Maosheng

Acknowledgements
Special thanks to awyd234, feijuan, Ggmatch, Heitao5200, huaqing89, LongJH, LilRachel, LeoLRH, Nono17, spareribs, sunchaothu, StevenLzq for their contributions to Pumpkin Book in the early days.
Scan the QR code below,Then reply with the keyword "Pumpkin Book" to join the "Pumpkin Book Reader Exchange Group"
Copyright Statement
This work is licensed under the Creative Commons Attribution-Noncommercial-Share Alike 4.0 International License.

The md file read above has a line break between each paragraph, which we can also remove using the replace method.

```python
md_page.page_content = md_page.page_content.replace('\n\n', '\n')
print(md_page.page_content)
```

Chapter 1 Introduction
Welcome to the prompt engineering section for developers. This section is based on the "Prompt Engineering for Developer" course by Andrew Ng. The "Prompt Engineering for Developer" course is taught by Andrew Ng and Isa Fulford, a member of the OpenAI technical team. Isa has developed the popular ChatGPT retrieval plug-in and has made great contributions to teaching the application of LLM (Large Language Model) technology in products. She also co-wroteOpenAI cookbook that teaches people to use prompts. We hope that through this module, we can share with you the best practices and tips for developing LLM applications using prompts.
There is a lot of material on the web about prompt design (we will keep this term in this tutorial), such as articles like "30 prompts everyone has to know", which mainly focus on ChatGPT's web interface, which many people use to perform specific, usually one-off tasks. But we think that for developers, the more powerful function of large language models (LLMs) is to be able to call them through API interfaces to quickly build software applications. In fact, we know that the team at AI Fund, DeepLearning.AI's sister company, has been working with many startups to apply these technologies to many applications. It's exciting to see that the LLM API allows developers to build applications very quickly.
In this module, we will share with readers various tips and best practices to improve the application of large language models. The book covers a wide range of typical application scenarios of language models, including software development prompt design, text summarization, reasoning, transformation, expansion, and building chatbots. We sincerely hope that this course will inspire readers’ imagination to develop better applications of language models.As LLMs have evolved, they can be roughly divided into two types, which will be referred to as basic LLMs and instruction-tuned LLMs. Basic LLMs are models that are trained to predict the next word based on text training data. They are usually trained on large amounts of data from the Internet and other sources to determine the most likely word that will appear next. For example, if you use "Once upon a time, there was a unicorn" as a prompt, the basic LLM may continue to predict "She lived in a magical forest with her unicorn friend." However, if you use "What is the capital of France" as a prompt, the basic LLM may predict the answer as "What is the largest city in France? What is the population of France?" based on an article on the Internet, because the article on the Internet is likely to be a list of question-and-answer questions about the country of France.
Unlike basic language models, instruction-tuned LLMs are specifically trained to better understand and follow instructions. For example, when asked "What is the capital of France?", this type of model is likely to directly answer "The capital of France is Paris." Instruction fine-tuning LLM training is usually based on pre-trained language models. Pre-training is first performed on large-scale text data to master the basic laws of language. On this basis, further training and fine-tuning are performed. The input is instructions, and the output is the correct response to these instructions. Sometimes RLHF (reinforcement learning from human feedback, human feedback reinforcement learning) technology, further enhances the model's ability to follow instructions based on human feedback on the model's output. Through this controlled training process. Instruction fine-tuning LLM can generate outputs that are highly sensitive to instructions, safer and more reliable, and less irrelevant and damaging content. Therefore. Many practical applications have turned to using such large language models.
Therefore, this course will focus on best practices for instruction fine-tuning LLM, and we also recommend that you use it for most usage scenarios. When you use instruction fine-tuning LLM, you can analogize it to providing instructions to another person (assuming that he is smart but does not know the specific details of your task). Therefore, when LLM does not work properly, sometimes it is because the instructions are not clear enough. For example, if you want to ask "Please write something about Alan Turing for me", it may be more helpful to make it clear that you want the text to focus on his scientific work, personal life, historical role, or other aspects. In addition, you can also specify the tone of the answer to better meet your needs. Options include professional journalist writing, or essays written to friends, etc.
If you treat the LLM as a recent college graduate and ask them to complete this task, you can even specify in advance which text fragments they should read to write about Alan Turing, which can helpThis newly graduated college student completed this task better. The next chapter of this book will elaborate on the two key principles of prompt word design: clarity and giving enough time to think.

## 4. Document segmentation

Since the length of a single document often exceeds the context supported by the model, the retrieved knowledge is too long to be processed by the model. Therefore, in the process of building a vector knowledge base, we often need to segment the document, split a single document into several chunks according to length or fixed rules, and then convert each chunk into a word vector and store it in the vector database.

When searching, we will use chunk as the meta-unit of retrieval, that is, each time we retrieve k chunks as knowledge that the model can refer to to answer user questions, this k is freely set.

In Langchain, the text segmenter is segmented according to `chunk_size` (chunk size) and `chunk_overlap` (the overlap size between chunks).

![image.png](../figures/C3-3-example-splitter.png)

* chunk_size refers to the number of characters or tokens (such as words, sentences, etc.) contained in each chunk

* chunk_overlap refers to the number of characters shared between two chunks, which is used to maintain the coherence of the context.Avoid losing context information during segmentation

Langchain provides multiple ways to segment documents, which differ in how to determine the boundaries between blocks, which characters/tokens make up a block, and how to measure the size of a block

- RecursiveCharacterTextSplitter(): Split text by string, recursively trying to split text by different delimiters.
- CharacterTextSplitter(): Split text by character.
- MarkdownHeaderTextSplitter(): Split markdown files based on the specified title.
- TokenTextSplitter(): Split text by token.
- SentenceTransformersTokenTextSplitter(): Split text by token
- Language(): For CPP, Python, Ruby, Markdown, etc.
- NLTKTextSplitter(): Split text by sentence using NLTK (Natural Language Toolkit).
- SpacyTextSplitter(): Use Spacy to cut text by sentence.

```python
''' 
* RecursiveCharacterTextSplitter Recursive Character Text Splitter
RecursiveCharacterTextSplitter will recursively split by different characters (according to this priority ["\n\n", "\n", " ", ""]),
so that all semantically related content can be kept in the same place for as long as possible
RecursiveCharacterTextSplitter needs to pay attention to 4 parameters:

* separators - separator string array
* chunk_size - character limit for each document
* chunk_overlap - length of the overlapping area of ​​two documents
* length_function - length calculation function
'''
#Import text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

```python
# Length of a single paragraph of text in the knowledge base
CHUNK_SIZE = 500

# Length of overlapping adjacent texts in the knowledge base
OVERLAP_SIZE = 50
```

```python
# Use recursive character text splitter
text_splitter =RecursiveCharacterTextSplitter( chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE ) text_splitter.split_text(pdf_page.page_content[0:1000]) ```` ['Preface\n"Zhou Zhihua's "Machine Learning" (Xigua Book) is one of the classic introductory textbooks in the field of machine learning. In order to enable as many readers as possible to understand machine learning through Xigua Book, Therefore, the derivation details of some formulas are not described in detail in the book, but this may be "not very friendly" to readers who want to delve into the details of the derivation of formulas. This book aims to explain the formulas that are more difficult to understand in the watermelon book. Analyze it, and add some specific derivation details to some formulas. "After reading this, you may wonder why the previous paragraph is in quotation marks, because this is just our initial reverie. Later we learned that Zhou \nThe real reason why the teacher omitted these derivation details is that he believes that "sophomore students with a solid foundation in science and engineering mathematics should have no difficulty with the derivation details in the watermelon book. The key points are all in the book. The omitted details should be filled in by imagination or practice. Therefore... This Pumpkin Book can only be regarded as the notes taken by me and other math losers when I was self-studying. I hope it can help everyone become A qualified "Sophomore students with a solid foundation in engineering mathematics". Instructions for use: All the contents of Pumpkin Book are expressed based on the contents of Watermelon Book as the prerequisite knowledge, so the best way to use Pumpkin Book is to use Watermelon Book as the main line, and then consult Pumpkin Book when you encounter formulas that you cannot derive or understand; for beginners in machine learning, it is strongly recommended not to delve into the formulas in Chapter 1 and Chapter 2 of Watermelon Book, just go through them briefly, and wait until you learn',
'It's not too late to come back and study when you feel a little bit off; We strive to explain the analysis and derivation of each formula from the perspective of undergraduate mathematics foundation, so we usually provide the mathematical knowledge beyond the syllabus in the form of appendices and references. Interested students can continue to study in depth based on the materials we provide; If the Pumpkin Book does not have the formula you want to look up, or you find an error in the Pumpkin Book, please do not hesitate to go to our GitHub\nIssues (address: https://github.com/datawhalechina/pumpkin-book/issues) for feedback, and submit the formula number or errata information you want to add in the corresponding section. We will usually reply to you within 24 hours. If you do not reply within 24 hours, you can contact us on WeChat (WeChat ID: at-Sm1les); \nSupporting video tutorial: https://www.bilibili.com/video/BV1Mh411e7VU\nOnline reading address: https://datawhalechina.github.io/pumpkin-book (only for the first edition)\nLatest PDF access address: https://github.com/datawhalechina/pumpkin-book/releases\nEditorial Board',
'Editorial Board\nEditor-in-Chief: Sm1les, archwalk']

```python
split_docs = text_splitter.split_documents(pdf_pages)
print(f"Number of files after segmentation: {len(split_docs)}")
```

Number of files after segmentation: 720

```python
print(f"Number of characters after segmentation (can be used to roughly evaluate the number of tokens): {sum([len(doc.page_content) for doc in split_docs])}")
```

Number of characters after segmentation (can be used to roughly evaluate the number of tokens) Number): 308931

Note: How to segment documents is actually the most important step in data processing, which often determinesThe lower limit of the retrieval system. However, how to choose the segmentation method is often highly business-related - for different businesses and different source data, it is often necessary to set a personalized document segmentation method. Therefore, in this chapter, we simply segment the document according to chunk_size. For readers who are interested in further exploration, please read our project examples in Part 3 to refer to how existing projects perform document segmentation.

The corresponding source code for this article is [here](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93/3.%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86.ipynb), if you need to reproduce, you can download and run the source code.