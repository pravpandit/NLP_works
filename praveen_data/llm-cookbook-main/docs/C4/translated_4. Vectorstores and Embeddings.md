# Chapter 4 Vectorstores and Embeddings

Let's review the overall workflow of Retrieval Enhancement Generation (RAG):

![overview.jpeg](../figures/C4/overview.png)

<div align='center'> Figure 4.4 Overall workflow of Retrieval Enhancement Generation </div>

In the first two lessons, we discussed `Document Loading` and `Splitting`.

Below we will use the knowledge from the first two lessons to load and split the document.

## 1. Read the document

The following document is the link of the official open source matplotlib tutorial of datawhale https://datawhalechina.github.io/fantastic-matplotlib/index.html , which can be downloaded from the website.

Note that this chapter requires the installation of third-party libraries `pypdf` and `chromadb`

```python
from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders_chinese = [
# Deliberately add duplicate documents to confuse the data
PyPDFLoader("docs/matplotlib/First round: Matplotlib first encounter.pdf"),
PyPDFLoader("docs/matplotlib/First round: Matplotlib first encounter.pdf"),
PyPDFLoader("docs/matplotlib/Second round: Art brush sees the universe.pdf"),
PyPDFLoader("docs/matplotlib/Third round: Layout format determines the square and circle.pdf")
]
docs = []
for loader in loaders_chinese:
docs.extend(loader.load())
```

After the document is loaded, we can use `RecursiveCharacterTextSplitter` (recursive character text splitter) to create blocks.

```python
# Split text
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
chunk_size = 1500, # The size of each text chunk. This means that each time you split the text, you will try to make each chunk contain 1500 characters.
chunk_overlap = 150 # The overlap between each text chunk.
)

splits = text_splitter.split_documents(docs)

print(len(splits))
```

27

## 2. Embeddings

What is `Embeddings`?

In machine learning and natural language processing (NLP), `Embeddings` is a technique for converting categorical data, such as words, sentences, or entire documents, into real vectors. These real vectors can be better understood and processed by computers. The main idea behind embeddings is that similar or related objects should be close to each other in the embedding space.

For example, we can use word embeddings to represent text data. In word embeddings, each word is converted into a vector that captures the semantic information of the word. For example, the words "king" and "queen" will be very close in the embedding space because they have similar meanings."pple" and "orange" will also be close because they are both fruits. "king" and "apple" will be far apart in the embedding space because they have different meanings.

Let's take our segments and embed them.

```python
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
```

Before using examples with real document data, let's try it with a few test case sentences to get a feel for embeddings.

Here are a few example sentences, the first two of which are very similar and the third is irrelevant. We can then create an embedding for each sentence using the embedding class.

```python
sentence1_chinese = "I like dogs"
sentence2_chinese = "I like canines"
sentence3_chinese = "The weather outside is terrible"

embedding1_chinese = embedding.embed_query(sentence1_chinesese)
embedding2_chinese = embedding.embed_query(sentence2_chinese)
embedding3_chinese = embedding.embed_query(sentence3_chinese)
```

We can then use `numpy` to compare them and see which ones are the most similar.

We expect that the first two sentences should be very similar.

Then, the first and second should be very different compared to the third.

We will use the dot product to compare the two embeddings.

If you don't know what a dot product is, that's OK. The important thing you just need to know is that the higher the score, the more similar the sentences are.

```python
import numpy as np

np.dot(embedding1_chinese, embedding2_chinese)
```

0.9440614936689298

We can see that the scores for the first two `embeddings` are pretty high at 0.94.

```python
np.dot(embedding1_chinese, embedding3_chinese)
```

0.792186975021313If we compare the first `embedding` to the third `embedding`, we can see that it is significantly lower, about 0.79.

```python
np.dot(embedding2_chinese, embedding3_chinese)
```

0.7804109942586283

We compare the second `embedding` to the third `embedding`, and we can see that its score is about 0.78.

## 3. Vectorstores

### 3.1 Initialize Chroma

Langchain integrates more than 30 different vector stores. We chose Chroma because it is lightweight and the data is stored in memory, which makes it very easy to start and start using.

First, we specify a persistence path:

```python
from langchain.vectorstores import Chroma

persist_directory_chinese = 'docs/chroma/matplotlib/'
```

If there are old database files in this path, you can delete them with the following command:

```python
!rm -rf './docs/chroma/matplotlib' # Delete the old database file (if there is one in the folder)
```

Next, create a vector database from the loaded document:

```python
vectordb_chinese = Chroma.from_documents(
documents=splits,
embedding=embedding,
persist_directory=persist_directory_chinese # Allows us to save the persist_directory directory to disk
)
```

100%|██████████| 1/1 [00:01<00:00, 1.64s/it]

You can see that the database length is also 30, which is the same as the number of splits we had before. Now let's start using it.

```python
print(vectordb_chinese._collection.count())
```

27

### 3.2 Similarity Search

First, we define a question to be answered:

```python
question_chinese = "What is Matplotlib?" 
```

Then call the loaded vector database to retrieve answers based on similarity:

```python
docs_chinese = vectordb_chinese.similarity_search(question_chinese,k=3)
```

Check the number of retrieved answers:

```python
len(docs_chinese)
```

3

Print its page_content attribute to see the text of the retrieved answer:

```python
print(docs_chinese[0].page_content)
```

First: Matplotlib introduction
1. Understanding matplotlib
Matplotlib is a Python 2D drawing library that can generate publication-quality graphics in multiple hardcopy formats and cross-platform interactive environments, and is used to draw various static, dynamic,
interactive charts.
Matplotlib can be used in Python scripts, Python and IPython Shell, Jupyter notebook, WebApplication servers and various graphical user interface toolkits, etc.
Matplotlib is the Thai in Python data visualization library. It has become a recognized data visualization tool in Python. The drawing interfaces of pandas and seaborn that we are familiar with
are actually high-level encapsulations based on matplotlib.
In order to have a better understanding of matplotlib, let us start with some basic concepts and then gradually transition to some advanced techniques.
2. The simplest drawing example
Matplotlib images are drawn on figures (such as windows, jupyter windows), and each figure contains one or more axes (a sub-area that can specify a coordinate system
). The simplest way to create a figure and axes is through the pyplot.subplots command. After creating axes, you can use Axes.plot to draw the simplest line chart.
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
fig,ax = plt.subplots() # Create a figure with an axes
ax.plot([1, 2, 3, 4], [1, 4, 2, 3]); # Draw an image
Trick: When using matplotlib in a jupyter notebook, you will find that after the code is run, it will automatically print out a sentence like <matplotlib.lines.Line2D at 0x23155916dc0>
This is because the drawing code of matplotlib prints out the last object by default. If you don't want to display this sentence, there are three ways to do it. You can find the use of these three methods in the code examples in this chapter
.
. Add a semicolon at the end of the code block;
. Add a sentence plt.show() at the end of the code block
. When drawing, explicitly assign the drawing object to a variable, such as changing plt.plot([1, 2, 3, 4]) to line =plt.plot([1, 2, 3, 4])
Similar to MATLAB commands, you can also draw images in a simpler way. The matplotlib.pyplot method can draw images directly on the current axes, such asIf the user
does not specify axes, matplotlib will automatically create one for you. So the above example can also be simplified to the following line of code.
line =plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
3. The composition of Figure
Now let's take a closer look at the composition of figure. Through a figure anatomy diagram, we can see that a complete matplotlib image usually includes the following four levels, which are also called containers (containers), which will be introduced in detail in the next section. In the world of matplotlib, we will use various command methods to manipulate each part of the image,
so as to achieve the final effect of data visualization. A complete image is actually a collection of various sub-elements.
Figure: Top level, used to hold all drawing elements
After this, we must ensure that the vector database is persisted by running vectordb.persist so that we can use it in future courses.
```python
vectordb_chinese.persist()
```

## 4. Failure modes

This looks good, and basic similarity search is easy to do.will get you 80% of the way there. However, there may be some cases where the similarity search fails. Here are some possible edge cases - we will fix them in the next section.

### 4.1 Duplicate Chunks

```python
question_chinese = "What is Matplotlib?"

docs_chinese = vectordb_chinese.similarity_search(question_chinese,k=5)

```

Note that we get duplicate chunks (because there are duplicate `第一回：Matplotlib初相识.pdf` in the index).

Semantic search gets all similar documents, but does not enforce diversity.

`docs[0]` and `docs[1]` are exactly the same.

```python
print("docs[0]")
print(docs_chinese[0])

print("docs[1]")
print(docs_chinese[1])
```

docs[0]
page_content='First: Introduction to Matplotlib\n1. Introduction to Matplotlib\nMatplotlib is a Python 2D drawingA library that can generate publication-quality graphics in multiple hardcopy formats and a cross-platform interactive environment for drawing various static, dynamic, and interactive charts. Matplotlib can be used in Python scripts, Python and IPython Shell, Jupyter notebooks, web application servers, and various graphical user interface toolkits. Matplotlib is the Thai of Python data visualization libraries. It has become a recognized data visualization tool in Python. The drawing interfaces of pandas and seaborn that we are familiar with are actually advanced encapsulations based on matplotlib. In order to have a better understanding of matplotlib, let's start with some of the most basic concepts and then gradually transition to some advanced techniques. \nSecond, a simplest drawing example\nMatplotlib images are drawn on figures (such as windows, jupyter windows), and each figure contains one or more axes (a subregion that can specify a coordinate system). The simplest way to create a figure and axes is through the pyplot.subplots command. After creating axes, you can use Axes.plot draws the simplest line chart. \nimport matplotlib.pyplot as plt\nimport matplotlib as mpl\nimport numpy as np\nfig, ax = plt.subplots() # Create a figure with an axes\nax.plot([1, 2, 3, 4], [1, 4, 2, 3]); # Draw an image\nTrick: When using matplotlib in jupyter notebook, you will find that after the code is run, it will automatically print out a sentence like <matplotlib.lines.Line2D at 0x23155916dc0>\nThis is because the drawing code of matplotlib prints out the last object by default. If you don't want to display this sentence, there are three methods below. You can find the use of these three methods in the code examples in this chapter\n. \n\x00. Add a semicolon at the end of the code block;\n\x00. Add plt.show() at the end of the code block\n\x00. When drawing, explicitly assign the drawing object to a variable, such as changing plt.plot([1, 2, 3, 4]) to line =plt.plot([1, 2, 3, 4])\nSimilar to MATLAB commands, you can also draw images in a simpler way. The matplotlib.pyplot method can draw images directly on the current axes. If the user\ndoes not specify axes, matplotlib will automatically create one for you. So the above example can also be simplified to the following line of code. \nline =plt.plot([1, 2, 3, 4], [1, 4, 2, 3]) \n3. Composition of Figure\nNow let's take a closer look at the composition of figure. Through a figure anatomy diagram, we can see that a complete matplotlib image usually includes the following four levels, which are also called containers (containers), which will be introduced in detail in the next section. In the world of matplotlib, we will use various command methods to manipulate each part of the image, thereby achieving the final effect of data visualization. A complete image is actually a collection of various sub-elements. Figure: The top level, used to accommodate all drawing elements' metadata={'source': 'docs/matplotlib/第一回：Matplotlib初相见.pdf', 'page': 0}
docs[1]
page_content='Episode 1: Getting to know Matplotlib\n1. Understanding matplotlib\nMatplotlib is a Python 2D drawing library that can generate publication-quality graphics in multiple hardcopy formats and cross-platform interactive environments, and is used to draw various static, dynamic, and interactive charts. \nMatplotlib can be used in Python scripts, Python and IPython Shell, Jupyter notebook, Web application servers, and various graphical user interface toolkits. \nMatplotlib is the Thai in Python data visualization libraries. It has become a recognized data visualization tool in Python. The drawing interfaces of pandas and seaborn that we are familiar with\nare actually advanced encapsulations based on matplotlib. \nIn order to have a better understanding of matplotlib, let's start with some of the most basic concepts to understand it, and then gradually transition to some advanced techniques. \nSecond, a simplest drawing example\nMatplotlib images are drawn on figures (such as windows, jupyter windows), and each figure contains one or more axes (asubplots() # Create a figure with one axes\nax.plot([1, 2, 3, 4], [1, 4, 2, 3]); # Draw an image\nTrick: When using matplotlib in jupyter notebook, you will find that after running the code, a paragraph like <matplotlib.lines.Line2D at 0x23155916dc0> is automatically printed out. This is because the drawing code of matplotlib prints out the last object by default. If you do not want to display this sentence, there are three ways to do it. You can find the use of these three methods in the code examples in this chapter. \n\x00. Add a semicolon at the end of the code block;\n\x00. Add a sentence at the end of the code blockplt.show()\n\x00. When drawing, explicitly assign the drawing object to a variable, such as changing plt.plot([1, 2, 3, 4]) to line =plt.plot([1, 2, 3, 4])\nSimilar to MATLAB commands, you can also draw images in a simpler way. The matplotlib.pyplot method can draw images directly on the current axes. If the user\ndoes not specify axes, matplotlib will automatically create one for you. So the above example can also be simplified to the following line of code. \nline =plt.plot([1, 2, 3, 4], [1, 4, 2, 3]) \n3. The composition of Figure\nNow let's take a closer look at the composition of figure. Through a figure anatomy diagram, we can see that a complete matplotlib image usually includes the following four levels, which are also called containers. The next section will introduce them in detail. In the world of matplotlib, we will use various command methods to manipulate each part of the image to achieve the final effect of data visualization. A complete image is actually a collection of various sub-elements. Figure: The top level, used to hold all drawing elements'metadata={'source': 'docs/matplotlib/第第：Matplotlib初见.pdf', 'page': 0}

### 4.2 Retrieving wrong answers

We can see a new failure situation.

The following questions ask questions about the second lecture, but also include results from other lectures.

```python
question_chinese = "What did they say about Figure in the second lecture?" 
docs_chinese = vectordb_chinese.similarity_search(question_chinese,k=5)

for doc_chinese in docs_chinese:
print(doc_chinese.metadata)

```

{'source': 'docs/matplotlib/第一回：Matplotlib初相识.pdf', 'page': 0}
{'source': 'docs/matplotlib/第一回：Matplotlib初相识.pdf', 'page': 0}
{'source': 'docs/matplotlib/第一回：Matplotlib初相识.pdf', 'page': 0}atplotlib/Second Chapter: Art Brush Sees the Universe.pdf', 'page': 9}
{'source': 'docs/matplotlib/Second Chapter: Art Brush Sees the Universe.pdf', 'page': 10}
{'source': 'docs/matplotlib/First Chapter: Matplotlib First Meeting.pdf', 'page': 1}

It can be seen that although the question we asked is about the second lecture, the first answer that appears is the content of the first lecture. The third answer is the correct answer we want.

```python
print(docs_chinese[2].page_content)
```

3. Object container - Object container
The container will contain some primitives, and the container also has its own attributes.
For example, Axes Artist is a container that contains many primitives, such as Line2D and Text. At the same time, it also has its own properties, such as xscal, which is used to control whether the X-axis is linear or log. 1. Figure container matplotlib.figure.Figure is an ArtistThe top-level container object container contains all the elements in the chart. The background of a chart is a rectangle in Figure.patch. When we add Figure.add_subplot() or Figure.add_axes() elements to the chart, these will be added to the Figure.axes list. fig = plt.figure()
ax1 = fig.add_subplot(211) # Make a 2*1 figure, select the first subplot
ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.3]) # Position parameter, the four numbers represent
(left, bottom, width, height)
print(ax1) 
print(fig.axes) # fig.axes contains two instances of subplot and axes, just added
AxesSubplot(0.125,0.536818;0.775x0.343182)
[<AxesSubplot:>, <Axes:>]
Because Figure maintains current axes, so you should not manually add or remove elements from the Figure.axes list, but use Figure.add_subplot(),
Figure.add_axes() to add elements, and Figure.delaxes() to remove elements. But you can iterate or access the Axes in Figure.axes, and then modify the properties of this
Axes.
For example, the following traverses the contents of the axes and adds grid lines:
fig = plt.figure()
ax1 = fig.add_subplot(211)
for ax in fig.axes:
ax.grid(True)
Figure also has its own text, line, patch, image. You can add them directly through the add primitive statement. But note that the default coordinate system of Figure is in pixels, you may need to convert to the figure coordinate system: (0,0) represents the lower left point, (1,1) represents the upper right point.
Common properties of Figure container: 
Figure.patch property: background matrix of FigureFigure.axes attribute: a list of Axes instances (including Subplot) Figure.images attribute: a list of FigureImages patches Figure.lines attribute: a list of Line2D instances (rarely used) Figure.legends attribute: a list of Figure Legend instances (different from Axes.legends) Figure.texts attribute: a list of Figure Text instances In the following chapters, we will explore methods that can effectively answer these two questions! ## Five, English version ** 1.1 Reading documents ** ``` python from langchain.document_loaders import PyPDFLoader # Load PDF loaders = [ # Deliberately add duplicate documents to confuse the data PyPDFLoader ("docs/cs229_lectures/MachineLearning-Lecture01.pdf"), PyPDFLoader ("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
docs.extend(loader.load())
```

Perform segmentation

```python
# Split text
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
chunk_size = 1500, # The size of each text block. This means that each time the text is split, each block will be tried to contain 1500 characters.
chunk_overlap = 150 # The overlap between each text block.
)
splits = text_splitter.split_documents(docs)

print(len(splits))
```

209

**2.1 Embedding**

```python
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np

embedding = OpenAIEmbeddings()

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

print("Sentence 1 VS sentence 2")
print(np.dot(embeddingedding1, embedding2))
print("Sentence 1 VS setence 3")
print(np.dot(embedding1, embedding3))
print("Sentence 2 VS sentence 3")
print(np.dot(embedding2, embedding3))
```

Sentence 1 VS setence 2
0.9632026347895142
Sentence 1 VS setence 3
0.7711302839662464
Sentence 2 VS sentence 3
0.759699788340627

**3.1 Initialize Chroma**

```python
from langchain.vectorstores import Chroma

persist_directory = 'docs/chroma/cs229_lectures/'

vectordb = Chroma.from_documents(
documents=splits,
embedding=embedding,
persist_directory=persist_directory # Allows us to save the persist_directory directory to disk
)
print(vectordb._collection.count())
```

100%|██████████| 1/1 [00:02<00:00, 2.62s/it]

209

**3.2 Similarity search**

```python
question = "is there an email i can ask for help" # "Is there an email I can ask for help"

docs = vectordb.similarity_search(question,k=3)

print("Length of docs: ", len(docs))
print("Page content:")
print(docs[0].page_content)
```

Length of docs: 3
Page content:
cs229-qa@cs.stanford.edu. This goes to an account that's read by all the TAs and me. So 
rather than sending us email individually, if you send email to this account, it will 
actually let us get back to you maximally quickly with answers to your questions. 
If you're asking questions about homework problems, please say in the subject line which 
assignment and which question the email refers to, since that will also help us to route 
your question to the appropriate TA or to me appropriately and get the response back to 
you quickly. 
Let's see. Skipping ahead — let's see — for homework, one midterm, one open and term 
project. Notice on the honor code. So one thing that I think will help you to succeed and 
do well in this class and even help you to enjoy this class more is if you form a study 
group. 
So start looking around where you' re sitting now or at the end of class today, mingle a 
little bit and get to know your classmates.I strongly encourage you to form study groups 
and sort of have a group of people to study with and have a group of your fellow students 
to talk over these concepts with. You can also post on the class news group if you want to 
use that to try to form a study group. 
But some of the problems sets in this class are reasonably difficult. People that have 
taken the class before may tell you they were very difficult. And just I bet it would be 
more fun for you, and yYou'd probably have a better learning experience if you form a

Persistent Database

```python
vectordb.persist()
```

**4.1 Repeating Blocks**

```python
question = "what did they say about matlab?" # "what did they say about matlab?"

docs = vectordb.similarity_search(question,k=5)

print("docs[0]")
print(docs[0])

print("docs[1]")
print(docs[1])
```

docs[0]
page_content='those homeworks will be done in either MATLA B or in Octave, which is sort of — I \nknow some people call it a free ve rsion of MATLAB, which it sortof is, sort of isn\'t. \nSo I guess for those of you that haven\'t seen MATLAB before, and I know most of you \nhave, MATLAB is I guess part of the programming language that makes it very easy to write codes using matrices, to write code for numerical routines, to move data around, to \nplot data. And it\'s sort of an extremely easy to learn tool to use for implementing a lot of \nlearning algorithms. \nAnd in case some of you want to work on your own home computer or something if you \ndon\nnot have a MATLAB license, for the purposes of this class, there\'s also — [inaudible] \nwrite that down [inaudible] MATLAB — there\' s also a software package called Octave \nthat you can download for free off the Internet. And it has somewhat fewer features than MATLAB, but it\'s free, and for the purposes of this class, it will work for just about \neverything. \nSo actually I, well, so yeah, just a side comment for those of you that haven\'t seen \nMATLAB before I guess, once a colleagueof mine at a different university, not at \nStanford, actually teaches another machine l earning course. He\'s taught it for many years. \nSo one day, he was in his office, and an old student of his from, like e, ten years ago came \ninto his office and he said, "Oh, professor, professor, thank you so much for your' metadata={'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 8}
docs[1]
page_content='those homeworks will be done in either MATLA B or in Octave, which is sort of — I \nknow some people call it a free ve rsion of MATLAB, which it sort of is, sort of isn\'t. \nSo I guess for those of you that haven\'t seen MATLAB before, and I know most of you \nhave, MATLAB is I guess part of the programming language that makes it very easy to write codes using matrices, to write code for numerical routines, to move data around, to \nplot data. And it\'s sort of an extremely easy to learn tool to use for implementing a lot of \nlearning algorithms. \nAnd incase some of you want to work on your own home computer or something if you \ndon\'t have a MATLAB license, for the purposes of this class, there\'s also — [inaudible] \nwrite that down [inaudible] MATLAB — there\' s also a software package called Octave \nthat you can download for free off the Internet. And it has somewhat fewer features than MATLAB, but it\'s free, and for the purposes of this class, it will work for just about \neverything. \nSo actually I, well, so yeah, just a side comment for those of you that haven\'t seen \nMATLAB before I guess, once a colleague of mine at a different university, not at \nStanford, actually teaches another machine l earning course. He\'s taught it for many years. \nSo one day, he was in his office, and an old student of his from, like e, ten years ago came \ninto his office and he said, "Oh, professor, professor, thank you so much for your' metadata={'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 8}

**4.2 Retrieve wrong answers**

```python
question = "what did they say about regression in the third lecture?" # "What did they say about regression in the third lecture?"

docs = vectordb.similarity_search(question,k=5)

for doc in docs:
print(doc.metadata)

print("docs-4:")
print(docs[4].page_content)
```

{'source': 'docs/cs229_lectures/MachineLearning-Lecture03.pdf', 'page': 0}
{'source': 'docs/cs229_lectures/MachineLearning-Lecture03.pdf', 'page': 14}
{'source': 'docs/cs229_lectures/MachineLearning-Lecture02.pdf', 'page': 0}
{'source':'docs/cs229_lectures/MachineLearning-Lecture03.pdf', 'page': 6}
{'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 8}
docs-4:
into his office and he said, "Oh, professor, thank you so much for your 
machine learning class. I learned so much from it. There's this stuff that I learned in your 
class, and I now use every day. And it's helped ed me make lots of money, and here's a 
picture of my big house." 
So my friend was very excited.He said, "W ow. That's great. I'm glad to hear this 
machine learning stuff was actually useful. So what was it that you learned? Was it 
logistic regression? Was it the PCA? Was it the data networks? What was it that you 
learned that was so helpful?" And the student said, "Oh, it was the MATLAB." 
So for those of you that don't know MATLAB yet, I hope you do learn it. It's not hard, 
and we'll actually have a short MATLAB tutori al in one of the discussion sections forthose of you that don't know it. 
Okay. The very last piece of logistical th ing is the discussion s ections. So discussion 
sections will be taught by the TAs, and attendance at discussion sections is optional, 
although they'll also be recorded and televised. And we'll use the discussion sections 
mainly for two things. For the next two or three weeks, we'll use the discussion sections 
to go over the prerequisites to this class or if some of you haven't seen probability or 
statistics for a while or maybe algebra, we'll go over those in the discussion sections as a 
refresher for those of you that want one.