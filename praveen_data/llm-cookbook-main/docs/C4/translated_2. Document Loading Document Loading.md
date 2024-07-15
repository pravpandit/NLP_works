# Chapter 2 Document Loading

User personal data can be presented in many forms: PDF documents, videos, web pages, etc. Based on the ability of LangChain to provide LLM with access to user personal data, the first thing to do is to load and process the user's diverse, unstructured personal data. In this chapter, we first introduce how to load documents (including documents, videos, web pages, etc.), which is the first step to access personal data.

Let's start with PDF documents.

## 1. PDF Documents

First, we will load a [PDF document](https://datawhalechina.github.io/fantastic-matplotlib/) from the following link. This is an open source tutorial provided by DataWhale, called "Fantastic Matplotlib". It is worth noting that in the English version, Professor Andrew Ng used his [2009 Machine Learning Course Subtitle File](https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture01.pdf) as an example. In order to be more suitable for Chinese readers, we chose the above Chinese tutorial as an example. However, you can still find Professor Andrew Ng’s machine learning course files as a reference in the original English code.. Subsequent code practices will also follow this adjustment.

Note that to run the following code, you need to install the third-party library pypdf:

```python
!pip install -q pypdf
```

### 1.1 Load PDF document

First, we will use `PyPDFLoader` to read and load PDF files.

```python
from langchain.document_loaders import PyPDFLoader

# Create a PyPDFLoader Class instance and input the path of the PDF document to be loaded
loader = PyPDFLoader("docs/matplotlib/第一回：Matplotlib初相识.pdf")

# Call the function load of PyPDFLoader Class to load the PDF file
pages = loader.load()
```

### 1.2 Explore the loaded data

Once the document is loaded, it will be stored in a variable named `pages`. In addition, the data structure of `pages` is a `List` type. To confirm its type, we can use Python's built-in `type` function to view the exact number of `pages`According to the type.

```python
print(type(pages))
```

<class 'list'>

By outputting the length of `pages`, we can easily understand the total number of pages contained in the PDF file.

```python
print(len(pages))
```

3

In the `page` variable, each element represents a document, and their data type is `langchain.schema.Document`.

```python
page = pages[0]
print(type(page))
```

<class 'langchain.schema.document.Document'>

The `langchain.schema.Document` type contains two attributes:

1. `page_content`: Contains the content of the document page.

```python
print(page.page_content[0:500])
```

First episode: Getting to know Matplotlib
1. Getting to know matplotlib
Matplotlib is a Python 2D drawing library, which can generate publication-quality graphics in multiple hardcopy formats and cross-platform interactive environments, and is used to draw various static, dynamic, and interactive charts. Matplotlib can be used in Python scripts, Python and IPython Shell, Jupyter notebook, Web application servers, and various graphical user interface toolkits. Matplotlib is the Thai in Python data visualization library. It has become a recognized data visualization tool in Python. The drawing interfaces of pandas and seaborn that we are familiar with are actually advanced encapsulations based on matplotlib. In order to have a better understanding of matplotlib, let us start with some of the most basic concepts and then gradually transition to some advanced techniques. 2. The simplest drawing example
Matplotlib images are drawn on figures (such as windows, jupyter windows), and each figure contains one or more axes (a sub-area that can specify a coordinate system). The simplest way to create a figure

2. `meta_data`: Descriptive data related to the document pageAccording to.

```python
print(page.metadata)
```

{'source': 'docs/matplotlib/第一回：Matplotlib初相识.pdf', 'page': 0}

## 2. YouTube audio

In the first part, we have discussed how to load PDF documents. In this part, for a given YouTube video link, we will discuss in detail:

- Use the `langchain` loading tool to download the corresponding audio for the specified YouTube video link to the local

- Use the `OpenAIWhisperPaser` tool to convert these audio files into readable text content

Note that to run the following code, you need to install the following two third-party libraries:

```python
!pip -q install yt_dlp
!pip -q install pydub
```

### 2.1 Loading Youtube audio documents

First, we will build a `GenericLoader` instance to download and load Youtube videos locally.

```python
from langchain.document_loaders.genericc import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url="https://www.youtube.com/watch?v=_PHdzsQaDgw"
save_dir="docs/youtube-zh/"

# Create a GenericLoader Class instance
loader = GenericLoader(
# Download the audio of the Youtube video in the link url and save it in the local path save_dir
YoutubeAudioLoader([url],save_dir), 

# Use OpenAIWhisperPaser parser to convert audio to text
OpenAIWhisperParser()
)

# Call the GenericLoader Class function loadLoad the audio file of the video
pages = loader.load()
```

[youtube] Extracting URL: https://www.youtube.com/watch?v=_PHdzsQaDgw
[youtube] _PHdzsQaDgw: Downloading webpage
[youtube] _PHdzsQaDgw: Downloading ios player API JSON
[youtube] _PHdzsQaDgw: Downloading android player API JSON
[youtube] _PHdzsQaDgw: Downloading m3u8 information

WARNING: [youtube] Failed to download m3u8 information: HTTP Error 429: Too Many Requests

[info] _PHdzsQaDgw: Downloading 1 format(s): 140
[download] docs/youtube-zh//【2023年7月最新】ChatGPT Registration Tutorial, Detailed Registration Process in China, Support Chinese Use, How to Use ChatGPT in China? .m4a has already been downloaded
[download] 100% of 7.72MiB
[ExtractAudio] Not converting audio docs/youtube-zh//【2023年7月最新】ChatGPT Registration Tutorial, Detailed Registration Process in China, Support Chinese Use, How to Use ChatGPT in China? .m4a; file is already in target format m4a
Transcribing part 1!

### 2.2 Explore the loaded data

The variables obtained by loading the Youtube audio file are similar to the above, so we will not explain them one by one here. The loaded data can be displayed through similar code:

```python
print("Type of pages: ", type(pages))
print("Length of pages: ", len(pages))

page = pages[0]
print("Type of page: ", type(page))
print("Page_content: ", page.page_content[:500])
print("Meta Data: ", page.metadata)
```

Type of pages: <class 'list'>
Length of pages: 1
Type of page: <class 'langchain.schema.document.Document'>
Page_content: Hello everyone, welcome to my channel. Today we will introduce how to register a ChetGBT account. I have introduced how to register a ChetGBT account before, but some friends still encounter some problems during the registration process. Today we will introduce the latest registration method in detail. Let's open this website first. I will put the URL of this website in the comment area below the video. You can click to open it directly. This website needs to be opened by climbing over the wall. It is recommended to use the global mode to climb over the wall. You can choose Taiwan, Singapore, Japan, and the United States nodes. Don't choose the Hong Kong node. I use the Taiwan node here. If you need this climbing over the wall software, I will also share it at the bottom of the video. In additionThe browser needs to be opened in Incognito mode. This is to open a new Incognito mode window. We can press the shortcut key, Ctrl+Shift+N to open a new Incognito mode window. Then use the Incognito mode window to open this website. Then click here. Then the login registration interface will appear. If this interface is not displayed, it shows access denied, which means that the node you are using may have a problem. We need to switch to other nodes. We can switch to other nodes in this way. Being able to open this page normally means that the node is fine. We can click to register. Here you need to fill in an email address and click Continue. Then you need to enter a password and click Continue. Then this prompt will appear. We need to receive an email. Refresh. The email has been received. Meta Data: {'source': 'docs/youtube-zh/【Latest in July 2023】ChatGPT registration tutorial, detailed registration process in China, support for Chinese use, how to use chatgpt in China? .m4a', 'chunk': 0}

## 3. Web Document

In the second part, for a given YouTube video link (URL), we use the LangChain loader to download the audio of the video to the local computer, and then use the OpenAIWhisperPaser parser to convert the audio into text.

In this part, we willStudy how to handle web links (URLs). To do this, we will take a markdown format document on GitHub as an example to learn how to load it.

### 3.1 Loading web documents

First, we will build a `WebBaseLoader` instance to load the web page.

```python
from langchain.document_loaders import WebBaseLoader

# Create a WebBaseLoader Class instance
url = "https://github.com/datawhalechina/d2l-ai-solutions-manual/blob/master/docs/README.md"
header = {'User-Agent': 'python-requests/2.27.1', 
'Accept-Encoding': 'gzip, deflate, br', 
'Accept': '*/*',
'Connection': 'keep-alive'}
loader = WebBaseLoader(web_path=url,header_template=header)

# Call the function load of WebBaseLoader Class to load the file
pages = loader.load()
```

### 3.2 Explore the loaded data

Similarly, we can display the loaded data through the above code:

```python
print("Type of pages: ", type(pages))
print("Length of pages: ", len(pages))

page = pages[0]
print("Type of page: ", type(page))
print("Page_content: ", page.page_content[:500])
print("Meta Data: ", page.metadata)
```

Type of pages: <class 'list'>
Length of pages: 1
Type of page: <class 'langchain.schema.document.Document'>
Page_content: {"payload":{"allShortcutsEnabled":false,"fileTree":{"docs":{"items":[{"name":"ch02","path":"docs/ch02","contentType":"directory"},{"name":"ch03","path":"docs/ch03","contentType":"directory"},{"name":"ch05","path":"docs/ch05","contentType":"directory"},{"name":"ch06","path":"docs/ch06","contentType":"directory"},{"name":"ch08","path":"docs/ch08","contentType":"directory"},{"name":"ch09","path":"docs/ch09","contentType":"directory"},{"name":"ch10","path":"docs/ch10","contentType":"directory"}rectory"},{"na
Meta Data: {'source': 'https://github.com/datawhalechina/d2l-ai-solutions-manual/blob/master/docs/README.md'}

You can see that the above document content contains a lot of redundant information. Generally speaking, we need to further process this data (Post Processing).

```python
import json
convert_to_json = json.loads(page.page_content)
extracted_markdow = convert_to_json['payload']['blob']['richText']
print(extracted_markdow)
```

Hands-on Deep Learning Exercises {docsify-ignore-all}
  Teacher Li Mu's "Hands-on Deep Learning" is a classic book for introductory deep learning. This book introduces deep learning based on the deep learning framework. The code in the book can achieve "what you learn is what you use". For general beginners, it is still difficult to independently solve the after-class exercises in the book.. This project answers the exercises of "Hands-on Deep Learning" as an exercise manual for the book to help beginners quickly understand the content of the book.
Instructions
  Hands-on Deep Learning Exercises Answers mainly complete all the exercises in the book, and provide code and screenshots after running. The content inside is based on the content of deep learning as the prerequisite knowledge. The best way to use this exercise answer is to follow Teacher Li Mu's "Hands-on Deep Learning" as the main line and try to complete the after-class exercises. If you encounter something you don't know, then check the exercise answers.
  If you feel that the answer is not detailed, you can click here to submit the derivation or exercise number you want to add, and we will add it as soon as we see it.
The version of "Hands-on Deep Learning" selected

Book title: Hands-on Deep Learning (PyTorch Edition)
Authors: Aston Zhang, [US] Zachary C. Lipton, Li Mu, [Germany] Alexander J. Smola
Translator: He Xiaoting, Ruichaoer Hu
Publisher: People's Posts and Telecommunications Press
Edition: 1st edition in February 2023

Project structure
codes---------------------------------------------- Exercise code
docs-----------------------------------------------Exercise answers
notebook-------------------------------------------Exercise answers in JupyterNotebook format
requirements.txt-----------------------------------Running environment dependency packages

Follow us

Scan the QR code below to follow the official account: Datawhale

  Datawhale, a learning community focused on the field of AI. The original intention is for the learner, and grow with the learners. At present, there are thousands of people joining the learning community, organizing content learning in multiple fields such as machine learning, deep learning, data analysis, data mining, crawlers, programming, statistics, Mysql, data competitions, etc. You can join us by searching the official account Datawhale on WeChat.
LICENSE
This work is licensed under the Creative Commons Attribution-Non-Commercial-Share Alike 4.0 International License.

## 4. Notion Document

- Click [Notion Sample Document](https://yolospace.notion.site/Blendle-s-Employee-Handbook-e31bff7da17346ee99f531087d8b133f)(https://yolospace.notion.site/Blendle-s-Employee-Handbook-e31bff7da17346ee99f531087d8b133f) Click the Duplicate button in the upper right corner to copy the document to your Notion space
- Click the `⋯` button in the upper right corner and select Export as Mardown&CSV. The exported file will be a zip folder
- Unzip and save the mardown document to the local path `docs/Notion_DB/`

### 4.1 Load Notion Markdown document

First, we will use `NotionDirectoryLoader` to load the Notion Markdown document.

```python
from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
pages = loader.load()
```### 4.2 Explore the loaded data

Similarly, using the above code:

```python
print("Type of pages: ", type(pages))
print("Length of pages: ", len(pages))

page = pages[0]
print("Type of page: ", type(page))
print("Page_content: ", page.page_content[:500])
print("Meta Data: ", page.metadata)
```

Type of pages: <class 'list'>
Length of pages: 51
Type of page: <class 'langchain.schema.document.Document'>
Page_content: # #letstalkaboutstress

Let’s talk about stress. Too much stress. 

We know thiscan be a topic.

So let’s get this conversation going. 

[Intro: two things you should know](#letstalkaboutstress%2064040a0733074994976118bbe0acc7fb/Intro%20two%20things%20you%20should%20know%20b5fd0c5393a9498b93396e79fe71e8bf.md)

[What is stress](#letstalkaboutstress%2064040a0733074994976118bbe0acc7fb/What%20is%20stress%20b198b685ed6a474ab14f6fafff7004b6.md)

[When is there too much stress?](#letstalkaboutstress%2
Meta Data: {'source': 'docs/Notion_DB/#letstalkaboutstress 64040a0733074994976118bbe0acc7fb.md'}

## 5. English version

**1.1 Load PDF document**

```python
from langchain.document_loaders import PyPDFLoader

# Create a PyPDFLoader Class instance, input the path of the PDF document to be loaded
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")

# Call the load function of PyPDFLoader Class to load the PDF file
pages = loader.load()
```

**1.2 Explore the loaded data**

```python
print("Type of pages: ", type(pages))
print("Length of pages: ", len(pages))

page = pages[0]
print("Type of page: ", type(page))
print("Page_content: ", page.page_content[:500])
print("Meta Data: ", page.metadata)
```

Type of pages: <class 'list'>
Length of pages: 22
Type of page: <class 'langchain.schema.document.Document'>
Page_content: MachineLearning-Lecture01 
Instructor (Andrew Ng): Okay. Good morning. Welcome to CS229, the machine 
learning class. So what I wanna do today is just spend a little time going over the logistics 
of the class, and then we'll start to talk a bit about machine learning. 
By way of introduction, my name's Andrew Ng and I'll be the instructor for this class. And so 
I personally work in machine learning, and I've worked on it for about 15 years now, and 
I actually think that machine learning i
Meta Data: {'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 0}

**2.1 Loading Youtube audio**

```python
# Note: Since the video is long and prone to network problems, it is not run here. Readers can run it on their own to explore

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="docs/youtube/"

# Create a GenericLoader Class instance
loader = GenericLoader(
# Download the audio of the Youtube video in the link url and save it in the local path save_dir
YoutubeAudioLoader([url],save_dir), 

# Use OpenAIWhisperPaser parser to convert audio to text
OpenAIWhisperParser()
)

# Call the GenericLoader Class function load to load the audio file of the video
docs = loader.load()
```

**2.2 Exploring loading**

```python
print("Type of pages: ", type(pages))
print("Length of pages: ", len(pages))

page = pages[0]
print("Type of page: ", type(page))
print("Page_content: ", page.page_content[:500])
print("Meta Data: ", page.metadata)
```

**3.1 Loading web documents**

```python
from langchain.document_loaders import WebBaseLoader

# Create an instance of WebBaseLoader Class
url = "https://github.com/basecamp/handbook/blob/master/37signals-is-you.md"
header = {'User-Agent': 'python-requests/2.27.1', 
'Accept-Encoding': 'gzip, deflate, br', 
'Accept': '*/*',
'Connection': 'keep-alive'}
loader = WebBaseLoader(web_path=url,header_template=header)

# Call the function load of WebBaseLoader Class to load the file
pages = loader.load()
```

**3.2 Explore the loaded data**

```python
print("Type of pages: ", type(pages))
print("Length of pages: ", len(pages))

page = pages[0]
print("Type of page: ", type(page))
print("Page_content: ", page.page_content[:500])
print("Meta Data: ", page.metadata)
```

Type of pages: <class 'list'>
Length of pages: 1
Type of page: <class 'langchain.schema.document.Document'>
Page_content: {"payload":{"allShortcutsEnabled":false,"fileTree":{"":{"items":[{"name":"37signals-is-you.md","path":"37signals-is-you.md","contentType":"file"},{"name":"LICENSE.md","path":"LICENSE.md","contentType":"file"},{"name":"README.md","path":"README.md","contentType":"file"},{"name":"benefits-and-perks.md","path":"benefits-and-perks.md","contentType":"file"},{"name":"code-of-conduct.md","path":"code-of-conduct.md","contentType":"file"},{"name":"faq.md","path":"faq.md","contentType":"file"},{"name":"ge
Meta Data: {'source': 'https://github.com/basecamp/handbook/blob/master/37signals-is-you.md'}

For further processing

```python
import json
convert_to_json = json.loads(page.page_content)
extracted_markdown = convert_to_json['payload']['blob']['richText']
print(extracted_markdown)
```

37signals Is You
Everyone working at 37signals represents 37signals. When a customer gets a responsefrom Merissa on support, Merissa is 37signals. When a customer reads a tweet by Eron that our systems are down, Eron is 37signals. In those situations, all the other stuff we do to cultivate our best image is secondary. What’s right in front of someone in a time of need is what they’ll remember.
That’s what we mean when we say marketing is everyone’s responsibility, and that it pays to spend the time to recognize that. This means avoiding the bullshit of outage language and bending our policycies, not just lending your ears. It means taking the time to get the writing right and consider how you’d feel if you were on the other side of the interaction.
The vast majority of our customers come from word of mouth and much of that word comes from people in our audience. This is an audience we’ve been educating and entertaining for 20 years and counting, and your voice is part of us now, whether you like it or not! Tell us and our audience what you have to say!
This goes for toolsand techniques as much as it goes for prose. 37signals not only tries to out-teach the competition, but also out-share and out-collaborate. We’re prolific open source contributors through Ruby on Rails, Trix, Turbolinks, Stimulus, and many other projects. Extracting the common infrastructure that others could use as well is satisfying, important work, and we should continue to do that.
It’s also worth mentioning that joining 37signals can be all-consuming. We’ve seen it happen. You dig 37signals, so you feel pressure to contribute, maybe overwhelmingly so. The people who work here are some of the best and brightest in our industry, so the self-imposed burden to be exceptional is real. But here’s the thing: stop it. Settle in. We’re glad you love this job because we all do too, but at the end of the day it’s a job. Do your best work, collaborate with your team, write, read, learn, and then turn off your computer and play with your dog. We’ll all be better for it.

**4.1 Loading Notion Documentation**

```python
from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
pages = loader.load()
```

**4.2 Explore the loaded data**

```python
print("Type of pages: ", type(pages))
print("Length of pages: ", len(pages))

page = pages[0]
print("Type of page: ", type(page))
print("Page_content: ", page.page_content[:500])
print("Meta Data: ", page.metadata)
```

Type of pages: <class 'list'>
Length of pages: 51
Type of page: <class 'langchain.schema.document.Document'>
Page_content: # #letstalkaboutstress

Let’s talk about stress. Too much stress. 

We know this can be a topic.

So let’s get this conversation going. 

[Intro: two things you should know](#letstalkaboutstress%2064040a0733074994976118bbe0acc7fb/Intro%20two%20things%20you%20should%20know%20b5fd0c5393a9498b93396e79fe71e8bf.md)

[What is stress](#letstalkaboutstress%2064040a0733074994976118bbe0acc7fb/What%20is%20stress%20b198b685ed6a474ab14f6fafff7004b6.md)

[When is there too much stress?](#letstalkaboutstress%2
Meta Data: {'source': 'docs/Notion_DB/#letstalkaboutstress 64040a0733074994976118bbe0acc7fb.md'}