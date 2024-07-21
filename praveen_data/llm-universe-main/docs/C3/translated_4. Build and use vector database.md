# Build and use vector database

The corresponding source code of this article is [here](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93/4.%E6%90%AD%E5%BB%BA%E5%B9%B6%E4%BD%BF%E7%94%A8%E5%90%91%E9%87%8F%E6%95%B0%E6%8D%AE%E5%BA%93.ipynb). If you need to reproduce, you can download and run the source code.
## 1. Pre-configuration
This section focuses on building and using a vector database, so after reading the data, we skip the data processing step and go straight to the topic. For data cleaning and other steps, please refer to Section 3

```python
import os
from dotenv import load_dotenv, find_dotenv

# Read local/project environment variables.
# find_dotenv() finds and locates the path of the .env file
# load_dotenv() reads the .env file and loads the environment variables in it into the current running environment
# If you set global environment variables, this line of code has no effect.
_ = load_dotenv(find_dotenv())

# If you need to access through a proxy port, you need to configure it as follows
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

# Get all file paths under folder_path and store them in file_paths
file_paths = []
folder_path = '../../data_base/knowledge_db'
for root, dirs, files in os.walk(folder_path):
for file in files:
file_path = os.path.join(root, file)
file_paths.append(file_path)
print(file_paths[:3])
```

['../../data_base/knowledge_db/prompt_engineering/6. Text transformation Transforming.md', '../../data_base/knowledge_db/prompt_engineering/4. Text summarization Summarizing.md', '../../data_base/knowledge_db/prompt_engineering/5. Inferring Inferring.md']

```python
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader

# Traverse the file path and store the instantiated loader in loaders
loaders = []

for file_path in file_paths:

file_type = file_path.split('.')[-1]
if file_type == 'pdf':
loaders.append(PyMuPDFLoader(file_path))
elif file_type == 'md':
loaders.append(UnstructuredMarkdownLoader(file_path))
```

```python
# Download the file and store it in text
texts = []

for loader in loaders: texts.extend(loader.load())
```

The variable type after loading is `langchain_core.documents.base.Document`, and the document variable type also contains two attributes
- `page_content` contains the content of the document.
- `meta_data` is the descriptive data related to the document.

```python
text = texts[1]
print(f"The type of each element: {type(text)}.", 
f"Descriptive data of the document: {text.metadata}", 
f"View the content of the document:\n{text.page_content[0:]}", 
sep="\n------\n")
```The type of each element: <class 'langchain_core.documents.base.Document'>.
------
Descriptive data of the document: {'source': '../../data_base/knowledge_db/prompt_engineering/4. Text Summarizing.md'}
------
View the content of this document:
Chapter 4 Text Summarizing

In the busy information age, Xiao Ming is an enthusiastic developer who faces the challenge of processing a large amount of text information. He needs to find key information for his project by studying countless literature, but time is far from enough. When he was overwhelmed, he discovered the text summarization function of the Large Language Model (LLM).

This function is like a lighthouse for Xiao Ming, illuminating his way to process the ocean of information. The powerful ability of LLM is that it can simplify complex text information and extract key ideas, which is undoubtedly a great help to him. He no longer needs to spend a lot of time reading all the documents, just summarizing them with LLM, and he can quickly get the information he needs.

By programming and calling the API interface, Xiao Ming successfully implemented this text abstraction.He exclaimed: "It's like a magic that turns the endless ocean of information into a clear source of information." Xiao Ming's experience shows the great advantages of LLM text summarization: saving time, improving efficiency, and accurately obtaining information. This is what we will introduce in this chapter. Let's explore how to use programming and calling API interfaces to master this powerful tool.

1. Single text summary

Take the summary task of product reviews as an example: For e-commerce platforms, there are often a large number of product reviews on the website, which reflect the ideas of all customers. If we have a tool to summarize these massive and lengthy reviews, we can quickly browse more reviews and understand customer preferences, thereby guiding the platform and merchants to provide better services.

Next, we provide an online product review as an example, which may come from an online shopping platform, such as Amazon, Taobao, JD.com, etc. The reviewer commented on a panda doll, including factors such as the quality, size, price and logistics speed of the product, as well as his daughter's love for the product.

python
prod_review = """
This panda doll is a birthday gift for my daughter. She likes it very much and takes it with her wherever she goes.
The doll is very soft, super cute, and has a very kind facial expression. But compared to the price,It's a bit small, and I feel like I could get a bigger one for the same price elsewhere.
The courier arrived a day earlier than expected, so I played with it for a bit before giving it to my daughter.
"""

1.1 Limit the length of the output text

Let's first try to limit the length of the text to 30 words.

```python
from tool import get_completion

prompt = f"""
Your task is to generate a short summary of a product review from an e-commerce website.

Please summarize the review text between three backticks, up to 30 words.

Review: {prod_review}
"""

response = get_completion(prompt)
print(response)
```

We can see that the language model gives us a satisfactory result.

Note: In the previous section, we mentioned that the language model relies on the tokenizer when calculating and judging the length of the text, and the tokenizer does not have perfect accuracy in character statistics.

1.2 Set the key angle to focus onIn some cases, we may focus on different aspects of the text for different business scenarios. For example, in a product review text, the logistics department may focus more on the timeliness of transportation, the merchant may focus more on price and product quality, and the platform may focus more on the overall user experience.

We can emphasize our emphasis on a particular perspective by enhancing the input prompt.

1.2.1 Focus on express service

```python
prompt = f"""
Your task is to generate a short summary of a product review from an e-commerce website.

Please summarize the review text between three backticks, up to 30 words, and focus on express service.

Comment: {prod_review}
"""

response = get_completion(prompt)
print(response)
```

From the output, we can see that the text starts with "Express delivery arrives ahead of schedule", which reflects the emphasis on express efficiency.

1.2.2 Focus on price and quality

```python
prompt = f"""Your task is to generate a short summary of a product review from an e-commerce website.

Please summarize the review text between three backticks, up to 30 words, with a focus on product price and quality.

Comment: {prod_review}
"""

response = get_completion(prompt)
print(response)
```

From the output, we can see that the text begins with "Cute panda doll, good quality but a little small, slightly expensive", which reflects the emphasis on product price and quality.

1.3 Extracting key information

In Section 1.2, although we did make the text summary more focused on a specific aspect by adding a prompt that focuses on key angles, we can find that some other information is also retained in the results, such as the information that "express delivery arrived ahead of schedule" is still retained in the summary that focuses on price and quality. If we only want to extract information from a certain angle and filter out all other information, we can ask LLM to extract text instead of summarizing.

Let's extract information from the text together!```python
prompt = f"""
Your task is to extract relevant information from product reviews on e-commerce websites.

Please extract product shipping related information from the review text between the following three backticks, up to 30 words.

Review: {prod_review}
"""

response = get_completion(prompt)
print(response)
```

2. Summarize multiple texts at the same time

In actual workflows, we often have to deal with a large number of review texts. The following example collects multiple user reviews in a list, and uses a for loop and the text summary (Summarize) prompt word to summarize the reviews to less than 20 words and print them in order. Of course, in actual production, for review texts of different sizes, in addition to using a for loop, you may also need to consider integrating reviews, distributing and other methods to improve computing efficiency. You can build a main control panel to summarize a large number of user reviews, and facilitate quick browsing for you or others, and you can also click to view the original review. In this way, you can efficiently grasp all the ideas of customers.

```python
review_1= prod_review

Review of a Floor Lamp

review_2 = """
I needed a nice bedroom lamp with extra storage for a reasonable price.
The delivery was very fast, arriving in just two days.
However, the cord of the lamp had issues during shipping, but the company was happy to send a brand new one.
The new cord was also delivered very quickly, in just a few days.
Assembly was very easy. However, I later discovered that a part was missing, so I contacted customer service and they promptly sent me the missing part!
To me, this is a great company that really cares about its customers and its products.
"""

Review of an Electric Toothbrush

review_3 = """
My dental hygienist recommended an electric toothbrush, so I bought this one.
So far, the battery life has been pretty good.
After the initial charge, I left the charger plugged in for the first week to condition the battery.
I have been using it to brush my teeth every morning and night for the past 3 weeks and the battery still holds its charge.
However, the brush head is too small. I have seen baby toothbrushes with bigger heads than this one.
I wish the brush head was biggerSome, with different lengths of bristles,
would allow for better cleaning between teeth, but this toothbrush doesn't do that.
Overall, if you can get this toothbrush for around $50, it's a pretty good deal.
The manufacturer's replacement heads are pretty expensive, but you can buy generic heads for a much more reasonable price.
This toothbrush makes me feel like I go to the dentist every day, my teeth feel so clean!
"""

Review of a blender

review_4 = """
During November, this 17-piece set was on seasonal sale for about $49, which was about 50% off.
But for some reason (let's call it price hikes), by the second week of December, all the prices had gone up,
and the same set was between $70-89. The 11-piece set also went up about $10 from $29.
It looks pretty good, but if you look closely at the base, the part where the blades lock doesn't look as nice as in previous years.
However, I intend to use it very carefully
(e.g. I will grind hard foods like beans, ice, rice, etc. in the blender first before grinding them to the desired particle size,
then switch to the whisk blade to get a finer flour if I need to make a finer/ less pulpy foods).
When making smoothies, I cut the fruits and vegetables I’m going to use into fine pieces and freeze them
(If using spinach, I lightly cook it first and freeze it until ready to use.
If making sorbet, I use a small to medium food processor) so you can avoid adding too much ice.
About a year later, the motor started making strange noises. I called customer service, but the warranty had expired,
so I had to buy another one. It’s worth noting that the overall quality of this type of product has declined over the past few years,
so they rely to some extent on brand recognition and consumer loyalty to maintain sales. In about two days, I received my new blender.
"""

reviews = [review_1, review_2, review_3, review_4]

```

```python
for i in range(len(reviews)):
prompt = f"""
Your task is to extract relevant information from product reviews on an e-commerce website.

```

English version 3.1.1 Single textsummarypython prod_review = """ Got this panda plush toy for my daughter's birthday, \ who loves it and takes it everywhere. It's soft and \ super cute, and its face has a friendly look. It's \ a bit small for what I paid though. I think there \ might be other options that are bigger for the \ same price. It arrived a day earlier than expected, \ so I got to play with it myself before I gave it \ to her. """ ``` python prompt = f"""Your task is to generate a short summary of a product \ review from an ecommerce site. Summarize the review below, delimited by triple backticks, in at most 30 words. Review: {prod_review} """ response = get_completion(prompt) print( response) ``` 1.2 Set key angles to focus on 1.2.1 Focus on express delivery services```python prompt = f""" Your task is to generate a short summary of a product \ review from an ecommerce site to give feedback to the \Shipping deparmtment. Summarize the review below, delimited by triple backticks, in at most 30 words, and focusing on any aspects \ that mention shipping and delivery of the product. Review: {prod_review} """ response = get_completion(prompt) print( response) ``` 1.2.2 Focus on price and quality```python prompt = f""" Your task is to generate a short summary of a product \ review from an ecommerce site to give feedback to the \pricing deparmtment, responsible for determining the \ price of the product. Summarize the review below, delimited by triple backticks, in at most 30 words, and focusing on any aspects \ that are relevant to the price and perceived value. Review: {prod_review} """ response = get_completion(prompt) print(response) ``` 1.3 Key information extraction```python prompt = f""" Your task is to extract relevant information from \ a productreview from an ecommerce site to give \ feedback to the Shipping department. From the review below, delimited by triple quotes \ extract the information relevant to shipping and \ delivery. Limit to 30 words. Review: {prod_review} """ response = get_completion (prompt) print(response) ``` 2.1 Summarize multiple pieces of text at the same time ```python review_1 = prod_review review for a standing lamp review_2 = """ Needed a nice lamp for my bedroom, and this one \ had additional storage and not too high of a price \ point. Got it fast - arrived in 2 days. The string \ to the lamp broke during the transit and the company \ happily sent over a new one. Came within a few days \ as well. It was easy to put together. Then I had a \ missing part, so I contacted their support and they \ very quickly got me the missing piece! Seems to me \ to be a great company that cares about their customers \ and productscts. """ review for an electric toothbrush review_3 = """ My dental hygienist recommended an electric toothbrush, \ which is why I got this. The battery life seems to be \ pretty impressive so far. After initial charging and \ leaving the charger plugged in for the first week to condition the battery, I've unplugged the charger and been using it for twice daily brushing for the last 3 weeks all on the same charge. But the toothbrush headis too small. I've seen baby toothbrushes bigger than this one. I wish the head was bigger with different length bristles to get between teeth better because this one doesn't. Overall if you can get this one around the $50 mark, it's a good deal. The manufacturer's replacements heads are pretty expensive, but you can get generic ones that're more reasonably priced. This toothbrush makes me feel like I've been to the dentist every day. My teethfeel sparkly clean! """ review for a blender review_4 = """ So, they still had the 17 piece system on seasonal \ sale for around $49 in the month of November, about \ half off, but for some reason (call it price gouging) \ around the second week of December the prices all went \ up to about anywhere from between $70-$89 for the same \ system. And the 11 piece system went up around $10 or \ so in price also from the earlier sale price of $29.\So it looks okay, but if you look at the base, the part where the blade locks into place doesn't look as good as in previous editions from a few years ago, but I plan to be very gentle with it ( example, I crush very hard items like beans, ice, rice, etc. in the blender first then pulverize them in the serving size I want in the blender then switch to the whipping blade for a finer flour, and use the cross cutting blade \ first when making smoothies, then use the flat blade if I need them finer/less pulpy). Special tip when making smoothies, finely cut and freeze the fruits and vegetables (if using spinach-lightly stew soften the spinach then freeze until ready for use-and if making \ sorbet, use a small to medium sized food processor) \ that you plan to use that way you can avoid adding so \ much ice if at all-when making your smoothie. \ After about a year, the motor was making a funny noise. \I called customer service but the warranty expired already, so I had to buy another one. FYI: The overall quality has gone done in these types of products, so they are kind of counting on brand recognition and consumer loyalty to maintain sales. Got it in about \ two days. """ reviews = [review_1, review_2, review_3, review_4] ```` ```python for i in range(len(reviews)): prompt = f""" Your task is to generate a short summary of a product \
review from an ecommerce site.

```

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=500, chunk_overlap=50)

split_docs = text_splitter.split_documents(texts)
```

## 2. Build Chroma vector library

Langchain integrates more than 30 different vector repositories. We chose Chroma because it is lightweight and the data is stored in memory, which makes it very easy to start and start using.

LangChain can directly use the Embedding of OpenAI and Baidu Qianfan. At the same time, we can also customize it for the Embedding API that is not supported by them. For example, we can seal based on the interface provided by LangChain.Install a zhupuai_embedding to connect Zhipu's Embedding API to LangChain. In this chapter, [Attached LangChain Custom Embedding Package Explanation](./Attached LangChain Custom Embedding Package Explanation.ipynb), we take Zhipu Embedding API as an example to introduce how to encapsulate other Embedding APIs into LangChain. Interested readers are welcome to read.

**Note: If you use Zhipu API, you can refer to the explanation content to implement the encapsulation code, or you can directly use our encapsulated code [zhipuai_embedding.py](./zhipuai_embedding.py), download the code to the same level directory of this Notebook, and you can directly import our encapsulated function. In the following code Cell, we use Zhipu's Embedding by default, and present the other two Embedding usage codes in an annotated way. If you are using Baidu API or OpenAI API, you can use the code in the following Cell according to the situation. **

```python
# Use OpenAI Embedding
# from langchain.embeddings.openai import OpenAIEmbeddings
# Use Baidu Qianfan Embedding
# from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
# Use our own packaged Zhipu Embedding, you need to download the packaged code to your local machine for use
from zhipuai_embedding import ZhipuAIEmbeddings

# Define Embeddings
# embedding = OpenAIEmbeddings() 
embedding = ZhipuAIEmbeddings()
# embedding = QianfanEmbeddingsEndpoint()

# Define persistence path
persist_directory = '../../data_base/vector_db/chroma'
```

```python
!rm -rf '../../data_base/vector_db/chroma' # Delete the old databaseFiles (if there are files in the folder), please delete them manually on Windows computers
```

```python
from langchain.vectorstores.chroma import Chroma

vectordb = Chroma.from_documents(
documents=split_docs[:20], # For speed, only select the first 20 split docs for generation; when using Qianfan, due to QPS restrictions, it is recommended to select the first 5 docs
embedding=embedding,
persist_directory=persist_directory # Allows us to save the persist_directory directory to disk
)
```

After this, we want to make sure to persist the vector database by running vectordb.persist so that we can use it in future courses.

Let's save it for later use!

```python
vectordb.persist()
```

```python
print(f"Number of vectors stored in the library: {vectordb._collection.count()}")
```Number of vectors stored in the database: 20

## 3. Vector search
### 3.1 Similarity search
Chroma's similarity search uses cosine distance, that is:
$$
similarity = cos(A, B) = \frac{A \cdot B}{\parallel A \parallel \parallel B \parallel} = \frac{\sum_1^n a_i b_i}{\sqrt{\sum_1^n a_i^2}\sqrt{\sum_1^n b_i^2}}
$$
Where $a_i$ and $b_i$ are the components of vectors $A$ and $B$ respectively.

When you need the database to return results strictly sorted by cosine similarity, you can use the `similarity_search` function.

```python
question="What is a large language model"
```

```python
sim_docs = vectordb.similarity_search(question,k=3)
print(f"Number of retrieved contents: {len(sim_docs)}")
```

Number of retrieved contents: 3

```python
for i, sim_doc in enumerate(sim_docs):
print(f"The {i}th content retrieved: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
```

The 0th content retrieved: 
Chapter 6 Text Conversion

The large language model has powerful text conversion capabilities and can realize different types of text conversion tasks such as multilingual translation, spelling correction, grammar adjustment, format conversion, etc. Using language models for various conversions is one of its typical applications.

In this chapter, we will introduce how to use language models to implement text conversion functions by programming API interfaces. Through code examples, readers can learn the specific methods of converting input text into the required output format.

Mastering the skills of calling the large language model interface for text conversion is an important step in developing various language applications. Text
--------------
The 1st content retrieved: 
Taking English-Chinese translation as an example, traditional statistical machine translation tends to directly replace English vocabulary, and the word order maintains the English structure, which is prone to the phenomenon of unauthentic use of Chinese vocabulary and unsmooth word order. The large language model can learn the grammatical differences between English and Chinese and perform dynamic structural transformation. At the same time, it can also understand the original sentence intention through the context and selectAppropriate Chinese words are used for conversion, rather than rigid literal translation.

These advantages of large language model translation make the generated Chinese text more authentic, fluent, and accurate in meaning. Using large language model translation, we can break through the barriers between multiple languages.
--------------
The second content retrieved: 
Through this example, we can see that the large language model can smoothly handle multiple conversion requirements, realizing functions such as Chinese translation, spelling correction, tone upgrade and format conversion.

Using the powerful combination conversion ability of the large language model, we can avoid calling the model multiple times for different conversions, greatly simplifying the workflow. This method of achieving multiple conversions at one time can be widely used in text processing and conversion scenarios.

VI. English version

1.1 Translate to Spanish

python
prompt = f"""
Translate the fo
--------------

### 3.2 MMR retrieval
If only the relevance of the retrieved content is considered, the content will be too single and important information may be lost.

Maximum marginal relevance (`MMR, Maximum marginal relevance`) can help us maintainWhile increasing the relevance, it also increases the richness of the content.

The core idea is to select a document with low relevance to the selected document but rich information after selecting a highly relevant document. This can increase the diversity of the content while maintaining relevance and avoid overly single results.

```python
mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)
```

```python
for i, sim_doc in enumerate(mmr_docs):
print(f"MMR retrieved the {i}th content: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
```

MMR retrieved the 0th content: 
Chapter 6 Text Conversion

The large language model has powerful text conversion capabilities and can realize different types of text conversion tasks such as multilingual translation, spelling correction, grammar adjustment, format conversion, etc. Using language models for various conversions is one of its typical applications.

In this chapter, we will introduce how to use the language model to implement text conversion functions by calling the API interface through programming.You can learn how to convert input text into the desired output format.

Mastering the skills of calling the large language model interface for text conversion is an important step in developing various language applications. Text
--------------
MMR retrieved the first content: 
"This phrase is to cherck chatGPT for spelling abilitty" # spelling
]
--------------
MMR retrieved the second content: 
room.

room. Yes, adults also like pandas

too.

too. She takes it everywhere with her, and it's super soft and

cute. One

cute. However, one of the ears is a bit lower than the other, and I don't t
--------------

This article corresponds to the sourceThe code is here. If you need to reproduce, you can download and run the source code.