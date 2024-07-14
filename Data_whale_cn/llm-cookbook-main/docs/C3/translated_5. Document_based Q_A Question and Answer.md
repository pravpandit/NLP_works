# Chapter 5 Document-based Q&A

Using a large language model to build a Q&A system that can answer questions about given documents and document collections is a very practical and effective application scenario. **Unlike relying solely on model pre-training knowledge, this method can further integrate users' own data to achieve more personalized and professional Q&A services**. For example, we can collect text materials such as internal documents and product manuals of a company and import them into the Q&A system. Then when users ask questions about these documents, the system can first retrieve relevant information in the documents and then provide them to the language model to generate answers.

In this way, the language model not only uses its own general knowledge, but also makes full use of the professional information of external input documents to answer user questions, significantly improving the quality and applicability of the answers. Building this kind of external document-based Q&A system allows language models to better serve specific scenarios instead of staying at the general level. This flexible application of language models is worth promoting in actual use.

In this process of document-based Q&A, we will involve other components in LangChain, such as: Embedding Models and Vector Stores. Let's learn about this part together in this chapter.

## 1. Directly use vector storage query

### 1.1 Import data

```python
from langchain.chains importRetrievalQA #Retrieval QA chain, search on documents
from langchain.chat_models import ChatOpenAI #openai model
from langchain.document_loaders import CSVLoader #Document loader, stored in csv format
from langchain.vectorstores import DocArrayInMemorySearch #Vector storage
from IPython.display import display, Markdown #Tools for displaying information in jupyter
import pandas as pd

file = '../data/OutdoorClothingCatalog_1000.csv'

# Import data using langchain document loader
loader = CSVLoader(file_path=file)

# Import data using pandas for viewing
data = pd.read_csv(file,usecols=[1, 2])
data.head()
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
<td>This ultracomfortablele lace-to-toe Oxford boast...</td>
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

The data is text data with fields `name` and `description`:

As you can see, the imported dataset is a CSV file of outdoor clothing, which we will use in the language model next.

### 1.2 Basic document loader creates vector storage

```python
#Import vector storage index creator
from langchain.indexes import VectorstoreIndexCreator

# Create a specified vector storage class. After creation, call it from the loader and load it through the document loader list
index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])
```

### 1.3 Query created vector storage

```python
query ="Please list all shirts with sun protection in a markdown table and summarize the description of each shirt"

#Use the index query to create a response and pass it to the query
response = index.query(query)

#View the query return content
display(Markdown(response))
```

| Name | Description |
| --- | --- |
| Men's Tropical Plaid Short-Sleeve Shirt | UPF 50+ rated sun protection, 100% polyester fabric, wrinkle-resistant, front and back cape venting, two front bellows pockets |
| Men's Plaid Tropic Shirt, Short-Sleeve | UPF 50+ rated sun protection, 52% polyesterand 48% nylon fabric, wrinkle-free, quickly evaporates perspiration, front and back cape venting, two front bellows pockets |
| Girls' Ocean Breeze Long-Sleeve Stripe Shirt | UPF 50+ rated sun protection, Nylon Lycra®-elastane blend fabric, quick-drying and fade-resistant, holds shape well, durable seawater-resistant fabric retains its color |

In the above, we get a Markdown table containing the `Name` and `Description` of all shirts with sun protection clothing, where the description is the result summarized by the language model.

## 2. Combining representation model and vector storage

Due to the context length limit of the language model, it is difficult to directly process long documents. To achieve question answering of long documents, we can introduceTechnologies such as embeddings and vector stores are introduced:

First, the document is vectorized using the text embedding algorithm so that semantically similar text fragments have similar vector representations. Secondly, the vectorized document is divided into small pieces and stored in the vector database. This process is the process of creating an index. The vector database indexes each document fragment to support fast retrieval. In this way, when a user asks a question, the question can be converted into a vector first, and the most semantically relevant document fragment can be quickly found in the database. These document fragments are then passed to the language model together with the question to generate an answer.

Through embedding vectorization and indexing technology, we have achieved slice retrieval and question-answering of long documents. This process overcomes the contextual limitations of the language model and can build a question-answering system that handles large-scale documents.

### 2.1 Import data

```python
#Create a document loader to load data in csv format
file = '../data/OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
docs = loader.load()

#View individual documents, each corresponding to a row of data in CSV
docs[0]
```Document(page_content=": 0\nname: Women's Campside Oxfords\ndescription: This ultracomfortable lace-to-toe Oxford boasts a super-soft canvas, thick cushioning, and quality construction for a broken-in feel from the first time you put them on. \n\nSize & Fit: Order regular shoe size. For half sizes not offered, order up to next whole size. \n\nSpecs: Approx. weight: 1 lb.1 oz. per pair. \n\nConstruction: Soft canvas material for a broken-in feel and look. Comfortable EVA innersole with Cleansport NXT® antimicrobial odor control. Vintage hunt, fish and camping motif on innersole. Moderate arch contour of innersole. EVA foam midsole for cushioning and support. Chain-tread-inspired molded rubber outsole with modified chain-tread pattern. Imported. \n\nQuestions? Please contact us for any inquiries.", metadata={'source': '../data/OutdoorClothingCatalog_1000.csv', 'row': 0})

### 2.2 Text vector representation model

```python
#Use OpenAIEmbedding class
from langchain.embeddings import OpenAIEmbeddings 

embeddings = OpenAIEmbeddings() 

#Because the document is short, there is no need to perform any block division here, and vector representation can be performed directly
#Use the query method embed_query on the initialized OpenAIEmbedding instance to create a vector representation for the text
embed = embeddings.embed_query("Hello, my name is cute")

#Check the length of the vector representation
print("\n\033[32mThe length of the vector representation: \033[0m \n", len(embed))

#Each element is a different numeric value, and the combination is the vector representation of the text
print("\n\033[32mThe first 5 elements of the vector representation: \033[0m \n", embed[:5])
```

The length of the vector representation: 
1536

The first 5 elements of the vector representation: 
[-0.019283676849006164, -0.006842594710511029, -0.007344046732916966, -0.024501312942119265, -0.026608679897592472]

### 2.3 Vector-based representationCreate and query vector storage

```python
# Store the text vector representation (embeddings) just created in the vector storage (vector store)
# Use the from_documents method of the DocArrayInMemorySearch class to implement
# This method accepts a list of documents and a vector representation model as input
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

query = "Please recommend a shirt with sun protection"
# Use the above vector storage to find text similar to the incoming query and get a list of similar documents
docs = db.similarity_search(query)
print("\n\033[32mThe number of documents returned: \033[0m \n", len(docs))
print("\n\033[32mThe first document: \033[0m \n", docs[0])
```

The number of documents returned: 
4

The first document: 
page_content=": 535\nname: Men's TropicVibe Shirt,Short-Sleeve\ndescription: This Men’s sun-protection shirt with built-in UPF 50+ has the lightweight feel you want and the coverage you need when the air is hot and the UV rays are strong. Size & Fit: Traditional Fit: Relaxed through the chest, sleeve and waist. Fabric & Care: Shell: 71% Nylon, 29% Polyester. Lining: 100% Polyester knit mesh. UPF 50+ rated – the highest rated sun protection possible. Machine wash and dry. Additional Features: Wrinkle resistant. Front and back cape venting letsin cool breezes. Two front bellows pockets. Imported.\n\nSun Protection That Won't Wear Off: Our high-performance fabric provides SPF 50+ sun protection, blocking 98% of the sun's harmful rays." metadata={'source': '../data/OutdoorClothingCatalog_1000.csv', 'row': 535}

We can see that one returned four results. The first result output is a shirt about sun protection, which meets the requirements of our query: `Please recommend a shirt with sun protection function`

### 2.4 Use query results to construct prompts to answer questions

```python
#Import large language model, here the default model gpt-3.5-turbo will appear 504 server timeout,
#So use gpt-3.5-turbo-0301
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301",temperature= 0.0) 

#Merge similar documents
qdocs = "".join([docs[i].page_content for i in range(len(docs))]) 

#Add the merged similar documents to the question and input it into `llm.call_as_llm`

#The question here is: List all shirts with sun protection in the form of a Markdown table and summarize them 
response = llm.call_as_llm(f"{qdocs}Question: Please list all shirts with sun protection in the form of a markdown table and summarize the description of each shirt") 

display(Markdown(response))
```

| Clothing name | Description summary |
| --- | --- |
| Men's TropicVibe Shirt, Short-Sleeve | Men's short-sleeved shirt, built-in UPF 50+ sun protection, lightweight and comfortable, front and back vents, two front pockets, wrinkle-resistant, the highest level of sun protection. |
| Men's Tropical Plaid Short-Sleeve Shirt | Men's short-sleeved shirt, UPF 50+ sun protection, 100% polyester, wrinkle-free, front and back ventilation|
| Men's Plaid Tropic Shirt, Short-Sleeve | Men's short-sleeved shirt, UPF 50+ sun protection, 52% polyester and 48% nylon, wrinkle-resistant, front and back vents, two front pockets, and the highest level of sun protection. |
| Girls' Ocean Breeze Long-Sleeve Stripe Shirt | Girls' long-sleeved shirt, UPF 50+ sun protection, nylon Lycra®-elastane blend, quick drying, fade-resistant, water-resistant, and the highest level of sun protection, suitable for matching with our swimwear series. |

### 2.5 Answering questions using retrieval question-answering chains

Create a retrieval question-answering chain through LangChain to answer questions for the retrieved documents. The input of the retrieval question-answering chain includes the following
- `llm`: language model for text generation
- `chain_type`: incoming chain type, here we use stuff to combine all the queried documents into one document and pass it to the next step. Other approaches include: 
- Map Reduce: pass all chunks along with the question to the language model, get the response, use another language model call to summarize all the individual responses into the final answer, it can run on any number of documents. Can process a single question in parallel, but also requires moreCall. It treats all documents as independent
- Refine: Used to loop over many documents, effectively iterative, building on answers from previous documents, great for contextual information and building answers incrementally over time, dependent on results from previous calls. It usually takes longer and requires essentially as many calls as Map Reduce
- Map Re-rank: Make a single language model call for each document, ask it to return a score, pick the highest score, this relies on the language model knowing what the score should be, need to tell it that if it is relevant to the document, it should score high, and fine-tune the instructions there, can batch them relatively fast, but more expensive

![](../figures/C3/3_additional%20methods.png)
<div align='center'>Figure 3.5 Retrieval Q&A Chain</div>

- `retriever`: Retriever

```python
# Create a retriever based on vector storage
retriever = db.as_retriever()

qa_stuff = RetrievalQA.from_chain_type(
llm=llm, 
chain_type="stuff", 
retriever=retriever, 
verbose=True
)

#Create a query and run the chain on it
query = "Please list all shirts with sun protection in a markdown table, summarizing the description of each shirt"

response = qa_stuff.run(query)

display(Markdown(response)) 
```

> Entering new RetrievalQA chain...

> Finished chain.

| Number | Name | Description |
| --- | --- | --- |
| 618 | Men's Tropical Plaid Short-Sleeve Shirt | Made of 100% polyester, lightweight, wrinkle-resistant, front and back vents, two front pleated pockets, UPF 50+ sun protection rating, blocks 98% of UV rays |
| 374 | Men's Plaid Tropic Shirt, Short-Sleeve | Made of 52% polyester and 48% nylon, lightweight, wrinkle-resistant, front and back vents, two front pleated pockets, UPF 50+Sun protection rating, blocks 98% of UV rays |
| 535 | Men's TropicVibe Shirt, Short-Sleeve | Made of 71% nylon and 29% polyester, lightweight, wrinkle-resistant, front and back vents, two front pleated pockets, UPF 50+ sun protection rating, blocks 98% of UV rays |
| 293 | Girls' Ocean Breeze Long-Sleeve Stripe Shirt | Nylon Lycra®-elastane blend, long sleeves, UPF 50+ sun protection rating, blocks 98% of UV rays, quick drying, fade-resistant, easily pairs with our swimwear collection |

In summary: These shirts are both sun-protective, have a UPF 50+ sun protection rating, and block 98% of UV rays. They are both lightweight, wrinkle-resistant, have front and back vents, and front pleated pockets. The girl's long-sleeved striped shirt is made of a nylon Lycra®-elastic fiber blend, which is quick-drying and fade-resistant, and can be easily matched with the swimsuit series.

You can see that the two methods in parts 2.5 and 2.6 return the same results.

## English version tips

**1. Directly use vector storage query**

```python
from langchain.document_loaders import CSVLoader 
fromlangchain.indexes import VectorstoreIndexCreator

file = '../data/OutdoorClothingCatalog_1000.csv'

loader = CSVLoader(file_path=file)

index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

response = index.query(query)

display(Markdown(response))
```

| Name | Description |
| --- | --- |
| Men's Tropical Plaid Short-Sleeve Shirt | UPF 50+ rated,100% polyester, wrinkle-resistant, front and back cape venting, two front bellows pockets |
| Men's Plaid Tropic Shirt, Short-Sleeve | UPF 50+ rated, 52% polyester and 48% nylon, machine washable and dryable, front and back cape venting, two front bellows pockets |
| Men's TropicVibe Shirt, Short-Sleeve | UPF 50+ rated, 71% Nylon, 29% Polyester, 100% Polyester knit mesh, machine wash and dry, front and back cape venting, two front bellows pockets |
| Sun Shield Shirt by | UPF 50+ rated, 78% nylonon, 22% Lycra Xtra Life fiber, handwash, line dry, wicks moisture, fits comfortably over swimsuit, abrasion resistant |

All four shirts provide UPF 50+ sun protection, blocking 98% of the sun's harmful rays. The Men's Tropical Plaid Short-Sleeve Shirt is made of 100% polyester and is wrinkle-resistant

**2. Combining representation model and vector storage**

```python

from langchain.document_loaders import CSVLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import DocArrayInMemorySearchembeddings = OpenAIEmbeddings() 
embed = embeddings.embed_query("Hi my name is Harrison")

print("\n\033[32m vector representation length: \033[0m \n", len(embed))

print("\n\033[32m vector representation first 5 elements: \033[0m \n", embed[:5])

file = '../data/OutdoorClothingCatalog_1000.csv'

loader = CSVLoader(file_path=file)

docs = loader.load()
embeddings = OpenAIEmbeddings() 
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

query = "Please suggest a shirt with sunblocking"

docs = db.similarity_search(query)

print("\n\033[32m returnsNumber of documents: \033[0m \n", len(docs))
print("\n\033[32mFirst document: \033[0m \n", docs[0])

# Use query results to construct prompts to answer questions
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301",temperature = 0.0)

qdocs = "".join([docs[i].page_content for i in range(len(docs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.")

print("\n\033[32mUse query results to construct prompts to answer questions: \033[0m \n", docs[0])
display(Markdown(response))

# Use retrieval question-answering chains to answer questions
retriever = db.as_retriever() 

qa_stuff = RetrievalQA.from_chain_type(
llm=llm, 
chain_type="stuff", 
retriever=retriever, 
verbose=True
)

query = "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

response = qa_stuff.run(query)

print("\n\033[32m Use the retrieval Q&A chain to answer the question: \033[0m \n")

display(Markdown(response)) 
```

Length of vector representation: 
1536

First 5 elements of vector representation: 
[-0.021913960932078383, 0.006774206755842609, -0.018190348816400977,-0.039148249368104494, -0.014089343366938917]

Number of documents returned: 
4

First document: 
page_content=': 255\nname: Sun Shield Shirt by\ndescription: "Block the sun, not the fun – our high-performance sun shirt is guaranteed to protect from harmful UV rays. \n\nSize & Fit: Slightly Fitted: Softly shapes the body. Falls at hip.\n\nFabric & Care: 78% nylon, 22% Lycra Xtra Life fiber. UPF 50+ rated – the highest rated sun protection possible. Handwash, line dry.\n\nAdditional Features: Wicks moisture for quick-drying comfort. Fits comfortably over your favorite swimsuit. Abrasion resistant for season after season of wear. Imported.\n\nSun Protection That Won\'t Wear Off\nOur high-performance fabric provides SPF 50+ sun protection, blocking 98% of the sun\'s harmful rays. This fabric is recommended by The Skin Cancer Foundation as an effective UV protectant.' metadata={'source': '../data/OutdoorClothingCatalog_1000.csv', 'row': 255}

Use query results to construct prompts to answer questions: 
page_content=':255\nname: Sun Shield Shirt by\ndescription: "Block the sun, not the fun – our high-performance sun shirt is guaranteed to protect from harmful UV rays. \n\nSize & Fit: Slightly Fitted: Softly shapes the body. Falls at hip.\n\nFabric & Care: 78% nylon, 22% Lycra Xtra Life fiber. UPF 50+ rated – the highest rated sun protection possible. Handwash, line dry.\n\nAdditional Features: Wicks moisture for quick-drying comfort. Fits comfortably over your favorite swimsuit. Abrasion resistant for seasonafter season of wear. Imported.\n\nSun Protection That Won\'t Wear Off\nOur high-performance fabric provides SPF 50+ sun protection, blocking 98% of the sun\'s harmful rays. This fabric is recommended by The Skin Cancer Foundation as an effective UV protectant.' metadata={'source': '../data/OutdoorClothingCatalog_1000.csv', 'row': 255}

| Name | Description |
| --- | --- |
| Sun Shield Shirt | High-performance sun shirt with UPF 50+ sun protection, moisture-wicking, and abrasion-resistant fabricic. Recommended by The Skin Cancer Foundation. |
| Men's Plaid Tropic Shirt | Ultracomfortable shirt with UPF 50+ sun protection, wrinkle-free fabric, and front/back cape venting. Made with 52% polyester and 48% nylon. |
| Men's TropicVibe Shirt | Men's sun-protection shirt with built-in UPF 50+ and front/back cape venting. Made with 71% nylon and 29% polyester. |
| Men's Tropical Plaid Short-Sleeve Shirt | Lightest hot-weather shirt with UPF 50+ sun protection, front/back cape venting, and twofront bellows pockets. Made with 100% polyester. |

All of these shirts provide UPF 50+ sun protection, blocking 98% of the sun's harmful rays. They also have additional features such as moisture-wicking, wrinkle-free fabric, and front/back cape venting for added comfort.

> Entering new RetrievalQA chain...

> Finished chain.

Use the RetrievalQA chain to answer questions: 

| Shirt Number | Name | Description |
| --- | --- | --- |
| 618 | Men's Tropical Plaid Short-Sleeve Shirt | RatedUPF 50+ for superior protection from the sun's UV rays. Made of 100% polyester and is wrinkle-resistant. With front and back cape venting that lets in cool breezes and two front bellows pockets. |
| 374 | Men's Plaid Tropic Shirt, Short-Sleeve | Rated to UPF 50+ and offers sun protection. Made with 52% polyester and 48% nylon, this shirt is machine washable and dryable. Additional features include front and back cape venting, two front bellows pockets. |
| 535 | Men's TropicVibe Shirt, Short-SleeveEve | Built-in UPF 50+ has the lightweight feel you want and the coverage you need when the air is hot and the UV rays are strong. Made with 71% Nylon, 29% Polyester. Wrinkle resistant. Front and back cape venting lets in cool breezes. Two front bellows pockets. |
| 255 | Sun Shield Shirt | High-performance sun shirt is guaranteed to protect from harmful UV rays. Made with 78% nylon, 22% Lycra Xtra Life fiber. Wicks moisture for quick-drying comfort. Fits comfortably over your favorite swimsuit.Abrasion-resistant. |

All of the shirts listed above provide sun protection with a UPF rating of 50+ and block 98% of the sun's harmful rays. The Men's Tropical Plaid Short-Sleeve Shirt is made of 100% polyester and has front and back cape venting and two front bellows pockets. The Men's Plaid Tropic Shirt, Short-Sleeve is made with 52% polyester and 48% nylon and has front and back cape venting and two front bellows pockets. The Men's TropicVibe Shirt, Short-Sleeve is made with 71% Nylon, 29%Polyester and has front and back cape venting and two front bellows pockets. The Sun Shield Shirt is made with 78% nylon, 22% Lycra Xtra Life fiber and is abrasion-resistant. It fits comfortably over your favorite swimsuit.