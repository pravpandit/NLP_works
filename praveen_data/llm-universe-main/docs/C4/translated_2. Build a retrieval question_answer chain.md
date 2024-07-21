# 4.2 Build a retrieval question-answering chain

The source files involved in the text can be obtained from the following path:

> - [C3 builds a knowledge base](https://github.com/datawhalechina/llm-universe/tree/main/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93)

---

In the `C3 builds a database` chapter, we have introduced how to build a vector knowledge base based on your own local knowledge documents. In the following content, we will use the built vector database to recall query questions, and combine the recall results with the query to build a prompt, which is input into the large model for question and answer. 

## 1. Load the vector database

First, we load the vector database that has been built in the previous chapter. Note that you need to use the same Emedding as when building it.

```python
import sys
sys.path.append("../C3 Build Knowledge Base") # Put the parent directory into the system path

# Use Zhipu Embedding API. Note that you need to download the encapsulation code implemented in the previous chapter to your local computer
from zhipuai_embedding import ZhipuAIEmbeddings

from langchain.vectorstores.chroma import Chroma
```

Load your API_KEY from the environment variable

```python
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv()) # read local .env file
zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']
```

Load the vector database, which contains the Embeddings of multiple documents under ../../data_base/knowledge_db

```python
# Define Embeddings
embedding = ZhipuAIEmbeddings()

# Vector database persistence path
persist_directory = '../C3 Build Knowledge Base/data_base/vector_db/chroma'

# Load database
vectordb = Chroma(
persist_directory=persist_directory, # allows us to save the persist_directory directory to disk
embedding_function=embedding
)
```

```python
print(f"Number of vectors stored: {vectordb._collection.count()}")
```

Number of vectors stored: 20

We can test the loaded vector database and use a question query to retrieve vectors. The following code will search the vector database based on similarity and return the top k most similar documents.

> ⚠️Before using similarity search, make sure you have installed the OpenAI open source fast word segmentation tool tiktoken package: `pip install tiktoken`

```python
question = "What is prompt engineering?"
docs = vectordb.similarity_search(question,k=3)
print(f"Number of retrieved contents: {len(docs)}")
```

Retrieved contentsNumber of capacity: 3

Print the retrieved content

```python
for i, doc in enumerate(docs):
print(f"The {i}th content retrieved: \n {doc.page_content}", end="\n-----------------------------------------------------\n")
```

The 0th content retrieved: 
On the contrary, we should use prompts to guide the language model to think deeply. It can be required to first list various views on the problem, explain the reasoning basis, and then draw the final conclusion. Adding the requirement of step-by-step reasoning in prompts can allow the language model to devote more time to logical thinking, and the output results will be more reliable and accurate.

In summary, giving the language model sufficient reasoning time is a very important design principle in prompt engineering. This will greatly improve the effect of language models in handling complex problems and is also the key to building high-quality prompts. Developers should pay attention to leaving room for thinking for the model to maximize the potential of the language model.

2.1 Specify the steps required to complete the task

Next, we will give a complex taskTask, give a series of steps to complete the task, to demonstrate the effectiveness of this strategy.

First, we describe the story of Jack and Jill, and give the prompt to perform the following operations: First, summarize the text delimited by three backticks in one sentence. Second, translate the summary into English. Third, list each name in the English summary. Fourth, output a JSON object containing the following keys: English summary and the number of names. The output is required to be separated by newlines.
-----------------------------------------------------
The first content retrieved: 
Chapter 2 Prompt Principles

How to use Prompt to give full play to the performance of LLM? First, we need to know the principles of designing Prompt, which are the basic concepts that every developer must know when designing Prompt. This chapter discusses two key principles for designing efficient Prompt: writing clear and specific instructions and giving the model enough time to think. Mastering these two points is particularly important for creating reliable language model interactions.

First, the prompt needs to express the requirements clearly and clearly, and provide sufficient context so that the language model can accurately understand our intentions, just like explaining the human world in detail to an alien. Too brief prompts often make it difficult for the model to grasp the desiredSpecific tasks to be completed.

Secondly, it is also extremely important to give the language model enough time to reason. Just like when humans solve problems, hasty conclusions are often wrong. Therefore, Prompt should add the requirement of step-by-step reasoning to give the model enough time to think, so that the generated results are more accurate and reliable.

If Prompt is optimized in both aspects, the language model will be able to maximize its potential and complete complex reasoning and generation tasks. Mastering these Prompt design principles is an important step for developers to achieve successful language model applications.

1. Principle 1 Write clear and specific instructions
-----------------------------------------------------
The second content retrieved: 
1. Principle 1 Write clear and specific instructions

Dear readers, when interacting with language models, you need to keep one thing in mind: express your needs in a clear and specific way. Suppose you have a new friend from another planet sitting in front of you who knows nothing about human language and common sense. In this case, you need to make your intentions very clear and leave no ambiguity. Similarly, when providing a prompt, make your needs and context clear in a sufficiently detailed and understandable way.

AndIt doesn't mean that prompts must be very short and concise. In fact, in many cases, longer and more complex prompts make it easier for the language model to grasp the key points and give expected responses. The reason is that complex prompts provide richer context and details, allowing the model to more accurately grasp the required operations and responses.

So, remember to express prompts in clear and detailed language, just like explaining the human world to aliens, "Adding more context helps the model understand you better."

Based on this principle, we provide several tips for designing prompts.

1.1 Use delimiters to clearly represent different parts of the input
-----------------------------------------------------

## 2. Create an LLM

Here, we call OpenAI's API to create an LLM. Of course, you can also use other LLM APIs to create

```python
import os 
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
```

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0)

llm.invoke("Please introduce yourself!")
```

AIMessage(content='Hello, I am an intelligent assistant that specializes in providing users with various services and help. I can answer questions, provide information, solve problems, and so on. If you need anything, please feel free to let me know and I will try my best to help you. Thank you for your use!', response_metadata={'token_usage': {'completion_tokens': 81, 'prompt_tokens': 20, 'total_tokens': 101}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3bc1b5746c', 'finish_reason': 'stop', 'logprobs': None}) ## 3. BuildRetrieval QA Chain

```python
from langchain.prompts import PromptTemplate

template = """Use the following context to answer the last question. If you don't know the answer, say you don't know, don't try to make up an answer. Use no more than three sentences. Try to keep your answers brief and to the point. Always say "Thanks for your question!" at the end of your answer.
{context}
Question: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
template=template)

```

Create another retrieval chain based on the template:

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm,
retriever=vectordb.as_retriever(),
return_source_documents=True,
chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

```

The method RetrievalQA.from_chain_type() for creating a retrieval QA chain has the following parameters:
- llm: specify the LLM to be used
- Specify chain type: RetrievalQA.from_chain_type(chain_type="map_reduce"), or use the load_qa_chain() method to specify chain type.
- Custom prompt: By specifying the chain_type_kwargs parameter in the RetrievalQA.from_chain_type() method, the parameter: chain_type_kwargs = {"prompt": PROMPT}
- Return source document: By specifying in the RetrievalQA.from_chain_type() method:return_source_documents=True parameter; you can also use the RetrievalQAWithSourceChain() method to return the reference of the source document (coordinates or primary keys, indexes)

## 4. Retrieval QA chain effect test

```python
question_1 = "What is Pumpkin Book?"
question_2 = "Who is Wang Yangming?"
```

### 4.1 Prompt effect based on recall results and query

```python
result = qa_chain({"query": question_1})
print("The result of answering question_1 after the large model + knowledge base:")
print(result["result"])
```

d:\Miniconda\miniconda3\envs\llm2\lib\site-packages\langchain_core\_api\deprecation.py:117: LangChainDeprecationWarning: The function `__call__` was deprecated in LangChain0.1.0 and will be removed in 0.2.0. Use invoke instead.
warn_deprecated(

The result of answering question_1 after the big model + knowledge base:
Sorry, I don't know what Pumpkin Book is. Thank you for your question!

```python
result = qa_chain({"query": question_2})
print("The result of answering question_2 after the big model + knowledge base:")
print(result["result"])
```

The result of answering question_2 after the big model + knowledge base:
I don't know who Wang Yangming is.

Thank you for your question!

### 4.2 The effect of the big model answering itself

```python
prompt_template = """Please answer the following questions:
{}""".format(question_1)

### Question and answer based on the big model
llm.predict(prompt_template)
```d:\Miniconda\miniconda3\envs\llm2\lib\site-packages\langchain_core\_api\deprecation.py:117: LangChainDeprecationWarning: The function `predict` was deprecated in LangChain 0.1.7 and will be removed in 0.2.0. Use invoke instead.
warn_deprecated(

'Pumpkin book refers to a book about pumpkin, usually a book about pumpkin planting, maintenance, cooking and other aspects. Pumpkin book can also refer to a literary work with pumpkin as the theme.'

```python
prompt_template = """Please answer the following questions:
{}""".format(question_2)

### Question and answer based on large models
llm.predict(prompt_template)
```

'Wang Yangming (1472-1529),Yangming, courtesy name Xian and pseudonym Yangming, was a native of Shaoxing, Zhejiang. He was a famous philosopher, military strategist, educator, and politician in the Ming Dynasty. He proposed important ideas such as "cultivating conscience" and "investigating things to acquire knowledge", emphasizing that people have conscience in their hearts, and as long as they exercise their conscience, they can understand moral truth. His thoughts have a profound influence on later generations and are called "Yangming's philosophy of mind". '

> ⭐ Through the above two questions, we found that LLM did not answer some recent knowledge and non-common sense professional questions very well. And adding our local knowledge can help LLM give better answers. In addition, it also helps to alleviate the "hallucination" problem of large models.

## 5. Add the memory function of historical conversations

Now we have realized that by uploading local knowledge documents and then saving them to the vector knowledge base, by combining the query questions with the recall results of the vector knowledge base and inputting them into LLM, we get a much better result than directly asking LLM to answer. When interacting with language models, you may have noticed a key problem - **they don't remember your previous communication content**. This poses a great challenge when we build some applications (such as chatbots), making the conversation seem to lack real continuity. How to solve this problem?

### 1. Memory

In this section, we will introduce the storage module in LangChain, that is, how to embed previous conversations into the language model., giving it the ability to have continuous conversations. We will use `ConversationBufferMemory`, which saves a list of chat message histories, which will be passed to the chatbot along with the questions when answering questions, thus adding them to the context.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
memory_key="chat_history", # Keep consistent with the input variable of prompt.
return_messages=True # Will return the chat history as a list of messages instead of a single string
)
```

For more information on the use of Memory, including retaining a specified number of conversation rounds, saving a specified number of tokens, saving a summary of historical conversations, etc., please refer to the relevant documentation of the Memory section of langchain.

### 2. ConversationalRetrievalChain

The ConversationalRetrievalChain is based on the retrieval QA chain, adding the ability to handle conversation history.

Its workflow is:
1. Merge the previous conversation with the new question to generate a complete query statement.
2. Search the vector database for relevant documents for the query.
3. After obtaining the results, store all answers in the conversation memory area.
4. The user can view the complete conversation process in the UI.

![](../../figures/Modular_components.png)

This chain method puts the new question in the context of the previous conversation for retrieval, which can handle queries that rely on historical information. And keep all information in the conversation memory for easy tracking.

Next, let's test the effect of this conversation retrieval chain:

Use the vector database and LLM in the previous section! First, ask a question without historical dialogue "Can I learn about prompt engineering?" and check the answer.

```python from langchain.chains import ConversationalRetrievalChain retriever=vectordb.as_retriever() qa = ConversationalRetrievalChain.from_llm( llm, retriever=retriever, memory=memory
)
question = "Can I learn about prompt engineering?"
result = qa({"question": question})
print(result['answer'])
```

Yes, you can learn about prompt engineering. This module is based on Andrew Ng's Prompt Engineering for Developer course, which aims to share best practices and techniques for developing large language model applications using prompt words. The course will introduce the principles of designing effective prompts, including writing clear and specific instructions and giving the model enough time to think. By learning these contents, you can better utilize the performance of large language models and build excellent language model applications.

Then based on the answer, the next question is "Why does this course need to teach this knowledge?":

```python
question = "Why does this course need to teach this knowledge?"
result = qa({"question": question})
print(result['answer'])
```

This course needs to teach about prompt engineering, mainly to help developers better use large language models (LLM) to completevarious tasks. By learning Prompt Engineering, developers can learn how to design clear and unambiguous prompts to guide language models to generate expected text output. This skill is very important for developing applications and solutions based on large language models, and can improve the efficiency and accuracy of models.

As you can see, LLM accurately judges this knowledge, referring to the content as reinforcement learning knowledge, that is, we have successfully passed it historical information. This ability to continuously learn and associate previous and subsequent questions can greatly enhance the continuity and intelligence level of the question-answering system.

---

#### The above-mentioned source file acquisition path:

[C3 builds a knowledge base](https://github.com/datawhalechina/llm-universe/tree/main/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93)