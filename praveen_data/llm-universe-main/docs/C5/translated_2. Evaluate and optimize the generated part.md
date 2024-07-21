# Evaluate and optimize the generation part

**Note: The source code corresponding to this article is in the [Github open source project LLM Universe](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C5%20%E7%B3%BB%E7%BB%9F%E8%AF%84%E4%BC%B0%E4%B8%8E%E4%BC%98%E5%8C%96/2.%E8%AF%84%E4%BC%B0%E5%B9%B6%E4%BC%98%E5%8C%96%E7%94%9F%E6%88%90%E9%83%A8%E5%88%86.ipynb), readers are welcome to download and run, welcome to give us a Star~**

In the previous chapter, we talked about how to evaluate the overall performance of a large model application based on the RAG framework. By constructing a validation set in a targeted manner, we can use a variety of methods to evaluate system performance from multiple dimensions. However, the purpose of the evaluation is to better optimize the application effect. To optimize the application performance, we need to combine the evaluation results, split the evaluated Bad Cases, and evaluate and optimize each part separately.

RAG stands for Retrieval Enhanced Generation, so it has two core parts: the retrieval part and the generation part. Retrieval partThe core function of the retrieval part is to ensure that the system can find the corresponding answer fragment according to the user query, while the core function of the generation part is to ensure that after obtaining the correct answer fragment, the system can give full play to the ability of the big model to generate a correct answer that meets the user's requirements.

To optimize a big model application, we often need to start from both parts at the same time, evaluate the performance of the retrieval part and the optimization part respectively, find out the bad cases and optimize the performance in a targeted manner. As for the generation part, in the case of a limited use of the big model base, we often optimize the generated answers by optimizing Prompt Engineering. In this chapter, we will first combine the big model application example we just built - the personal knowledge base assistant, to explain to you how to evaluate and analyze the performance of the generation part, find out the bad cases in a targeted manner, and optimize the generation part by optimizing Prompt Engineering.

Before we start, let's load our vector database and search chain:

```python
import sys
sys.path.append("../C3 Build Knowledge Base") # Put the parent directory into the system path

# Use Zhipu Embedding API. Note that you need to download the encapsulation code implemented in the previous chapter to your local computer
from zhipuai_embedding importZhipuAIEmbeddings

from langchain.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv()) # read local .env file
zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Define Embeddings
embedding = ZhipuAIEmbeddings()

# Vector database persistence path
persist_directory = '../../data_base/vector_db/chroma'

# Load database
vectordb = Chroma(
persist_directory=persist_directory, # Allows us to save the persist_directory directory to disk
embedding_function=embedding
)
# Use OpenAI GPT-3.5 model
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0)

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

```

Let's first create a template-based retrieval chain using the initialized Prompt:

```python
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

template_v1 = """Use the following context to answer the final question. If you don't know the answer, say you don't know, don't try to make up an answer. Use no more than three sentences. Try to keep your answer brief and to the point. Always say "Thanks for your question!" at the end of your answer.
{context}
Question: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
template=template_v1)

qa_chain = RetrievalQA.from_chain_type(llm,
retriever=vectordb.as_retriever(),
return_source_documents=True,
chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
```

Test the effect first:

```python
question = "What is a pumpkin book"
result = qa_chain({"query": question})
print(result["result"])
```

Pumpkin Book is a book that analyzes and supplements the derivation details of the formulas that are difficult to understand in "Machine Learning" (Watermelon Book). The best way to use Pumpkin Book is to use Watermelon Book as the main line, and then refer to Pumpkin Book when you encounter formulas that are difficult to derive or that you cannot understand. Thank you for your question!

## 1. Improve the quality of intuitive answers

There are many ways to find Bad Cases. The most intuitive and simplest is to evaluate the quality of intuitive answers, and combine the original data content to determine where there are deficiencies. For example, the above test can be constructed into a Bad Case:

Question: What is Pumpkin Book
Initial answer: Pumpkin Book is a book that analyzes and supplements the derivation details of the formulas that are difficult to understand in "Machine Learning" (Watermelon Book). Thank you for your question!

Insufficient: The answer is too brief and needs to be more specific; Thank you for your question feels rigid and can be removed
We then modify the Prompt template in a targeted manner, add a requirement for specific answers, and remove the "Thank you for your question" part:

```python
template_v2 = """Use the following context to answer the final questions. If you don't know the answer, just say you don't know and don't try to make up an answer. You should make your answer as detailed and specific as possible, but stay on topic. If you don't know the answer, just say you don't know.The answer is long, please divide it into sections as appropriate to improve the reading experience of the answer.
{context}
Question: {question}
Useful answer: """

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
template=template_v2)
qa_chain = RetrievalQA.from_chain_type(llm,
retriever=vectordb.as_retriever(),
return_source_documents=True,
chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

question = "What is a pumpkin book"
result = qa_chain({"query": question})
print(result["result"])
```

Pumpkin Book is a supplementary book for "Machine Learning" (Xigua Book) by Zhou Zhihua. It aims to analyze the formulas in Xigua Book that are more difficult to understand, and supplement the specific derivation details to help readers better understand the knowledge in the field of machine learning. The content of Pumpkin Book is expressed with Xigua Book as the prerequisite knowledge. The best way to use it is to consult it when you encounter a formula that you cannot derive or understand. The writing team of Pumpkin Book is committed to helping readers become qualified "sophomore students with a solid foundation in science and engineering mathematics", and provides online reading addresses and the latest PDF acquisition address for readers to use.

As you can see, the improved v2 version can give more specific and detailed answers, solving the previous problems. But we can think further, will requiring the model to give specific and detailed answers lead to unclear and vague answers to some key points? We test the following questions:

```python
question = "What are the principles for constructing Prompt when using a large model?"
result = qa_chain({"query": question})
print(result["result"])
```

When using a large language model, what are the principles for constructing Prompt?The principles of ompt mainly include writing clear and specific instructions and giving the model enough time to think. First, the prompt needs to express the requirements clearly and clearly, and provide enough contextual information to ensure that the language model accurately understands the user's intentions. This is like explaining things to an alien who knows nothing about the human world, which requires a detailed and clear description. An overly brief prompt will make it difficult for the model to accurately grasp the task requirements.

Secondly, it is also crucial to give the language model sufficient reasoning time. Similar to the time humans need to think when solving problems, the model also needs time to reason and generate accurate results. Hasty conclusions often lead to incorrect outputs. Therefore, when designing prompts, the requirement of step-by-step reasoning should be added to give the model enough time to think logically, thereby improving the accuracy and reliability of the results.

By following these two principles, designing optimized prompts can help language models fully realize their potential and complete complex reasoning and generation tasks. Mastering these prompt design principles is an important step for developers to successfully apply language models. In practical applications, constantly optimizing and adjusting prompts and gradually approaching the optimal form is a key strategy for building efficient and reliable model interactions.

As we can see, the model’s answer to our question about the LLM course is indeed detailed and specific, and it also fully refers to the course content. However, the answer begins with words such as firstly and secondly, andThe overall answer is divided into 4 paragraphs, which makes the answer not particularly clear and difficult to read. Therefore, we construct the following Bad Case:

Question: What are the principles for constructing prompts when using large models?

Initial answer: Omitted

Deficiencies: No focus, vague

For this Bad Case, we can improve the prompt and require it to mark the answers with several points in points to make the answers clear and specific:

```python
template_v3 = """Use the following context to answer the last question. If you don't know the answer, say you don't know, don't try to make up an answer. You should make your answer as detailed and specific as possible, but don't go off topic. If the answer is long, please divide it into paragraphs as appropriate to improve the reading experience of the answer.
If the answer has several points, you should answer it with points in points to make the answer clear and specific
{context}
Question: {question}
Useful answer:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
template=template_v3)
qa_chain = RetrievalQA.from_chain_type(llm,
retriever=vectordb.as_retriever(),
return_source_documents=True,
chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

question = "What are the principles for constructing prompts when using large models?"
result = qa_chain({"query": question})
print(result["result"])
```

1. Writing clear and specific instructions is the first principle for constructing prompts. Prompts need to clearly express requirements and provide sufficient context so that the language model can accurately understand the intent. Too brief prompts will make it difficult for the model to complete the task.

2. Giving the model enough time to think is the second principle for constructing prompts. Language models need time to reason and solve complex problems.Conclusions drawn hastily may be inaccurate. Therefore, prompts should include requirements for step-by-step reasoning, so that the model has enough time to think and generate more accurate results.

3. When designing prompts, specify the steps required to complete the task. By giving a complex task and a series of steps to complete the task, the model can better understand the task requirements and improve the efficiency of task completion.

4. Iterative optimization is a common strategy for constructing prompts. Through the process of continuous trial, analysis of results, and improvement of prompts, the optimal prompt form is gradually approached. Successful prompts are usually obtained through multiple rounds of adjustments.

5. Adding table descriptions is a way to optimize prompts. Requiring the model to extract information and organize it into a table, specifying the columns, table names, and formats of the table can help the model better understand the task and generate expected results.

In short, the principles for constructing prompts include clear and specific instructions, giving the model enough time to think, specifying the steps required to complete the task, iterative optimization, and adding table descriptions. These principles can help developers design efficient and reliable prompts and maximize the potential of language models.

There are many ways to improve the quality of answers. The core is to think about specific business, find out the unsatisfactory points in the initial answers, and improve them in a targeted manner. I will not go into details here..

## 2. Indicate the source of knowledge to improve credibility

Due to the hallucination problem of large models, sometimes we suspect that the model's answer is not derived from the existing knowledge base content, which is particularly important for some scenarios that need to ensure authenticity, such as:

```python
question = "What is the definition of reinforcement learning"
result = qa_chain({"query": question})
print(result["result"])
```

Reinforcement learning is a machine learning method that aims to allow an agent to learn how to make a series of good decisions through interaction with the environment. In reinforcement learning, the agent will choose an action based on the state of the environment, and then adjust its strategy based on the feedback (reward) from the environment to maximize the long-term reward. The goal of reinforcement learning is to make the best decision under uncertainty, similar to the process of letting a child learn to walk through continuous trial and error. Reinforcement learning has a wide range of applications, including game play, robot control, traffic optimization and other fields. In reinforcement learning, the agent and the environment interact continuously, and the agent adjusts its strategy based on the feedback from the environment to obtain the maximum reward.

We can ask the model to indicate the source of knowledge when generating answers, which can prevent the model from fabricating knowledge that does not exist in the given data. At the same time, it can also improve our credibility of the answers generated by the model:

```python
template_v4 = """Use the following context to answer the last question. If you don't know the answer, just say you don't know, don't try to make up an answer. You should make your answer as detailed and specific as possible, but don't get off topic. If your answer is long, please break it up into paragraphs as appropriate to make it easier to read. If your answer has several points, you should give your answer in bullet points to make it clear and specific. Please include the source of your answer to ensure the correctness of your answer.
{context}
Question: {question}
Useful answer:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
template=template_v4)
qa_chain = RetrievalQA.from_chain_type(llm,
retriever=vectordb.as_retriever(),
return_source_documents=True,
chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

question = "What is the definition of reinforcement learning"
result = qa_chain({"query": question})
print(result["result"])
```

Reinforcement learning is a machine learning method that aims to enable an agent to learn how to make a series of good decisions through interaction with the environment. In this process, the agent adjusts its behavior based on the feedback (rewards) from the environment to maximize the sum of long-term rewards. The goal of reinforcement learning is to make the best decision under uncertainty, similar to the process of letting a child learn to walk through continuous trial and error. The interaction process of reinforcement learning consists of two parts: the agent and the environment. The agent selects actions based on the state of the environment, and the environment outputs the next state and reward based on the action of the agent. Reinforcement learning has a wide range of applications, including game play, robot control, traffic management and other fields. [Source: Mushroom Book One Language Two Language Two Reinforcement Learning Tutorial].

However, attaching the original source often results in an increase in context and a decrease in response speed. We need to consider whether to require the original source to be attached based on the business scenario.

## 3.Constructing a thought chain

Large models can often understand and execute instructions well, but the model itself still has some limitations, such as illusions of large models, inability to understand more complex instructions, inability to execute complex steps, etc. We can minimize its ability limitations by constructing a thought chain and constructing Prompt into a series of steps. For example, we can construct a two-step thought chain and require the model to reflect in the second step to eliminate the illusion problem of large models as much as possible.

First, we have a Bad Case:

Question: How should we construct an LLM project
Initial answer: Omitted
Insufficient: In fact, the content about how to construct an LLM project in the knowledge base is to use the LLM API to build an application. The model's answer seems to make sense, but it is actually an illusion of a large model, which is obtained by splicing some related texts. There are problems

```python
question = "How should we construct an LLM project"
result = qa_chain({"query": question})
print(result["result"])
```

To build an LLM project, you need to consider the following steps:

1. Determine the project goals and requirements: First, you need to clarify what problem your project is to solve or what goal it is to achieve, and determine the need to use LLM's specific scenarios and tasks.

2. Collect and prepare data: Collect and prepare appropriate data sets according to project requirements, ensure the quality and diversity of data, and improve the performance and effect of LLM.

3. Design prompts and fine-tune instructions: Design appropriate prompts according to project requirements, ensure that the instructions are clear and can guide LLM to generate expected text.

4. Train and fine-tune the model: Use basic LLM or instruction fine-tuning LLM to train and fine-tune the data to improve the performance and accuracy of the model on specific tasks.

5. Test and evaluate the model: After training, test and evaluate the model to check its performance and effect in different scenarios, and make necessary adjustments and optimizations based on the evaluation results.

6. Deploy and apply the model: Deploy the trained LLM model to actual applications to ensure that it can run normally and achieve the expected results, and continuously monitor and optimize the performance of the model.

Source: Summarize based on the context provided.

To this end, we can optimize the prompt and turn the previous prompt into two steps, requiring the model to reflect in the second step:

```python
template_v4 = """
Please follow the steps below:
① Use the following context to answer the mostIf you don't know the answer, just say you don't know, don't try to make up an answer.
You should make your answer as detailed and specific as possible, but don't go off topic. If your answer is long, please break it up into paragraphs as appropriate to make it easier to read.
If your answer has several points, you should answer them in points to make them clear and specific.
Context:
{context}
Question:
{question}
Helpful answers:
② Based on the context provided, reflect on whether there is anything incorrect or not based on the context in your answer. If so, answer that you don't know
Make sure you follow every step and don't skip any steps.
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
template=template_v4)
qa_chain = RetrievalQA.from_chain_type(llm,
retriever=vectordb.as_retriever(),return_source_documents=True,
chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

question = "How should we construct an LLM project"
result = qa_chain({"query": question})
print(result["result"])
```

According to the information provided in the context, the following steps need to be considered to construct an LLM project:

1. Determine the project goal: First, you need to clarify what your project goal is, whether it is text summarization, sentiment analysis, entity extraction or other tasks. Determine how to use LLM and how to call the API interface based on the project goal.

2. Design Prompt: Design a suitable prompt based on the project goal. The prompt should be clear and unambiguous, guiding LLM to generate expected results. The design of the prompt needs to take into account the specific requirements of the task. For example, in the text summary task, the prompt should contain the text content that needs to be summarized.

3. Call the API interface: According to the designed prompt,Generate results by programming and calling the API interface of LLM. Make sure that the API interface is called correctly to obtain accurate results.

4. Analyze results: After obtaining the results generated by LLM, analyze the results to ensure that the results meet the project goals and expectations. If the results do not meet expectations, you can adjust the prompt or other parameters to generate results again.

5. Optimize and improve: According to the feedback of the analysis results, continuously optimize and improve the LLM project to improve the efficiency and accuracy of the project. You can try different prompt designs, adjust the parameters of the API interface, and other methods to optimize the project.

Through the above steps, you can build an effective LLM project and use the powerful functions of LLM to achieve tasks such as text summarization, sentiment analysis, and entity extraction to improve work efficiency and accuracy. If there is anything unclear or further guidance is needed, you can always seek help from experts in related fields.

It can be seen that after asking the model to reflect on itself, the model repaired its illusion and gave the correct answer. We can also complete more functions by constructing a thinking chain, which will not be repeated here. Readers are welcome to try.

## 4. Add a command parsing

We often face a requirement that we need the model to output in the format we specify. However, since we use Prompt Template to fill in user questions, there areThe format requirements are often ignored, for example:

```python
question = "What are the categories of LLM? Return me a Python List"
result = qa_chain({"query": question})
print(result["result"])
```

According to the information provided by the context, the classification of LLM (Large Language Model) can be divided into two types, namely basic LLM and instruction fine-tuning LLM. Basic LLM is a model that trains the ability to predict the next word based on text training data, usually by training on a large amount of data to determine the most likely word. Instruction fine-tuning LLM fine-tunes the basic LLM to better adapt to specific tasks or scenarios, similar to providing instructions to another person to complete a task.

According to the context, a Python List can be returned, which contains two categories of LLM: ["Basic LLM", "Instruction fine-tuning LLM"].

As you can see, although we require the model to return a Python List, the output requirement is wrapped in Template and ignored by the model. For this problem, we can construct a Bad Case:

Question: What are the categories of LLM? Return me a Pythonn List
Initial answer: According to the context provided, LLM can be divided into basic LLM and instruction fine-tuning LLM.
Insufficient: The output is not in accordance with the requirements in the instruction

For this problem, an existing solution is to add a layer of LLM before our retrieval LLM to implement instruction parsing, and separate the format requirements of user questions from the content of the questions. This idea is actually the prototype of the currently popular Agent mechanism, that is, for user instructions, set up an LLM (ie Agent) to understand the instructions, determine what tools need to be executed for the instructions, and then call the tools to be executed in a targeted manner. Each tool can be an LLM based on different prompt engineering, or it can be a database, API, etc. In fact, there is an Agent mechanism designed in LangChain, but we will not go into details in this tutorial. Here we only implement this function based on the native interface of OpenAI:

```python
# Use the native interface of OpenAI mentioned in Chapter 2

from openai import OpenAI

client = OpenAI(
# This is the default and can be omitted
api_key=os.environ.get("OPENAI_API_KEY"),
)
def gen_gpt_messages(prompt):
'''
Construct GPT model request parameters messages

Request parameters:
prompt: corresponding user prompt words
'''
messages = [{"role": "user", "content": prompt}]
return messages

def get_completion(prompt, model="gpt-3.5-turbo", temperature = 0):
'''
Get GPT model call results

Request parameters:
prompt: corresponding prompt words
model: the model called, the default is gpt-3.5-turbo, you can also select other models such as gpt-4 as needed
temperature: the temperature coefficient of the model output, which controls the randomness of the output, and the value range is 0~2. The lower the temperature coefficient, the more consistent the output content.
'''
response = client.chat.completions.create(
model=model,
messages=gen_gpt_messages(prompt),
temperature=temperature,
)
if len(response.choices) > 0:
return response.choices[0].message.content
return "generate answer error"

prompt_input = '''
Please determine whether the following question contains format requirements for the output and output according to the following requirements:
Please return me a parseable Python list. The first element of the list is the format requirement for the output, which should be an instruction; the second element is the original question without the format requirement
If there is no format requirement, please set the first element to empty
Questions to be determined:
~~~
{}
~~~
Do not output any other content or format to ensure that the returned result is parseable.
'''

```

Let's test the LLM's ability to decompose format requirements:

```python
response = get_completion(prompt_input.format(question))
response
```

'```\n["Return a Python List to me", "What are the categories of LLM?"]\n```'

As you can see, through the above prompt, LLM can well implement the output format parsing. Next, we can set another LLM to parse the output content according to the output format requirements:

```python
prompt_output = '''
Please answer the question according to the given format requirements based on the answer text and output format requirements
Questions to be answered:
~~~
{}
~~~
Answer text:
~~~
{}
~~~
Output format requirements:
~~~
{}
~~~
'''
```

Then we can connect the two LLMs with the search chain in series:

```python
question = 'What are the categories of LLM? Return me a Python List'
# First split the format requirement and the question
input_lst_s = get_completion(prompt_input.format(question))
# Find the start and end characters of the list after splitting
start_loc = input_lst_s.find('[')
end_loc = input_lst_s.find(']')
rule, new_question = eval(input_lst_s[start_loc:end_loc+1])
# Then use the split question to call the search chain
result = qa_chain({"query": new_question})
result_context = result["result"]
# Then call the output format parsing
response = get_completion(prompt_output.format(new_question, result_context, rule))
response
```

"['Basic LLM', 'Instruction fine-tuning LLM']"

As you can see, after the above steps, we have successfully achieved the output format limitation. Of course, in the above code, the core is to introduce the Agent idea. In fact, whether it is the Agent mechanism or the Parser mechanism (that is, the limited output format), LangChain provides a mature tool chain for use. Interested readers are welcome to explore in depth. I will not explain it here.

Through the ideas explained above, combined with the actualBusiness situation, we can continuously find bad cases and optimize prompts in a targeted manner, thereby improving the performance of the generation part. However, the premise of the above optimization is that the retrieval part can retrieve the correct answer fragment, that is, the accuracy and recall of the retrieval are as high as possible. So, how can we evaluate and optimize the performance of the retrieval part? We will explore this issue in depth in the next chapter.

**Note: The source code corresponding to this article is in the [Github open source project LLM Universe](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C5%20%E7%B3%BB%E7%BB%9F%E8%AF%84%E4%BC%B0%E4%B8%8E%E4%BC%98%E5%8C%96/2.%E8%AF%84%E4%BC%B0%E5%B9%B6%E4%BC%98%E5%8C%96%E7%94%9F%E6%88%90%E9%83%A8%E5%88%86.ipynb), welcome to download and run, welcome to give us a star~**