# How to evaluate LLM applications

**Note: The source code corresponding to this article is in the [Github open source project LLM Universe](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C5%20%E7%B3%BB%E7%BB%9F%E8%AF%84%E4%BC%B0%E4%B8%8E%E4%BC%98%E5%8C%96/1.%E5%A6%82%E4%BD%95%E8%AF%84%E4%BC%B0%20LLM%20%E5%BA%94%E7%94%A8.ipynb). Readers are welcome to download and run it, and welcome to give us a Star~**

## 1. General ideas for verification and evaluation

Now, we have built a simple, generalized large model application. Looking back at the entire development process, we can find that large model development, which focuses on calling and using large models, pays more attention to verification and iteration than traditional AI development. Since you can quickly build an LLM-based application, define a prompt in a few minutes, and get feedback results in a few hours, it would be extremely cumbersome to stop and collect a thousand test samples. Because now, you can get results without any training samples.

Therefore, when using LLWhen you build an application, you might go through the following process: First, you tweak Prompt on a small sample of one to three examples, trying to get it to work on those samples. Then, as you test the system further, you might run into some tricky examples that can't be solved by Prompt or the algorithm. This is the challenge faced by developers building applications using LLM. In this case, you can add these extra few examples to the set you are testing, organically adding other difficult examples. Eventually, you will add enough of these examples to your gradually expanding development set that it becomes a bit inconvenient to manually run every example to test Prompt. Then, you start to develop some metrics for measuring the performance of these small sample sets, such as average accuracy. The interesting thing about this process is that if you feel that your system is good enough, you can stop at any time and stop improving it. In fact, many deployed applications stop at the first or second step, and they work very well.

![](../figures/C5-1-eval.png)

In this chapter, we will introduce the general methods of large model application verification and evaluation one by one, and design the verification and iteration process of this project to optimize the application function. However, please note that since system evaluation and optimization is a topic closely related to business, this chapter mainly introduces the theory. Readers are welcome to actively conduct their ownI practice and explore.

We will first introduce several methods for evaluating large model development. For tasks with simple standard answers, evaluation is easy to achieve; but large model development generally requires the implementation of complex generation tasks. How to achieve evaluation without simple answers or even standard answers, and accurately reflect the effect of the application, we will briefly introduce several methods.

As we continue to find bad cases and make targeted optimizations, we can gradually add these bad cases to the validation set to form a validation set with a certain number of samples. For this kind of validation set, it is impractical to evaluate one by one. We need an automatic evaluation method to achieve an overall evaluation of the performance on the validation set.

After mastering the general idea, we will specifically explore how to evaluate and optimize application performance in large model applications based on the RAG paradigm. Since large model applications developed based on the RAG paradigm generally include two core parts: retrieval and generation, our evaluation optimization will also focus on these two parts, respectively, to optimize the system retrieval accuracy and the generation quality under the given material.

In each section, we will first introduce some tips on how to find bad cases, as well as general ideas for optimizing search or prompts for bad cases. Note that in this process, you should always keep in mind the series of large models we described in the previous chapters.Development principles and techniques, and always ensure that the optimized system will not make mistakes on the samples that originally performed well.

Verification iteration is an important step in building LLM-centric applications. By constantly finding bad cases, adjusting prompts or optimizing retrieval performance in a targeted manner, we can drive the application to achieve the performance and accuracy we aim for. Next, we will briefly introduce several methods for large model development evaluation, and summarize the general idea from targeted optimization of a few bad cases to overall automated evaluation.

## 2. Large model evaluation method

In the specific large model application development, we can find bad cases, and continuously optimize prompts or retrieval architectures to solve bad cases, thereby optimizing the performance of the system. We will add each bad case we find to our verification set. After each optimization, we will re-verify all verification cases in the verification set to ensure that the optimized system will not lose its capabilities or performance degradation on the original good cases. When the validation set is small, we can use manual evaluation, that is, manually evaluate the quality of the system output for each validation case in the validation set; however, as the validation set continues to expand with the optimization of the system, its size will continue to increase, so that the time and labor cost of manual evaluation will increase to an unacceptable level. Therefore, we need to use automatic evaluation.The method of evaluation automatically evaluates the output quality of each verification case of the system, thereby evaluating the overall performance of the system.

We will first introduce the general ideas of manual evaluation for reference, and then introduce the general methods of automatic evaluation of large models in depth, and conduct actual verification on this system to comprehensively evaluate the performance of this system and prepare for further optimization and iteration of the system. Similarly, before officially starting, we first load our vector database and retrieval chain:

```python
import sys
sys.path.append("../C3 创建知识库") # Put the parent directory into the system path

# Use Zhipu Embedding API. Note that the encapsulation code implemented in the previous chapter needs to be downloaded locally
from zhipuai_embedding import ZhipuAIEmbeddings

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

```

### 2.1 General idea of ​​manual evaluation

In the early stage of system development, the verification group is small, and the simplest and most intuitive method isManually evaluate each verification case in the verification set. However, there are also some basic principles and ideas for manual evaluation, which are briefly introduced here for learners' reference. But please note that the evaluation of the system is strongly related to the business, and the design of specific evaluation methods and dimensions needs to be considered in depth in combination with specific business.

#### Principle 1 Quantitative evaluation

In order to ensure a good comparison of the performance of different versions of the system, quantitative evaluation indicators are very necessary. We should give a score to the answer to each verification case, and finally calculate the average score of all verification cases to get the score of this version of the system. The quantified dimension can be 0~5 or 0~100, which can be determined according to personal style and actual business conditions.

The quantified evaluation indicators should have certain evaluation specifications. For example, if condition A is met, the score can be y points to ensure relative consistency between different evaluators.

For example, we give two verification cases:

① Who is the author of "Pumpkin Book"?

② How should the Pumpkin Book be used?

Next, we use version A prompt (brief and to the point) and version B prompt (detailed and specific) to ask the model to answer:

```python
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

template_v1 = """Answer the last question using the following context. If you don't know the answer, say you don't know, don't try to make up an answer. Use three sentences at most. Try to keep your answer short and to the point. Always say "Thanks for your question!" at the end of your answer.
{context}
Question: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
template=template_v1)

qa_chain = RetrievalQA.from_chain_type(llm,
retriever=vectordb.as_retriever(),
return_source_documents=True,
chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

print("Question 1:")
question = "What is the relationship between Pumpkin Book and Watermelon Book?"

result = qa_chain({"query": question})
print(result["result"])

print("Question 2:")
question = "How should Pumpkin Book be used? "
result = qa_chain({"query": question})
print(result["result"])
```

Question 1:
The Pumpkin Book is parsed and supplemented with the Watermelon Book as the prerequisite knowledge, mainly to parse and supplement the formulas in the Watermelon Book that are more difficult to understand. The best way to use the Pumpkin Book is to use the Watermelon Book as the main line, and then consult the Pumpkin Book when you encounter a formula that you cannot derive or understand. Thank you for your question!
Question 2:
The Watermelon Book should be the main line, and you should consult the Pumpkin Book when you encounter a formula that you cannot derive or understand. It is not recommended for beginners to delve into the formulas in Chapter 1 and Chapter 2, and come back when you are more proficient. If you need to consult a formula that is not in the Pumpkin Book or find an error, you can feedback on GitHub. Thank you for your question!

The above is the answer to version A Prompt, let's test version B:```python
template_v2 = """Use the following context to answer the final question. If you don't know the answer, say you don't know, don't try to make up an answer. You should make your answer as detailed and specific as possible, but stay on topic. If your answer is long, break it up into paragraphs as appropriate to make it easier to read.
{context}
Question: {question}
Helpful answer: """

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
template=template_v2)

qa_chain = RetrievalQA.from_chain_type(llm,
retriever=vectordb.as_retriever(),
return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

print("Question 1:")
question = "What is the relationship between Pumpkin Book and Watermelon Book?"
result = qa_chain({"query": question})
print(result["result"])

print("Question 2:")
question = "How should Pumpkin Book be used?"
result = qa_chain({"query": question})
print(result["result"])
```

Question 1:
The relationship between Pumpkin Book and Watermelon Book is that Pumpkin Book is expressed with the content of Watermelon Book as the pre-knowledge. The purpose of Pumpkin Book is to parse the formulas in Watermelon Book that are more difficult to understand, and to supplement the specific derivation details to help readers better understand and learn the knowledge in the field of machine learning. Therefore, the best way to use it is to use Watermelon Book as the main line, and then refer to Pumpkin Book when you encounter a formula that you cannot derive or understand. The content of the Pumpkin Book is mainly to help readers who want to delve into the details of the formula derivation, providing more detailed explanations and supplements.
Question 2:
The Pumpkin Book should be used as a supplement to the Watermelon Book, mainly when you encounter a formula that you cannot derive.or understand the formula. For beginners of machine learning, it is recommended to go through the formulas in Chapter 1 and Chapter 2 of the Pumpkin Book first, and then come back to study them in depth after learning more deeply. The analysis and derivation of each formula are explained from the perspective of undergraduate mathematics foundation. The mathematical knowledge beyond the syllabus will be given in the appendix and references for interested students to continue to study in depth. If you can't find the formula you want to look up in the Pumpkin Book, or find an error, you can submit feedback in GitHub's Issues, and you will usually get a reply within 24 hours. In addition, the Pumpkin Book also provides supporting video tutorials and online reading addresses, as well as the latest PDF acquisition address. Finally, the content of the Pumpkin Book is licensed under the Creative Commons Attribution-Noncommercial-Share Alike 4.0 International License.

As you can see, the prompt of version A has a better effect on case ①, but the prompt of version B has a better effect on case ②. If we do not quantify the evaluation indicators and only use the relative evaluation, we cannot judge which prompt is better between version A and version B, and we need to find a prompt that performs better in all cases before further iteration; however, this is obviously very difficult and not conducive to our iterative optimization.

We can assign a score of 1 to 5 to each answer. For example, in the above case, we give answer ① of version A a score of 4 and answer ② a score of 2.Give answer ① of version B a score of 3, and answer ② a score of 5; then, the average score of version A is 3 points, and the average score of version B is 4 points, so version B is better than version A.

#### Criteria 2 Multi-dimensional evaluation

The large model is a typical generative model, that is, its answer is a sentence generated by the model. Generally speaking, the answer of the large model needs to be evaluated in multiple dimensions. For example, in the personal knowledge base question and answer project of this project, users generally ask questions about the content of the personal knowledge base. The answer of the model needs to meet the requirements of fully using the content of the personal knowledge base, the answer is consistent with the question, the answer is true and effective, and the answer statement is smooth. An excellent question and answer assistant should be able to answer users' questions well, ensure the correctness of the answer, and reflect full intelligence.

Therefore, we often need to start from multiple dimensions, design evaluation indicators for each dimension, and score each dimension to comprehensively evaluate the system performance. At the same time, it should be noted that multi-dimensional evaluation should be effectively combined with quantitative evaluation. For each dimension, the same dimension can be set or different dimensions can be set, and it should be fully combined with business reality.

For example, in this project, we can design the following evaluation dimensions:

① Knowledge search correctness. This dimension needs to check the intermediate results of the system searching for relevant knowledge fragments from the vector database, and evaluate whether the knowledge fragments found by the system can answer the question. This dimension is a 0-1 evaluation, that is, a score of 0 means that the knowledge found is correct.The fragment cannot give an answer, and a score of 1 means that the found knowledge fragment can give an answer.

② Answer consistency. This dimension evaluates whether the system's answer is targeted at the user's question, whether there is a deviation from the question, or whether the question is misunderstood. The dimension is also designed to be 0~1, 0 is completely off-topic, 1 is completely relevant, and the intermediate results can be taken at will.

③ Answer hallucination ratio. This dimension needs to integrate the system answer and the found knowledge fragments to evaluate whether the system's answer has hallucinations and how high the hallucination ratio is. This dimension is also designed to be 0~1, 0 is all model hallucinations, and 1 is no hallucinations.

④ Answer correctness. This dimension evaluates whether the system's answer is correct and whether it fully answers the user's question. It is one of the most core evaluation indicators of the system. This dimension can be scored arbitrarily between 0 and 1.

The above four dimensions are all centered around the correctness of knowledge and answers, and are highly relevant to the question; the next few dimensions will be centered around the anthropomorphism and grammatical correctness of the results generated by the large model, which are less relevant to the question:

⑤ Logic. This dimension evaluates whether the system answers logically coherently, whether there are conflicts and logical confusions. This dimension is evaluated on a 0-1 scale.

⑥ Fluency. This dimension evaluates whether the system answers smoothly and grammatically, and can be scored anywhere between 0 and 1.

⑦ Intelligence. This dimension evaluates whether the system answers are personified and intelligent, and whether it can fully confuse users with manual answers and intelligent answers. This dimension can be scored anywhere between 0 and 1.

For example, we evaluate the following answer:Answer for evaluation:

```python
print("question:")
question = "How should the Pumpkin Book be used?"
print(question)
print("Model answer:")
result = qa_chain({"query": question})
print(result["result"])
```

Question:
How should the Pumpkin Book be used?
Model answer:
The Pumpkin Book should be used as a supplement to the Watermelon Book, mainly for reference when encountering formulas that you cannot derive or understand. For beginners of machine learning, it is recommended to go through Chapters 1 and 2 of the Pumpkin Book first, and then come back to study it in depth after learning more deeply. The analysis and derivation of each formula are explained from the perspective of undergraduate mathematics foundation. The mathematical knowledge beyond the syllabus will be given in the appendix and references. Interested students can continue to study in depth. If the Pumpkin Book does not have the formula you want to check, or if you find an error, you can submit feedback in the Issues of GitHub, and you will usually get a reply within 24 hours. The ultimate goal is to help readers better understand and apply machine learning knowledge and become qualified science and engineering students.

The following are the knowledge fragments found by the system:

```python
print(result["source_documents"])
```

[Document(page_content='The main line is to refer to the Pumpkin Book when you encounter a formula that you cannot derive or understand;\n• For beginners of machine learning, it is strongly recommended not to study the formulas in Chapter 1 and Chapter 2 of the Watermelon Book in depth. Just go through them briefly. When you are a little bit drifting, you can come back to read them;\n• We strive to explain the analysis and derivation of each formula from the perspective of undergraduate mathematics foundation, so we usually provide the mathematical knowledge that is beyond the syllabus in the form of appendices and references. Interested students can continue to study in depth according to the materials we provide;\n• If the Pumpkin Book does not have the formula you want to refer to,\nor if you find an error in the Pumpkin Book,\nplease do not hesitate to go to our GitHub\nIssues (address: https://github.com/datawhalechina/pumpkin-book/issues) for feedback, and submit the formula number or errata information you want to add in the corresponding section. We usually respond within 24 hours. We will reply you within 24 hours. If you haven't received a reply within 24 hours, please contact us on WeChat (WeChat ID: at-Sm1les); Supporting video tutorial: https://www.bilibili.com/video/BV1Mh411e7VU', metadata={'author': '', 'creationDate': "D:20230303170709-00'00'", 'creator': 'LaTeX with hyperref', 'file_path': './data_base/knowledge_db/pumkin_book/ pumpkin_book.pdf', 'format': 'PDF 1.5', 'keywords': '', 'modDate': '', 'page': 1, 'producer': 'xdvipdfmx (20200315)', 'source': ' ./data_base/knowledge_db/pumkin_book/pumpkin_book.pdf', 'subject': '', 'title': '', 'total_pages': 196, 'trapped': ''}), Document(page_content='Online reading address: https://datawhalechina.github.io/pumpkin-book (only for the 1st edition)\nLatest version PDF acquisition addressURL: https://github.com/datawhalechina/pumpkin-book/releases\nEditorial Board\nEditors-in-Chief: Sm1les, archwalker, jbb0523\nEditorial Board: juxiao, Majingmin, MrBigFan, shanry, Ye980226\nCover Design: Concept - Sm1les, Creation - Linwang Maosheng\nAcknowledgments\nSpecial thanks to awyd234, feijuan, Ggmatch, Heitao5200, huaqing89, LongJH, LilRachel, LeoLRH, Nono17, spareribs, sunchaothu, StevenLzq for their contributions to Pumpkin Book in its earliest days. \nScan the QR code below and reply with the keyword "Pumpkin Book"\nto join the "Pumpkin Book Reader Exchange Group"\nCopyright Statement\nThis work is licensed under a Creative Commons Attribution-Noncommercial-Share Alike 4.0 International License. ', metadata={'author': '', 'creationDate': "D:20230303170709-00'00'", 'creator': 'LaTeX with hyperref', 'file_path': './data_base/knowledge_db/pumkin_book/pumpkin_book.pdf', 'format': 'PDF 1.5', 'keywords': '', 'modDate': '', 'page': 1, 'producer': 'xdvipdfmx (20200315)', 'source': './data_base/knowledge_db/pumkin_book/pumpkin_book.pdf', 'subject': '', 'title': '', 'total_pages': 196, 'trapped': ''}), Document(page_content='\x01本\x03:1.9.9\n发布日期:2023.03\n南 ｠ 书\nPUMPKIN\nB O O K\nDatawhale', metadata={'author': '', 'creationDate': "D:20230303170709-00'00'", 'creator': 'LaTeX with hyperref', 'file_path': './data_base/knowledge_db/pumkin_book/pumpkin_book.pdf', 'format': 'PDF 1.5', 'keywords': '', 'modDate': '', 'page': 0, 'producer': 'xdvipdfmx (20200315)', 'source': './data_base/knowledge_db/pumkin_book/pumpkin_book.pdf', 'subject': '', 'title': '', 'total_pages': 196, 'trapped': ''}), Document(page_content='Preface\n"Zhou Zhihua's "Machine Learning"\n (Xigua Book) is one of the classic introductory textbooks in the field of machine learning. In order to make as many readers as possible understand machine learning through Xigua Book, Therefore, the derivation details of some formulas are not detailed in the book, but this may be "not very friendly" to readers who want to delve into the details of the derivation of formulas. This book aims to analyze the formulas that are more difficult to understand in the watermelon book and to supplement the specific derivation details of some formulas. \n"\nReadHere, you may wonder why the previous paragraph is in quotation marks, because this is just our initial reverie. Later we learned that the real reason why Mr. Zhou omitted these derivation details is that he himself believes that "sophomore students with a solid foundation in science and engineering mathematics should have no difficulty with the derivation details in the watermelon book. The key points are all in the book, and the omitted details should be able to be supplemented in the mind or practiced." So... This pumpkin book can only be regarded as the notes taken by math scumbags like me when we were studying by ourselves. I hope it can help everyone become a qualified "sophomore student with a solid foundation in science and engineering mathematics." \nInstructions\n• All the contents of Pumpkin Book are expressed based on the contents of Watermelon Book as the prerequisite knowledge, so the best way to use Pumpkin Book is to use Watermelon Book\n as the main line, and then consult Pumpkin Book when you encounter a formula that you cannot derive or understand;', ​​metadata={'author': '', 'creationDate': "D:20230303170709-00'00'", 'creator': 'LaTeX with hyperref', 'file_path': './data_base/knowledge_db/pumkin_book/pumpkin_book.pdf', 'format': 'PDF 1.5', 'keywords': '', 'modDate': '', 'page': 1, 'producer': 'xdvipdfmx (20200315)', 'source': './data_base/knowledge_db/pumkin_book/pumpkin_book.pdf', 'subject': '', 'title': '', 'total_pages': 196, 'trapped': ''})]

We make corresponding evaluations:

① Knowledge search accuracy - 1

② Answer consistency - 0.8 (the question was answered, but the topic similar to "feedback" was off topic)

③ Answer illusion ratio - 1

④ Answer correctness - 0.8 (same reason as above)

⑤ Logic - 0.7 (the subsequent content is not logically coherent with the previous content)

⑥ Fluency - 0.6 (the final summary is wordy and invalid)

⑦ Intelligence - 0.5 (has a distinct AI style of answering)

Combining the above seven dimensions, we can comprehensively and comprehensively evaluate the performance of the system in each case. Taking into account the scores of all cases, we can evaluate the performance of the system in each dimension. If all dimensions are unified, we can also calculate the average score of all dimensions to evaluate the score of the system. We can also assign different importance to different dimensions.weights, and then calculate the weighted average of all dimensions to represent the system score.

However, we can see that the more comprehensive and specific the evaluation, the greater the difficulty and cost of the evaluation. Taking the above seven-dimensional evaluation as an example, we need to conduct seven evaluations for each case of each version of the system. If we have two versions of the system and there are 10 verification cases in the verification set, then each evaluation will require $ 10 \times 2 \times 7 = 140$ times; but as our system continues to improve and iterate, the verification set will expand rapidly. Generally speaking, a mature system verification set should be at least a few hundred in size, and there should be at least dozens of iterative improvement versions. Then the total number of our evaluations will reach tens of thousands, which will bring high manpower and time costs. Therefore, we need a method to automatically evaluate the model answers.

### 3.2 Simple automatic evaluation

One of the important reasons why large model evaluation is complicated is that the answers to the generated models are difficult to judge, that is, the evaluation of objective questions is simple, but the evaluation of subjective questions is difficult. Especially for some questions without standard answers, it is particularly difficult to achieve automatic evaluation. However, at the expense of a certain degree of evaluation accuracy, we can transform complex subjective questions without standard answers into questions with standard answers, and then achieve this through simple automatic evaluation. Here are two methods: constructing objective questions and calculating the similarity of standard answers.

#### Method 1: Constructing objective questionsThe evaluation of subjective questions is very difficult, but objective questions can be directly compared to see if the system answer is consistent with the standard answer, so as to achieve simple evaluation. We can construct some subjective questions into multiple or single choice objective questions to achieve simple evaluation. For example, for the question:

[Question and answer] Who is the author of the Pumpkin Book?

We can construct this subjective question into the following objective question:

[Multiple choice question] Who is the author of the Pumpkin Book? A Zhou Zhiming B Xie Wenrui C Qinzhou D Jia Binbin

The model is required to answer this objective question. We give the standard answer as BCD. The model's answer can be compared with the standard answer to achieve evaluation and scoring. Based on the above ideas, we can construct a prompt question template:

```python
prompt_template = '''
Please answer the following multiple choice questions:

Title: Who is the author of the Pumpkin Book?
Options: A Zhou Zhiming B Xie Wenrui C Qinzhou D Jia Binbin
Knowledge snippets you can refer to:
~~~
{}
~~~
Please return only the selected options
If you cannot make a choice, please return empty
'''
```

Of course, due to the instability of large models, even if we ask it to only give selected options, the system may return a lot of text, which explains in detail why the following options are selected. Therefore, we need to extract the options from the model answer. At the same time, we need to design a scoring strategy. In general, IWe can use the general scoring strategy for multiple-choice questions: 1 point for selecting all, 0.5 points for missing an answer, and no points for not selecting the wrong answer:

```python
def multi_select_score_v1(true_answer : str, generate_answer : str) -> float:
# true_anser : correct answer, str type, for example 'BCD'
# generate_answer : model generates answers, str type
true_answers = list(true_answer)
'''For ease of calculation, we assume that each question has only four options: A B C D'''
# Find the wrong answer set first
false_answers = [item for item in ['A', 'B', 'C', 'D'] if item not in true_answers]
# If the generated answer has an incorrect answer
for one_answer in false_answers:
if one_answer in generate_answer:
return 0
# Check if all correct answers are selected
if_correct = 0
for one_answer in true_answers:
if one_answer in generate_answer:
if_correct += 1
continue
if if_correct == 0:
# Do not select
return 0
elif if_correct == len(true_answers):
# Select all
return 1
else:
# Missing
return 0.5
```

Based on the above scoring function, we can test four answers:

① B C

② Except for A Zhou Zhihua, all others are the authors of Pumpkin Book

③ B C D should be selected

④ I don't know

```python
answer1 = 'B C'
answer2 = 'The author of Watermelon Book is A Zhou Zhihua'
answer3 = 'B C D should be selected'
answer4 = 'I don't know'
true_answer = 'BCD'
print("Answer 1 score:", multi_select_score_v1(true_answer, answer1))
print("Answer 2 score:", multi_select_score_v1(true_answer, answer2))
print("Answer 3 score:", multi_select_score_v1(true_answer, answer3))
print("Answer 4 score:", multi_select_score_v1(true_answer, answer4))
```

Answer 1 score: 0.5
Answer 2 score: 0
Answer 3 score: 1
Answer 4 score: 0

But we can see that we require the model not to make a choice when it cannot answer, rather than just making a random choice. However, in our scoring strategy, both wrong selection and no selection are 0 points, which actually encourages the model to hallucinate answers. Therefore, we can adjust the scoring strategy according to the situation and deduct one point for wrong selection:

```python
def multi_select_score_v2(true_answer : str, generate_answer : str) -> float:
# true_anser : correct answer, str type, for example 'BCD'
# generate_answer : model generates answers, str type
true_answers = list(true_answer)
'''For ease of calculation, we assume that each question has only four options: A B C D'''
# First find the wrong answer set
false_answers = [item for item in ['A', 'B', 'C', 'D'] if item not in true_answers]
# If the generated answer has an incorrect answer
for one_answer in false_answers:
if one_answer in generate_answer:
return -1
# Then determine whether all correct answers are selected
if_correct = 0
for one_answer in true_answers:
if one_answer in generate_answer:
if_correct += 1
continue
if if_correct == 0:
# Not selected
return 0
elif if_correct == len(true_answers):
# All selected
return 1
else:
# Missed selected
return 0.5
```

As above, we use the second version of the scoring function to score the four answers again:

```python
answer1 = 'B C'
answer2 = 'The author of the watermelon book is A Zhou Zhihua'
answer3 = 'You should choose B C D'
answer4 = 'I don't know'
true_answer = 'BCD'
print("Answer 1 score:", multi_select_score_v2(true_answer, answer1))
print("Answer 2 score:", multi_select_score_v2(true_answer,answer2))
print("Answer three score:", multi_select_score_v2(true_answer, answer3))
print("Answer four score:", multi_select_score_v2(true_answer, answer4))
```

Answer one score: 0.5
Answer two score: -1
Answer three score: 1
Answer four score: 0

As you can see, we have achieved fast, automatic and discriminative automatic evaluation. In this way, we only need to construct each verification case, and then each verification and iteration can be fully automated, thus achieving efficient verification.

However, not all cases can be constructed as objective questions. For some cases that cannot be constructed as objective questions or that constructing them as objective questions will cause the difficulty of the questions to drop sharply, we need to use the second method: calculating the answer similarity.

#### Method 2: Calculating the answer similarity

Evaluating the answer to the generated question is actually not a new problem in NLP. Whether it is machine translation, automatic abstraction and other tasks, it is actually necessary to evaluate the quality of the generated answer. NLP generally uses the method of manually constructing standard answers to generate questions and calculating the similarity between the answer and the standard answer to achieve automatic evaluation.

For example, for the question:

PumpkinWhat is the goal of the book?

We can first construct a standard answer manually:

Zhou Zhihua's "Machine Learning" (Xigua Book) is one of the classic introductory textbooks in the field of machine learning. In order to make as many readers as possible understand machine learning through Xigua Book, Zhou did not elaborate on the derivation details of some formulas in the book, but this may be "not very friendly" to those readers who want to delve into the details of formula derivation. This book aims to analyze the formulas that are more difficult to understand in Xigua Book, and to supplement the specific derivation details of some formulas.

Then calculate the similarity between the model answer and the standard answer. The more similar, the more correct we think the answer is.

There are many ways to calculate similarity. We can generally use BLEU to calculate similarity. The principle is detailed in: [Zhihu|Detailed Explanation of BLEU](https://zhuanlan.zhihu.com/p/223048748). For students who do not want to delve into the principles of the algorithm, it can be simply understood as topic similarity.

We can call the bleu scoring function in the nltk library to calculate:

```python
from nltk.translate.bleu_score import sentence_bleu
import jieba

def bleu_score(true_answer : str, generate_answer : str) -> float:
# true_anser : standard answer, str type
# generate_answer : model generates answers, str type
true_answers = list(jieba.cut(true_answer))
# print(true_answers)
generate_answers = list(jieba.cut(generate_answer))
# print(generate_answers)
bleu_score = sentence_bleu(true_answers, generate_answers)
return bleu_score
```

Test it:

```python
true_answer = 'Zhou Zhihua's "Machine Learning" (Xigua Book) is one of the classic introductory textbooks in the field of machine learning. In order to make as many readers as possible understand machine learning through Xigua Book, Zhou did not elaborate on the derivation details of some formulas in the book, but this may be "not very friendly" to readers who want to delve into the details of formula derivation. This book aims to explain the more difficult to understand formulas in Xigua Book.'

print("Answer 1:")
answer1 = 'Zhou Zhihua's "Machine Learning" (Xigua Book) is one of the classic introductory textbooks in the field of machine learning. In order to enable as many readers as possible to understand machine learning through Xigua Book, Zhou did not elaborate on the derivation details of some formulas in the book. However, this may be "not very friendly" to readers who want to delve into the details of formula derivation. This book aims to analyze the formulas that are more difficult to understand in Xigua Book, and to supplement the specific derivation details of some formulas. '
print(answer1)
score = bleu_score(true_answer, answer1)
print("Score:", score)
print("Answer 2:")
answer2 = 'This Pumpkin Book can only be regarded as the notes taken down by math losers like me when we were self-studying. I hope it can help everyone become a qualified "sophomore student with a solid foundation in science and engineering mathematics"'
print(answer2)
score = bleu_score(true_answer, answer2)
print("Score:", score)
```

Answer 1:
Mr. Zhou Zhihua's "Machine Learning" (Watermelon Book) is one of the classic introductory textbooks in the field of machine learning. Mr. ZhouIn order to make as many readers as possible understand machine learning through the Watermelon Book, the teacher did not elaborate on the derivation details of some formulas in the book, but this may be "not very friendly" to those readers who want to delve into the details of the formula derivation. This book aims to analyze the formulas that are more difficult to understand in the Watermelon Book, and to supplement the specific derivation details of some formulas.
Score: 1.2705543769116016e-231
Answer 2:
This Pumpkin Book can only be regarded as the notes taken down by math scumbags like me when I was self-studying. I hope it can help everyone become a qualified "sophomore student with a solid foundation in science and engineering mathematics"
Score: 1.1935398790363042e-231

It can be seen that the higher the consistency between the answer and the standard answer, the higher the evaluation score. Through this method, we also only need to construct a standard answer for each question in the validation set, and then we can achieve automatic and efficient evaluation.

However, this method also has several problems: ① The standard answer needs to be constructed manually. For some vertical fields, it may be difficult to construct a standard answer; ② There may be problems in evaluating by similarity. For example, if the generated answer is highly consistent with the standard answer but is exactly the opposite in several core places, resulting in a completely wrong answer, the BLEU score will still be high; ③ The flexibility of calculating the consistency with the standard answer is poor. If the model generates a better answer than the standard answer,, but the evaluation score will be reduced; ④ It is impossible to evaluate the intelligence and fluency of the answer. If the answer is a combination of keywords from various standard answers, we believe that such an answer is unusable and incomprehensible, but the bleu score will be higher.

Therefore, for business situations, sometimes we also need some advanced evaluation methods that do not require the construction of standard answers.

### 2.3 Use large models for evaluation

Using manual evaluation has high accuracy and comprehensiveness, but the labor cost and time cost are high; using automatic evaluation has low cost and fast evaluation speed, but there are problems of insufficient accuracy and incomplete evaluation. So, do we have a way to combine the advantages of both and achieve fast and comprehensive generation question evaluation?

Large models represented by GPT-4 provide us with a new method: using large models for evaluation. We can construct Prompt Engineering to let the large model act as an evaluator, thereby replacing the evaluator of manual evaluation; at the same time, the large model can give results similar to manual evaluation, so we can adopt the multi-dimensional quantitative evaluation method in manual evaluation to achieve fast and comprehensive evaluation.

For example, we can construct the following prompt engineering to let the big model score:

```python
prompt = '''
You are a model answer assessor.
Next, I will give you a question, the corresponding knowledge fragment, and the modelThe model answers the question based on the knowledge fragment.
Please evaluate the performance of the model answers in the following dimensions in turn and give scores respectively:

① Knowledge search correctness. Evaluate whether the knowledge fragment given by the system can answer the question. If the knowledge fragment cannot answer, the score is 0; if the knowledge fragment can answer, the score is 1.

② Answer consistency. Evaluate whether the system's answer is based on the user's question, whether there is a deviation from the question or a misunderstanding of the question, and the score is between 0 and 1, 0 is completely off-topic and 1 is completely relevant.

③ Answer hallucination ratio. This dimension requires the system answer to be combined with the knowledge fragment found to evaluate whether the system answer has hallucinations. The score is between 0 and 1, 0 is all model hallucinations, and 1 is no hallucinations.

④ Answer correctness. This dimension evaluates whether the system answer is correct and whether it fully answers the user's question. The score is between 0 and 1, 0 is completely incorrect and 1 is completely correct.

⑤ Logic. This dimension evaluates whether the system answer is logically coherent, whether there are conflicts and logical confusion. The scoring value is between 0 and 1, where 0 means the logic is completely chaotic and 1 means there is no logic problem at all.

⑥ Fluency. This dimension evaluates whether the system answers smoothly and grammatically. The scoring value is between 0 and 1, where 0 means the sentence is completely incoherent and 1 means the sentence is completely smooth without any grammatical problems.

⑦ Intelligence. This dimension evaluates whether the system answers are humanized and intelligent, and whether it can fully confuse users with manual answers and intelligent answersAnswer. The score is between 0 and 1, 0 is a very obvious model answer, and 1 is highly consistent with the human answer.

You should be a strict assessor and rarely give a high score.
User question:
~~~
{}
~~~
Answer to be evaluated:
~~~
{}
~~~
Given knowledge fragment:
~~~
{}
~~~
You should return me a directly parseable Python dictionary, the keys of the dictionary are the dimensions above, and the values ​​are the evaluation scores corresponding to each dimension.
Do not output anything else.
'''
```

We can actually test its effect:

```python
# Use the OpenAI native interface mentioned in Chapter 2

from openai import OpenAI

client = OpenAI(
# This is the default and can be omitted
api_key=os.environ.get("OPENAI_API_KEY"),
)

def gen_gpt_messages(prompt):
'''
Construct GPT model request parameter messages

Request parameters:
prompt: corresponding user prompt word
'''messages = [{"role": "user", "content": prompt}]
return messages

def get_completion(prompt, model="gpt-3.5-turbo", temperature = 0):
'''
Get the result of calling the GPT model

Request parameters:
prompt: the corresponding prompt word
model: the model to be called, the default is gpt-3.5-turbo, and other models such as gpt-4 can also be selected as needed
temperature: the temperature coefficient of the model output, which controls the degree of randomness of the output, and the value range is 0~2. The lower the temperature coefficient, the more consistent the output content.
'''
response = client.chat.completions.create(
model=model,
messages=gen_gpt_messages(prompt),
temperature=temperature,
)
if len(response.choices) > 0:
return response.choices[0].message.content
return "generate answer error"

question = "How should the pumpkin book be used?"

result = qa_chain({"query": question})
answer = result["result"]
knowledge = result["source_documents"]

response = get_completion(prompt.format(question, answer, knowledge))
response
```

'{\n "Knowledge search correctness": 1,\n "Answer consistency": 0.9,\n "Answer illusion ratio": 0.9,\n "Answer correctness": 0.9,\n "Logic": 0.9,\n "Fluency": 0.9,\n "Intelligence": 0.8\n}'

But note that there are still problems with using large models for evaluation:

① Our goal is to iteratively improve Prompt to improve the performance of the large model, so the evaluation large model we choose needs to have better performance than the large model base we use. For example, the most powerful large model is still GPT-4. It is recommended to use GPT-4 for evaluation, which has the best effect.

② Large models have powerful capabilities, but they also have their limits. If the questions and answers are too complex, the knowledge fragments are too long, or too many evaluation dimensions are required, even GPT-4 will have incorrect evaluations, incorrect formats, and inability to understand instructions. For these situations, we recommend considering the following solutions to improve the performance of large models:

1. Improve Prompt Engineering. In a similar way to the improvement of Prompt Engineering in the system itself, iteratively optimize and evaluate Prompt Engineering, especially pay attention to whether the basic principles and core recommendations of Prompt Engineering are followed;

2. Split the evaluation dimensions. If there are too many evaluation dimensions, the model may have an incorrect format, resulting in an unparseable return. You can consider splitting the multiple dimensions to be evaluated, calling the large model once for each dimension for evaluation, and finally obtaining a unified result;

3. Merge the evaluation dimensions. If the evaluation dimension is too detailed, the model may not be able to understand it correctly and the evaluation may be incorrect. You can consider merging multiple dimensions to be evaluated, for example, merging logic, fluency, and intelligence into intelligence, etc.;

4.Provide detailed evaluation specifications. Without evaluation specifications, it is difficult for the model to give ideal evaluation results. You can consider giving detailed and specific evaluation specifications to improve the evaluation ability of the model;

5. Provide a small number of examples. The model may find it difficult to understand the evaluation specifications. At this time, you can give a small number of evaluation examples for the model to refer to for correct evaluation.

### 2.4 Mixed evaluation

In fact, the above evaluation methods are not isolated or opposite. Compared with using a certain evaluation method independently, we recommend mixing multiple evaluation methods and selecting the appropriate evaluation method for each dimension, taking into account the comprehensiveness, accuracy and efficiency of the evaluation.

For example, for the personal knowledge base assistant of this project, we can design the following mixed evaluation method:

1. Objective correctness. Objective correctness means that for some questions with fixed correct answers, the model can give correct answers. We can select some cases and use the method of constructing objective questions to evaluate the model and evaluate its objective correctness.

2. Subjective correctness. Subjective correctness means that for subjective questions without fixed correct answers, the model can give correct and comprehensive answers. We can select some cases and use the large model evaluation method to evaluate whether the model's answer is correct.

3. Intelligence. Intelligence refers to whether the model's answer is sufficiently humanized. Since intelligence is weakly correlated with the question itself, and strongly correlated with the model and prompt, and the model's ability to judge intelligence is relatively weak, we can manually evaluate its intelligence by sampling a small number of cases..

4. Knowledge search correctness. Knowledge search correctness refers to whether the knowledge fragments retrieved from the knowledge base are correct and sufficient to answer the question for a specific question. It is recommended to use a large model to evaluate the knowledge search correctness, that is, the model is required to determine whether the given knowledge fragment is sufficient to answer the question. At the same time, the evaluation results of this dimension combined with subjective correctness can calculate the hallucination situation, that is, if the subjective answer is correct but the knowledge search is incorrect, it means that the model hallucination has occurred.

Using the above evaluation method, based on the obtained validation set examples, a reasonable evaluation of the project can be made. Due to time and manpower limitations, it will not be shown in detail here.

**Note: The source code corresponding to this article is in the [Github open source project LLM Universe](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C5%20%E7%B3%BB%E7%BB%9F%E8%AF%84%E4%BC%B0%E4%B8%8E%E4%BC%98%E5%8C%96/1.%E5%A6%82%E4%BD%95%E8%AF%84%E4%BC%B0%20LLM%20%E5%BA%94%E7%94%A8.ipynb), welcome to download and run, welcome to give us a star~**