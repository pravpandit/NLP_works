# Hands-on learning of large model application development

<div align=center>
<img src="figures/C0-0-logo.png" width = "1000">
</div>

## Project Introduction

This project is a tutorial on large model application development for novice developers. It aims to complete the key points of large model development through a course based on Alibaba Cloud Server and combined with the personal knowledge base assistant project. The main contents include:

1. Introduction to large models, what is a large model, what are the characteristics of a large model, what is LangChain, how to develop an LLM application, a simple introduction for novice developers;

2. How to call the large model API. This section introduces various calling methods of APIs of well-known large model products at home and abroad, including calling native APIs, encapsulating as LangChain LLM, encapsulating as Fastapi, etc. At the same time, multiple large model APIs including Baidu Wenxin, iFlytek Spark, Zhipu AI, etc. are encapsulated in a unified form;

3. Knowledge base construction, loading and processing of different types of knowledge base documents, and construction of vector databases;

4. Build RAG applications, including LLM is connected to LangChain to build a retrieval question-answering chain, and Streamlit is used for application deployment
5. Verification iteration, how to implement verification iteration in large model development, what are the general evaluation methods?;

This project mainly includes three parts:

1. Introduction to LLM development. The simplified version of V1 is designed to help beginners get started with LLM development as quickly and conveniently as possible, understand the general process of LLM development, and build a simple Demo.

2. LLM development skills. More advanced skills for LLM development, including but not limited to: prompt engineering, processing of multiple types of source data, optimized retrieval, recall sorting, agent framework, etc.

3. LLM application examples. Introduce some successful open source cases, analyze the ideas, core ideas, and implementation frameworks of these application examples from the perspective of this course, and help beginners understand what kind of applications they can develop through LLM.

At present, the first part has been completed, and everyone is welcome to read and learn; the second and third parts are being created.

**Directory structure description:**

requirements.txt: installation dependencies in the official environment
notebook: Notebook source code file
docs: Markdown document file
figures: pictures
data_base: the knowledge base source file used

## Project significance

LLM is gradually becoming a new revolutionary force in the information world. Through its powerful natural language understanding and natural language generation capabilities, it has opened up new opportunities for developers.It provides developers with new and more powerful application development options. With the explosive opening of LLM API services at home and abroad, how to quickly and conveniently develop applications with stronger capabilities and integrated LLM based on LLM API has begun to become an important skill for developers.

At present, there are many introductions to LLM and scattered LLM development skills courses, but the quality is uneven and not well integrated. Developers need to search for a large number of tutorials and read a lot of content that is not very relevant and necessary to initially master the necessary skills for large model development. The learning efficiency is low and the learning threshold is also high.

Starting from practice, this project combines the most common and general personal knowledge base assistant project to gradually disassemble the general process and steps of LLM development in a simple and easy-to-understand way, aiming to help novices without algorithm foundation complete the basic introduction of large model development through a course. At the same time, we will also add advanced skills of RAG development and interpretation of some successful LLM application cases to help readers who have completed the first part of the study to further master more advanced RAG development skills and be able to develop their own and fun applications by learning from existing successful projects.

## Project Audience

All developers who have basic Python skills and want to master LLM application development skills.

**This project does not require learners to have a foundation in artificial intelligence or algorithms. You only need to master basic Python syntax and basic Pythonthon development skills. **

Considering the environment construction problem, this project provides a free way for students to get Alibaba Cloud Server. Students can get Alibaba Cloud Server for free and complete the study of this course through Alibaba Cloud Server. This project also provides a guide to setting up the environment for personal computers and non-Alibaba Cloud servers. This project has basically no requirements for local hardware and does not require a GPU environment. Both personal computers and servers can be used for learning.

**Note: This project mainly uses the APIs provided by major model manufacturers for application development. If you want to learn to deploy and apply local open source LLM, you are welcome to learn [Self LLM ｜ Open Source Large Model Eating Guide] (https://github.com/datawhalechina/self-llm), which is also produced by Datawhale. This project will teach you how to quickly pass the full link of open source LLM deployment and fine-tuning! **

**Note: Considering the difficulty of learning, this project is mainly for beginners and introduces how to use LLM to build applications. If you want to further study the theoretical basis of LLM and further understand and apply LLM based on the theory, you are welcome to learn [So Large LM | Big Model Foundation](https://github.com/datawhalechina/so-large-lm), this project will provide you with comprehensive and in-depth LLM theoretical knowledge and practical methods! **

## Project highlights

1. Fully oriented to practice, hands-on learning of large model development. Compared with other similar tutorials that start from theory and have a large gap with practice, this tutorial is based on the universal personal knowledge base assistant project, integrating the universal large model development concept into project practice, helping learners to master large model development skills by building personal projects.

2. Starting from scratch, a comprehensive and short large model tutorial. This project is aimed at the personal knowledge base assistant project, and has carried out project-led reconstruction of the relevant large model development theories, concepts and basic skills, deleting the underlying principles and algorithm details that do not need to be understood, and covering all the core skills of large model development. The overall duration of the tutorial is within a few hours, but after completing this tutorial, you can master all the core skills of basic large model development.

3. Both unified and extensible. This project has unified the encapsulation of major LLM APIs at home and abroad, such as GPT, Baidu Wenxin, iFlytek Spark, and Zhipu GLM, and supports one-click calling of different LLMs, helping developers to focus more on learning applications and optimizing the model itself, without spending time on tedious calling details; at the same time, this tutorial is planned to be launched on [Qiming Planet | AIGC Co-creation Community Platform](https://1aigc.cn/), supporting learners to customize projects to add extended content to this tutorial, with full extensibility.

## Online reading address

[LLM Universe V2](https://datawhalechina.github.io/llm-universe/#/)

## Content outline

### Part I Introduction to LLM development

Leader: Zou Yuheng

1. [LLM introduction](./C1/) @Gao Liye

1. [LLM theory introduction](./C1/1.Large language model%20LLM%20Theoretical introduction.md)

2. [What is RAG, RAG's core advantages](./C1/2.Retrieval enhancement generation%20RAG%20Introduction.md)

3. [What is LangChain](./C1/3.LangChain%20Introduction.md)

4. [Overall process of developing LLM applications](./C1/4.Overall process of developing%20LLM%20 applications.md)

5. [Basic use of Alibaba Cloud Server](./C1/5.Basic use of Alibaba Cloud Server.md)
6. [Basic use of GitHub Codespaces (optional)](./C1/6.Basic use of GitHub%20Codespaces%20 (optional).md)
7. [Environment configuration](./C1/7.Environment configuration.md)
2.[Using LLM API to develop applications](./C2/) @毛雨
1. [Basic concepts](./C2/1.%20Basic concepts.md)
2. [Using LLM API](./C2/2.%20Using%20LLM%20API.md)
- ChatGPT
- Wenxinyiyan
- iFlytek Spark
- Zhipu GLM
3. [Prompt Engineering](./C2/3.%20Prompt%20Engineering.md)
3. [Building a knowledge base](./C3/) @娄天奥
1. [Introduction to word vectors and vector knowledge base](./C3/1.Introduction to word vectors and vector knowledge base.md)
2. [Using Embedding API](./C3/2.Using%20Embedding%20API.md)
3. [Data processing: reading, cleaning and slicing](./C3/3.Data processing.md)
4. [Build and use vector database](./C3/4.Build and use vector database.md)
4. [Build RAG application](./C4/) @徐虎
1. [Connect LLM to LangChain](./C4/1.LLM%20Access%20LangChain.md)
- ChatGPT
- Wenxinyiyan
- iFlytek Spark
- Zhipu GLM
2. [Building a search question and answer chain based on LangChain](./C4/2.Building a search question and answer chain.md)
3. [Deploying a knowledge base assistant based on Streamlit](./C4/3.Deploying a knowledge base assistant.md)
5. [System evaluation and optimization](./C5/) @Zou Yuheng
1. [How to evaluate LLM applications](./C5/1.How to evaluate%20LLM%20 applications.md)
2. [Evaluate and optimize the generation part](./C5/2.Evaluate and optimize the generation part.md)
3. [Evaluate and optimize the search part](./C5/3.Evaluate and optimize the search part.md)

### Part 2 Advanced RAG skills (under creation)

Person in charge: Gao Liye

1. Background
1. Architecture Overview
2. Existing Problems
3. Solutions
2. Data Processing
1. Multi-Type Document Processing
2. Block Optimization
3. Vector Model Selection
4. Fine-tuning Vector Model (Advanced)
3. Index Level
1. Index Structure
2. MixedCombined search
3. Hypothetical questions
4. Search phase
1. Query filtering
2. Align query and document
3. Aligned search and LLM
5. Generation phase
1. Post-processing
2. Fine-tuning LLM (advanced)
3. Reference citation
6. Enhancement phase
1. Context enhancement
2. Enhancement process
7. RAG engineering evaluation

### Part III Interpretation of open source LLM application

Leader: Xu Hu

1. Interpretation of ChatWithDatawhale——Personal knowledge base assistant

2. Interpretation of Tianji——Human relations and worldly affairs model

## Acknowledgements

**Core contributors**

- [Zou Yuheng-Project leader](https://github.com/logan-zou) (Datawhale member-graduate student of University of International Business and Economics)

- [Gao Liye-Leader of the second part](https://github.com/0-yy-0) (DataWhale member-algorithm engineer)
- [Xu Hu - Head of the third part](https://github.com/xuhu0115) (Datawhale member - algorithm engineer)

**Main contributor**

- [Mao Yu - content creator](https://github.com/Myoungs ) (Backend Development Engineer)
- [Lou Tianao-Content Creator](https://github.com/lta155) (Datawhale Jingying Teaching Assistant-Postgraduate of University of Chinese Academy of Sciences)
- [Cui Tengsong-Project Supporter](https://github.com/2951121599) (Datawhale Member-Co-sponsor of Qixiang Planet)
- [June-Project Supporter](https://github.com/JuneYaooo) (Datawhale Member-Co-sponsor of Qixiang Planet)

**Other**

1. Special thanks to [@Sm1les](https://github.com/Sm1les) and [@LSGOMYP](https://github.com/LSGOMYP) for their help and support for this project;

2. Special thanks to [Qixiang Planet | AIGC Co-creation Community Platform](https://1aigc.cn/) for its support, welcome to follow;

3. If you have any ideas, please contact us DataWhale also welcomes everyone to raise issues;

4. Special thanks to the following students who contributed to the tutorial!

<a href="https://github.com/datawhalechina/llm-universe/graphs/contributors"> <img src="https://contrib.rocks/image?repo=datawhalechina/llm-universe" /> </a> Made with [contrib.rocks](https://contrib .rocks). ## Star History [![Star History Chart](https://api.star-history.com/svg?repos=datawhalechina/llm-universe&type=Date)](https://star-history. com/#datawhalechina/llm-universe&Date)