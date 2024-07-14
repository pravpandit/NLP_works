# Big Model Handbook for Developers - LLM Cookbook

## Project Introduction

This project is a big model handbook for developers, targeting the actual needs of domestic developers, focusing on the comprehensive entry practice of LLM. This project is based on the content of the Big Model series of courses by Andrew Ng, and screens, translates, reproduces and optimizes the original course content, covering the entire process from Prompt Engineering to RAG development and model fine-tuning, and guides domestic developers on how to learn and get started with LLM-related projects in the most suitable way for domestic learners.

According to the characteristics of different contents, we have translated and reproduced a total of 11 big model courses of Andrew Ng, and graded and sorted different courses based on the actual situation of domestic learners. Beginners can first systematically study our required courses, master the basic skills and concepts required for all directions of getting started with LLM, and then selectively study our elective courses to continuously explore and learn in the direction of their interest.

If there is a large model course by Andrew Ng that you really like but we haven't reproduced yet, we welcome every developer to refer to the format and writing of our existing courses to reproduce the course and submit a PR. After the PR is reviewed and approved, we will grade and merge the courses according to the course content. Every developer is welcome to contribute! 

**Online reading address: [LLM introductory course for developers-online reading](https://datawhalechina.github.io/prompt-engineering-for-developers/)**

**PDF download address: [LLM introductory tutorial for developers-PDF](https://github.com/datawhalechina/prompt-engineering-for-developers/releases)**

**English original address: [Andrew Ng's series of courses on large models](https://learn.deeplearning.ai)**

## Project significance

LLM is gradually changing people's lives, and for developers, how to quickly and conveniently develop some applications with stronger capabilities and integrated LLM based on the API provided by LLM to conveniently implement some more novel and practical capabilities is an important ability that needs to be learned urgently.

The large model series of tutorials launched by Andrew Ng in cooperation with OpenAI, starting from the basic skills of developers in the era of large models, introduces in an easy-to-understand way how to quickly develop applications that combine the powerful capabilities of large models based on large model APIs and LangChain architecture. Among them, the "Prompt Engineering for Developers" tutorial is aimed at the development of entry-level LLM.The tutorial "LangChain for LLM Application Development" introduces how to develop applications based on the classic large model open source framework LangChain, and introduces how to develop applications with practical functions and comprehensive capabilities based on the LangChain framework. The tutorial "LangChain Chat With Your Data" further introduces how to use the LangChain architecture to develop personalized large model applications in combination with personal private data. The tutorials "Building Generative AI Applications with Gradio" and "Evaluating and Debugging Generative AI" respectively introduce two practical tools, Gradio and W&B, to guide developers on how to combineThese two tools are used to build and evaluate generative AI applications.

The above tutorials are very suitable for developers to learn and start building applications based on LLM. Therefore, we translated this series of courses into Chinese and reproduced its sample code. We also added Chinese subtitles to one of the videos to support domestic Chinese learners to use it directly to help Chinese learners better learn LLM development; we also implemented a Chinese prompt with roughly equivalent effects to support learners to experience the learning and use of LLM in the Chinese context and compare and master the prompt design and LLM development in a multi-language context. In the future, we will also add more advanced prompt techniques to enrich the content of this course and help developers master more and more clever prompt skills.

## Project audience

All developers who have basic Python skills and want to get started with LLM.

## Project Highlights

Tutorials such as "ChatGPT Prompt Engineering for Developers" and "Building Systems with the ChatGPT API" are official tutorials jointly launched by Professor Andrew Ng and OpenAI. They will become important introductory tutorials for LLM in the foreseeable future. However, they are currently only available in English and have limited access in China. It is of great significance to create a Chinese version of the tutorial that can be accessed smoothly in China.; At the same time, GPT has different understanding abilities for Chinese and English. After multiple comparisons and experiments, this tutorial determined the Chinese prompt with roughly equivalent effects, supporting learners to study how to improve ChatGPT's understanding and generation abilities in the Chinese context.

## Learning Guide

This tutorial is suitable for all developers who have basic Python skills and want to get started with LLM.

If you want to start learning this tutorial, you need to have in advance:

1. At least one LLM API (preferably OpenAI, if it is other APIs, you may need to refer to [other tutorials](https://github.com/datawhalechina/llm-universe) to modify the API call code)

2. Ability to use Python Jupyter Notebook

This tutorial includes a total of 11 courses, divided into two categories: required and elective. Compulsory courses are the most suitable courses for beginners to learn and get started with LLM. They include the basic skills and concepts that need to be mastered in all directions of LLM. We have also made online reading and PDF versions suitable for reading compulsory courses. When studying compulsory courses, we recommend that learners study in the order we list them. Elective courses are an extension of compulsory courses, including RAG development, model fine-tuning, and model evaluation.It covers multiple aspects, suitable for learners to choose the direction and courses of their interest after mastering the required courses.

Required courses include:

1. Prompt Engineering for developers. Based on Andrew Ng's "ChatGPT Prompt Engineering for Developers" course, it is aimed at developers who are entering LLM. It introduces in an easy-to-understand way how to construct prompts for developers and implement various common functions including summary, inference, conversion, etc. based on the API provided by OpenAI. It is the first step to enter LLM development.
2. Build a question-and-answer system based on ChatGPT. Based on Andrew Ng's "Building Systems with the ChatGPT API" course, it guides developers on how to develop a complete and comprehensive intelligent question-and-answer system based on the API provided by ChatGPT. Through code practice, the whole process of developing a question-and-answer system based on ChatGPT is realized, and a new paradigm based on large model development is introduced, which is the practical basis for large model development.
3. Use LangChain to develop applications. Based on Andrew Ng's "LangChain for LLM Application Development" course, LangChain is expanded.In-depth introduction to help learners understand how to use LangChain and develop complete and powerful applications based on LangChain.
4. Use LangChain to access personal data. Based on Andrew Ng's course "LangChain Chat with Your Data", it further expands the personal data access capabilities provided by LangChain and guides developers on how to use LangChain to develop large-scale model applications that can access users' personal data and provide personalized services.

Elective courses include (Note: There is no online reading version for elective courses):

1. Use Gradio to build generative AI applications. Based on Andrew Ng's course "Building Generative AI Applications with Gradio", it guides developers on how to use Gradio to quickly and efficiently build user interfaces for generative AI through Python interface programs.
2. Evaluate and improve generative AI. Based on Andrew Ng's course "Evaluating and Debugging Generative AI", combined with wandb, it provides a set of systematic methods and tools to help developers effectively track and debug generative AI models.
3. Fine-tune large language models. Based on Andrew Ng's course "Building Generative AI Applications with Gradio", it guides developers on how to use Gradio to quickly and efficiently build user interfaces for generative AI through Python interface programs.Based on the course "Finetuning Large Language Model" by Andrew Ng, combined with the lamini framework, it tells how to fine-tune the open source large language model based on personal data locally conveniently and efficiently.
4. Large Model and Semantic Retrieval. Based on the course "Large Language Models with Semantic Search" by Andrew Ng, it describes a variety of advanced retrieval techniques for more accurate and efficient retrieval enhancement LLM generation effects.
5. Advanced Retrieval Based on Chroma. Based on the course "Advanced Retrieval for AI with Chroma" by Andrew Ng, it aims to introduce advanced retrieval technology based on Chroma and improve the accuracy of retrieval results.
6. Building and Evaluating Advanced RAG Applications. Based on the course "Building and Evaluating Advanced RAG Applications" by Andrew Ng, it introduces the key technologies and evaluation frameworks required to build and implement high-quality RAG systems.
7. Functions, Tools and Agents of LangChain. Based on the course "Functions, Tools and Agents with LangChain" by Andrew NgThis article introduces how to build an Agent based on LangChain's new syntax.
8. Prompt advanced tips. Original content, in progress.

Other materials include:

**Bilingual subtitles video address: [Prompt Engineering course professional translation version of Andrew Ng x OpenAI](https://www.bilibili.com/video/BV1Bo4y1A7FU/?share_source=copy_web)**

**Chinese and English subtitles download: [Unofficial Chinese and English subtitles of "ChatGPT Prompt Engineering"](https://github.com/GitHubDaily/ChatGPT-Prompt-Engineering-for-Developers-in-Chinese)**

**Video explanation: [Prompt Engineering for developers (Digital Nomad Conference)](https://www.bilibili.com/video/BV1PN4y1k7y2/?spm_id_from=333.999.0.0)**

## Acknowledgements

**Core Contributors**

- [Zou Yuheng-Project Leader](https://github.com/logan-zou) (Datawhale member-graduate student at University of International Business and Economics)
- [Changqin-Project Initiator](https://yam.gift/) (Content Creator-Datawhale Member-AI Algorithm Engineer)
- [Yulin-Project Initiator](https://github.com/Sophia-Huang) (Content Creator-Datawhale Member)
- [Xu Hu-Tutorial Compiler](https://github.com/xuhu0115) (Content Creator-Datawhale Member)
- [Liu Weihong-Tutorial Compiler](https://github.com/Weihong-Liu) (Content Creator-Part-time Graduate Student at Jiangnan University)
- [Joye-Tutorial Compiler](https://Joyenjoye.com) (Content Creator-Data Scientist)
- [Gao Liye](https://github.com/0-yy-0) (Content Creator-DataWhale Member-Algorithm Engineer)
- [Deng Yuwen](https://github.com/GKDGKD) (Content creator - Datawhale member)
- [Hun Xi](https://github.com/wisdom-pan) (Content creator - front-end engineer)
- [Song Zhixue](https://github.com/KMnO4-zx) (Content creator-Datawhale member)
- [Han Yikun](https://github.com/YikunHan42) (Content creator-Datawhale member)
- [Chen Yihan](https://github.com/6forwater29) (Content creator-Datawhale intended member-AI enthusiast)
- [Zhongtai](https://github.com/ztgg0228) (Content creator-Datawhale member)
- [Wan Lixing](https://github.com/leason-wan) (Content creator-Video translator)
- [Wang Yiming](https://github.com/Bald0Wang) (Content creator-Datawhale member)
- [Zeng Haolong](https://yetingyun.blog.csdn.net) (Content creator-Datawhale intended member-JLU AI graduate student)
- [Xiaofan](https://github.com/xinqi-fan) (Content creator)
- [Sun Hanyu](https://github.com/sunhanyu714]) (Content creator-Algorithm Quantitative Deployment Engineer)
- [Zhang Yinhan](https://github.com/YinHan-Zhang) (Content Creator-Datawhale Member)
- [Zuo Chunsheng](https://github.com/LinChentang) (Content Creator-Datawhale Member)
- [Zhang Jin](https://github.com/Jin-Zhang-Yaoguang) (Content Creator-Datawhale Member)
- [Li Jiaojiao](https://github.com/Aphasia0515) (Content Creator-Datawhale Member)
- [Deng Kaijun](https://github.com/Kedreamix) (Content Creator-Datawhale Member)
- [Fan Zhiyuan](https://github.com/Zhiyuan-Fan) (Content Creator-Datawhale Member)
- [Zhou Jinglin](https://github.com/Beyondzjl) (Content Creator-Datawhale Member)
- [Zhu Shiji](https://github.com/very-very-very) (Content creator-algorithm engineer)
- [Zhang Yixin](https://github.com/YixinZ-NUS) (Content creator-IT enthusiast)
- Sarai (Content creator-AI application enthusiast)

**Others**

1. Special thanks to [@Sm1les](https://github.com/Sm1les) and [@LSGOMYP](https://github.com/LSGOMYP) for their help and support for this project;

2. Thanks to [GithubDaily](https://github.com/GitHubDaily) for providing bilingual subtitles;

3. If you have any ideas, please contact us DataWhale also welcomes everyone to raise issues;

4. Special thanks to the following students who contributed to the tutorial!

<a href="https://github.com/datawhalechina/prompt-engineering-for-developers/graphs/contributors">
<img src="https://contrib.rocks/image?repo=datawhalechina/prompt-engineering-for-developers" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=datawhalechina/llm-cookbook&type=Date)](https://star-history.com/#datawhalechina/llm-cookbook&Date)

## Follow us

<div align=center>
<p>Scan the QR code below to follow the official account: Datawhale</p>
<img src="https://github.com/datawhalechina/llm-cookbook/blob/main/figures/qrcode.jpeg" width = "180" height = "180">
</div>
Datawhale is an open source organization focusing on data science and AI. It brings together outstanding learners from many fields, colleges and well-known companies, and gathers a group of team members with open source and exploratory spirits. WeChat search official accountDatawhale can join us.

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.