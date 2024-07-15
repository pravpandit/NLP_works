# Basics of Large Models

Updated
>[Datawhale Open Source Large Model Introduction Course - Section 1 - Advancing AI: Panorama of Large Model Technology](https://www.bilibili.com/video/BV14x4y1x7bP/?spm_id_from=333.999.0.0&vd_source=4d086b5e84a56b9d46078e927713ffb0)
>
> [Text Tutorial: Llama Open Source Family: From Llama-1 to Llama-3](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch14.md)
> 
> [Video tutorial: Llama open source family: from Llama-1 to Llama-3](https://www.bilibili.com/video/BV1Xi421C7Ca/?share_source=copy_web&vd_source=df1bd9526052993d540dbd5f7938501f)

## Project audience

1. Researchers and practitioners in the fields of artificial intelligence, natural language processing, and machine learning: This project aims to provide researchers and practitioners with knowledge andtechnology, helping them to have a deeper understanding of the latest developments and research progress in the current field.
2. People in academia and industry who are interested in large-scale language models: The project content covers all aspects of large-scale language models, from data preparation, model building to training and evaluation, as well as security, privacy and environmental impact. This helps to broaden the audience's knowledge in this field and deepen their understanding of large-scale language models.
3. People who want to participate in large-scale language model open source projects: This project provides code contributions and theoretical knowledge to lower the threshold for audiences to learn large-scale pre-training.
4. Other large-scale language model-related industry personnel: The project content also involves legal and ethical considerations of large-scale language models, such as copyright law, fair use, fairness, etc., which helps related industry practitioners better understand the relevant issues of large-scale language models.

## Project Introduction

&emsp;&emsp;This project aims to serve as a tutorial for large-scale pre-trained language models, providing open source knowledge from data preparation, model building, training strategies to model evaluation and improvement, as well as the model's security, privacy, environment and legal ethics.

&emsp;&emsp;The project will be based on [Stanford University Large-Scale Language Models Course](https://stanford-cs324.github.io/winter2022/) and [Li Hongyi's Generative AI Course](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php), combined with the supplements and improvements from open source contributors, as well as the timely update of cutting-edge large model knowledge, to provide readers with a relatively comprehensive and in-depth theoretical knowledge and practical methods. Through systematic explanations of model construction, training, evaluation and improvement, as well as actual code, we hope to establish a project with broad reference value.

&emsp;&emsp;Our project team members will be responsible for the content sorting and writing of each chapter, and it is expected to complete the initial version content within three months. Subsequently, we will continue to update and optimize the content based on community contributions and feedback to ensure the continuous development of the project and the timeliness of knowledge. We look forward to contributing a valuable resource to the field of large language model research through this project, and promoting the rapid development and widespread application of related technologies.

## Project Significance

In today's era, the field of natural language processing (NLP) and other branches of artificial intelligence (AI) have ushered in a revolutionary change, and the core driving force of this change is the emergence and development of large models (LLMs). These models not only form the basis of the most advanced systems in many tasks, but have also demonstrated unprecedented powerful capabilities and application potential in many industries such as healthcare, finance, and education.

As the influence of these large models at the social level continues to expand, they have become the focus of public discussion and have stimulated the development of artificial intelligence in all walks of life.In-depth thinking and broad interest in trends and potential impacts. However, despite the attention paid to this field, the quality of related discussions and articles is uneven, lacking in systematization and depth, which is not conducive to the public's true understanding of the complexity of this technology.

Based on this situation, this tutorial is written to fill this gap and provide a set of large model tutorials that are not only easy to understand but also rich in theory: Through this tutorial, we hope that the general public will not only have a deep understanding of the principles and working mechanisms of large models, but also master their key technologies and methods in practical applications, so as to continue to explore and innovate in this field.

Especially for beginners in the field of natural language processing, facing various emerging technologies and knowledge with large models as the core, being able to quickly get started and effectively learn is the key to entering this field. The current existing natural language processing tutorials are still insufficient in covering the content of large models, which undoubtedly increases the learning difficulty for beginners. Therefore, this tutorial starts from the most basic concepts and gradually goes deeper, striving to fully cover the core knowledge and technical points of large models, so that readers can have a deep understanding and mastery from theory to practice.

> For the practical part, you are welcome to learn the [self-llm open source course](https://github.com/datawhalechina/self-llm) also produced by Datawhale. This course provides a comprehensive practical guide to simplify the use of open source large models through the AutoDL platform.Deployment, use and application process. This enables students and researchers to master skills such as environment configuration, local deployment and model fine-tuning more efficiently. After learning the basics of large models and large model deployment, Datawhale's large model development course [llm-universe](https://github.com/datawhalechina/llm-universe) aims to help beginners get started with LLM development as quickly and conveniently as possible, understand the general process of LLM development, and build a simple Demo.

**We firmly believe that through such a comprehensive and in-depth set of learning materials, we can greatly promote people's interest and understanding in the field of natural language processing and artificial intelligence, and further promote the healthy development and technological innovation in this field. **

## Project Highlights

1. Timeliness of the project: Currently, large models are developing rapidly, and society and learners lack a more comprehensive and systematic large model tutorial

2. Project sustainability: Currently, large models are still in their early stages of development and have not yet fully penetrated the industry. Therefore, as large models develop, this project can continue to provide help to learners

## Project Planning

**Table of Contents**
1. [Introduction](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch01.md)- Project goal: Focus on the current knowledge of large-scale pre-trained language models
- Project background: The emergence of large language models such as GPT-3, and the development of research in related fields
2. [Capabilities of large models](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch02.md)
- Model adaptation conversion: Migrate large model pre-training to downstream tasks
- Model performance evaluation: Evaluate and analyze the GPT-3 model based on multiple tasks
3. [Model architecture](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch03.md)
- Model structure: Study and implement network structures such as RNN and Transformer
- Details of each layer of Transformer: From position information encoding to attention mechanism
4. [New model architecture](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch04.md)
- Mixture of Experts (MoE) Model
-Retrieval-based model
5. [Data for large models](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch05.md)
- Data collection: Get the data required for training and evaluation from public datasets, such as The Pile dataset
- Data preprocessing: Data cleaning, word segmentation, etc.
6. [Model training](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch06.md)
- Objective function: Training method for large models
- Optimization algorithm: Optimization algorithm used for model training
7. [Adaptation of large models](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch07.md)
- Discuss why adaptation is needed
- Current mainstream adaptation methods (probing/fine-tuning/efficient fine-tuning)
8. [Distributed training](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch07.md)na/so-large-lm/blob/main/docs/content/ch08.md)
- Why do we need distributed training?
- Common parallel strategies: data parallelism, model parallelism, pipeline parallelism, hybrid parallelism
9. [The harmfulness of large models-Part 1](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch09.md)
- Model performance differences: pre-training or data processing affects the performance of large models
- Social bias: explicit social bias shown by the model
10. [The harmfulness of large models-Part 2](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch10.md)
- Model harmful information: the case of toxic information in the model
- Model false information: the case of false information in large models
11. [Law of large models](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch11.md)
- Judicial challenges caused by new technologies:- Summary of past judicial cases: Summary of past cases
12. [Environmental impact](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch12.md)
- Understand the impact of large language models on the environment
- Estimate the emissions generated by model training
13. [Agent](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch13.md)
- Understand the details of each component of Agent
- Challenges and opportunities of Agent
14. [Llama open source family: from Llama-1 to Llama-3](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch14.md)
- The evolution of Llama (Section 1)/ Model architecture (Section 2)/Training data (Section 3)/Training method (Section 4)/Effect comparison (Section 5)/Community ecology (Section 6)

## Core contributors

- [Chen Anton](https://github.com/andongBlue): PhD candidate in Natural Language Processing at Harbin Institute of Technology (initiator, project leader, project content construction)
- [Zhang Fan](https://github.com/zhangfanTJU): Master of Natural Language Processing Methods at Tianjin University (project content construction)

### Participants
- [Wang Maolin](https://github.com/mlw67): PhD candidate at Huazhong University of Science and Technology (solving issues)

## Project leader

Chen Andong 
Contact: ands691119@gmail.com