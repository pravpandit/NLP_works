# CharacterGLM-6B
## Introduction
CharacterGLM-6B is a new generation of dialogue pre-training model jointly released by Lingxin Intelligence and Tsinghua University CoAI Laboratory.
The files under this folder contain the deployment and fine-tuning process of CharacterGLM-6B.

## 01-CharacterGLM-6B Transformer deployment call
This project is about how to use the CharacterGLM-6B model for dialogue generation deployment call. With this deployment, you can easily rent a graphics card machine with enough video memory in the Autodl platform, and then configure the environment, download the model and run the demo to realize the dialogue generation function based on the model.
### Environment preparation
1. Rent a graphics card machine with 24G video memory in the Autodl platform.
2. Open JupyterLab and configure the environment, download the model and run the demo in the terminal.
### Model download
Download the model by using the snapshot_download function in modelscope. You can specify the model name and download path. The model download path is set to /root/autodl-tmp, and the model size is 12 GB.
### Code Preparation
In the code preparation section, you need to load the model and tokenizer, and move the model to the GPU (if available).Then, you can generate dialogues by using the evaluation mode of the model. The sample code includes an example of three rounds of dialogues, showing how to use the model.
### Deployment
By running the trans.py file in the terminal, you can implement the Transformers deployment call of the CharacterGLM-6B model. Observing the loading checkpoint in the command line indicates that the model is loading, and you can start generating dialogues after the model is loaded.
With these steps, you can easily deploy and use the CharacterGLM-6B model for dialogue generation.

## 02-CharacterGLM-6B FastApi deployment call
This project is about how to use the CharacterGLM-6B model for dialogue generation FastApi deployment calls. With this deployment, you can easily rent a graphics card machine with enough video memory in the Autodl platform, then configure the environment, download the model and run the demo to implement the dialogue generation function based on the model.
### Environment preparation
1. Rent a graphics card machine with 24G video memory in the Autodl platform.
2. Open JupyterLab and configure the environment, download the model, and run the demo in the terminal.
### Model download
Download the model by using the snapshot_download function in modelscope.Type, you can specify the model name and download path. The download path of the model is set to /root/autodl-tmp, and the model size is 12 GB.
### Code preparation
In the code preparation section, you need to load the model and tokenizer, and move the model to the GPU (if available).
### Deployment
By running the api file in the terminal, you can implement the FastApi deployment call of the CharacterGLM-6B model. Observing the loading checkpoint in the command line indicates that the model is loading, and you can start generating conversations after the model is loaded.
With these steps, you can easily deploy and use the CharacterGLM-6B model for conversation generation.

## 03-CharacterGLM-6B-chat
This file contains the code for using the CharacterGLM-6B model for chatbot-like reasoning. The CharacterGLM-6B model is a fine-tuned large language model for generating text replies in conversation scenarios.
### Environment setup
1. Rent a server with a 24GB GPU on the autodl platform.
2. Open JupyterLab on the rented server and start configuring the environment, downloading models, and running demos in the terminal.
3. Replace the pip source and install the required packages.
### ModelDownload
Use the `snapshot_download` function in modelscope to download the model and cache it in the specified path.
### Code preparation
Clone the code repository and modify the code path and configuration file as needed.
### Demo run
Modify the code path and run the demo to interact with the model through the browser or command line.

## 04-CharacterGLM-6B Lora fine-tuning
This article briefly introduces how to fine-tune the CharacterGLM-6B-chat model with Lora based on transformers and peft frameworks. For the principle of Lora, please refer to the blog: [Zhihu|Lora in Depth](https://zhuanlan.zhihu.com/p/650197598)
### Environment configuration
After completing the basic environment configuration and local model deployment, you also need to install some third-party libraries.
### Instruction set construction
LLM fine-tuning generally refers to the instruction fine-tuning process.
### The difference and connection between QA and Instruction
QA refers to a question-and-answer format, usually a user asks a question and the model gives an answer. Instruction comes from Prompt Engineering, which splits the question into two parts: Instruction is used to describe the task, Instruction is used to describe the task, and Instruction is used to describe the task.put is used to describe the object to be processed.
### Data formatting
The data trained by Lora needs to be formatted and encoded before being input to the model for training.
### Loading tokenizer and half-precision model
The model is loaded in half-precision form.
### Define LoraConfig
Many parameters can be set in the LoraConfig class.
### Customize TraininArguments parameters
The source code of the TrainingArguments class also introduces the specific role of each parameter.
### Training with Trainer
Put the model in, the parameters set above, the data set in, and start training.
### Model inference
### Reload
The model fine-tuned by PEFT can be reloaded and inferred using the following method.