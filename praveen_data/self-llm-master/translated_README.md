<div align=center>
<img src="./images/head-img.png" >
<h1>Guide to using open source big models</h1>
</div>

&emsp;&emsp;This project is a Chinese baby-exclusive big model tutorial based on the AutoDL platform, centered around open source big models and aimed at domestic beginners. It provides full-process guidance for various open source big models, including environment configuration, local deployment, efficient fine-tuning and other skills, simplifies the deployment, use and application process of open source big models, allows more ordinary students and researchers to better use open source big models, and helps open source and free big models integrate into the lives of ordinary learners more quickly.

&emsp;&emsp;The main contents of this project include:

1. An open source LLM environment configuration guide based on the Linux platform, providing different detailed environment configuration steps for different model requirements;
2. Deployment and use tutorials for mainstream open source LLMs at home and abroad, including LLaMA, ChatGLM, InternLM, etc.;
3. Deployment and application guidance for open source LLM, including command line calls, online Demo deployment, LangChain framework integration, etc.;
4. Full-scale fine-tuning and efficient fine-tuning methods for open source LLM, including distributed full-scale fine-tuning, LoRA, ptuning, etc.

&emsp;&emsp;**The main content of the project is the tutorial, so that more students and future practitioners can understand and become familiar with the consumption of open source large models! Anyone can raise an issue or submit a PR to jointly build and maintain this project. **

&emsp;&emsp;Students who want to participate in depth can contact us, and we will add you to the maintainers of the project.

> &emsp;&emsp;***Learning suggestions: The learning suggestions for this project are to learn the environment configuration first, then learn the deployment and use of the model, and finally learn fine-tuning. Because the environment configuration is the basis, the deployment and use of the model is the basis, and fine-tuning is advanced. Beginners can choose Qwen1.5, InternLM2, MiniCPM and other models to learn first. ***

> Note: If students want to understand the model structure of large models, as well as hand-write RAG, Agent and Eval tasks from scratch, they can learn another project of Datawhale [Tiny-Universe] (https://github.com/datawhalechina/tiny-universe). Large models are currently a hot topic in the field of deep learning, but most of the existing large model tutorials only teach you how to call the API to complete the application of large models, and few people can explain the model structure, RAG, Agent and Eval from the principle level. Therefore, this warehouse will provide all hand-written, noComplete the RAG, Agent, and Eval tasks of the large model by calling the API.

> Note: Considering that some students want to learn the theoretical part of the large model before learning this project, if you want to further study the theoretical basis of LLM and further understand and apply LLM on the basis of theory, you can refer to Datawhale's [so-large-llm](https://github.com/datawhalechina/so-large-lm.git) course.

> Note: If some students want to develop large model applications by themselves after learning this course. Students can refer to Datawhale's [Hands-on Learning Large Model Application Development](https://github.com/datawhalechina/llm-universe) course. This project is a large model application development tutorial for novice developers. It aims to present the large model application development process to students based on Alibaba Cloud Server and combined with the personal knowledge base assistant project.

## Project Significance

&emsp;&emsp;What is a large model?

>Large model (LLM) in a narrow sense refers to the natural language processing (NLP) model trained based on deep learning algorithms, which is mainly used in fields such as natural language understanding and generation. In a broad sense, it also includes machine vision (CV) large models and multimodal large models.and scientific computing big models, etc.

&emsp;&emsp;The Hundred Models War is in full swing, and open source LLMs are emerging in an endless stream. Nowadays, many excellent open source LLMs have emerged at home and abroad, such as LLaMA and Alpaca abroad, and ChatGLM, BaiChuan, InternLM (Shusheng·Pu Yu) in China. Open source LLM supports local deployment and private domain fine-tuning by users. Everyone can build their own unique big model based on open source LLM.

&emsp;&emsp;However, ordinary students and users who want to use these big models need to have certain technical capabilities to complete the deployment and use of the models. For the endless and distinctive open source LLMs, it is a challenging task to quickly master the application method of an open source LLM.

&emsp;&emsp;This project aims to first implement the deployment, use and fine-tuning tutorials of mainstream open source LLM at home and abroad based on the experience of core contributors; after implementing the relevant parts of mainstream LLM, we hope to fully gather co-creators to enrich the world of open source LLM and create more and more comprehensive LLM tutorials. Sparks gather into a sea.

&emsp;&emsp;***We hope to become a ladder between LLM and the general public, and embrace a more magnificent and vast LLM world with the spirit of freedom and equality of open source. ***

## Project audience&emsp;&emsp;This project is suitable for the following learners:

* Want to use or experience LLM, but have no conditions to obtain or use related APIs;

* Hope to apply LLM in a long-term, low-cost, and large-scale manner;

* Interested in open source LLM and want to get started with open source LLM;

* Studying NLP and want to learn more about LLM;

* Hope to combine open source LLM to create a private domain LLM with field characteristics;

* And the largest and most common student groups.

## Project planning and progress

&emsp;&emsp; This project is planned to be organized around the entire process of open source LLM application, including environment configuration and use, application deployment, fine-tuning, etc. Each part covers the mainstream and characteristic open source LLM:

### Supported models

- [Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
- [x] [Gemma-2-9b-it FastApi deployment call](./Gemma2/01-Gemma-2-9b-it%20FastApi%20deployment call.md) @不要葱姜姜
- [x] [Gemma-2-9b-it langchain access](./Gemma2/02-Gemma-2-9b-it%20langchain%20Access.md) @不要葱姜姜
- [x] [Gemma-2-9b-it WebDemo deployment](./Gemma2/03-Gemma-2-9b-it%20WebDemo%20deployment.md) @不要葱姜姜
- [x] [Gemma-2-9b-it Peft Lora fine-tuning](./Gemma2/04-Gemma-2-9b-it%20peft%20lora fine-tuning.md) @不要葱姜姜

- [Yuan2.0](https://github.com/IEIT-Yuan/Yuan-2.0)
- [x] [Yuan2.0-2B FastApi deployment call](./Yuan2.0/01-Yuan2.0-2B%20FastApi%20deployment call.md) @张帆
- [x] [Yuan2.0-2B Langchain access](./Yuan2.0/02-Yuan2.0-2B%20Langchain%20access.md) @张帆
-[x] [Yuan2.0-2B WebDemo deployment](./Yuan2.0/03-Yuan2.0-2B%20WebDemo deployment.md) @张帆
-[x] [Yuan2.0-2B vLLM deployment call](./Yuan2.0/04-Yuan2.0-2B%20vLLM deployment call.md) @张帆
- [x] [Yuan2.0-2B Lora fine-tuning](./Yuan2.0/05-Yuan2.0-2B%20Lora fine-tuning.md) @张帆

- [Yuan2.0-M32](https://github.com/IEIT-Yuan/Yuan2.0-M32)
- [x] [Yuan2.0-M32 FastApi deployment call](./Yuan2.0-M32/01-Yuan2.0-M32%20FastApi%20deployment call.md) @张帆
- [x] [Yuan2.0-M32 Langchain Access](./Yuan2.0-M32/02-Yuan2.0-M32%20Langchain%20Access.md) @张帆
- [x] [Yuan2.0-M32 WebDemo deployment](./Yuan2.0-M32/03-Yuan2.0-M32%20WebDemo deployment.md) @张帆

- [DeepSeek-Coder-V2](https://github.com/deepseek-ai/DeepSeek-Coder-V2)
- [x] [DeepSeek-Coder-V2-Lite-Instruct FastApi deployment call](./DeepSeek-Coder-V2/01-DeepSeek-Coder-V2-Lite-Instruct%20FastApi%20deployment call.md) @姜舒凡
- [x] [DeepSeek-Coder-V2-Lite-Instruct langchain access](./DeepSeek-Coder-V2/02-DeepSeek-Coder-V2-Lite-Instruct%20access%20LangChain.md) @姜舒凡
- [x] [DeepSeek-Coder-V2-Lite-Instruct WebDemo Deployment](./DeepSeek-Coder-V2/03-DeepSeek-Coder-V2-Lite-Instruct%20WebDemo%20Deployment.md) @Kailigithub
-[x] [DeepSeek-Coder-V2-Lite-Instruct Lora Fine-tuning](./DeepSeek-Coder-V2/04-DeepSeek-Coder-V2-Lite-Instruct%20Lora%20Fine-tuning.md) @Yu Yang

- [Bilibili Index-1.9B](https://github.com/bilibili/Index-1.9B)
- [x] [Index-1.9B-Chat FastApi deployment call](./bilibili_Index-1.9B/01-Index-1.9B-chat%20FastApi%20deployment call.md) @Deng Kaijun
- [x] [Index-1.9B-Chat langchain access](./bilibili_Index-1.9B/02-Index-1.9B-Chat%20access%20LangChain.md) @Zhang Youdong
- [x] [Index-1.9B-Chat WebDemo Deployment](./bilibili_Index-1.9B/03-Index-1.9B-chat%20WebDemo deployment.md) @九月
- [x] [Index-1.9B-Chat Lora fine-tuning](./bilibili_Index-1.9B/04-Index-1.9B-Chat%20Lora%20fine-tuning.md) @姜舒凡

- [Qwen2](https://github.com/QwenLM/Qwen2)
- [x] [Qwen2-7B-Instruct FastApi deployment call](./Qwen2/01-Qwen2-7B-Instruct%20FastApi%20deployment call.md) @康婧淇
- [x] [Qwen2-7B-Instruct langchain access](./Qwen2/02-Qwen2-7B-Instruct%20Langchain%20access.md) @不要葱姜葱
- [x] [Qwen2-7B-Instruct WebDemo deployment](./Qwen2/03-Qwen2-7B-Instruct%20WebDemo deployment.md) @三水
- [x] [Qwen2-7B-Instruct vLLM Deployment call](./Qwen2/04-Qwen2-7B-Instruct%20vLLM%20Deployment call.md) @Jiang Shufan
-[x] [Qwen2-7B-Instruct Lora fine-tuning](./Qwen2/05-Qwen2-7B-Instruct%20Lora%20fine-tuning.md) @走走

- [GLM-4](https://github.com/THUDM/GLM-4.git)
- [x] [GLM-4-9B-chat FastApi deployment call](./GLM-4/01-GLM-4-9B-chat%20FastApi%20deployment call.md) @张友东
- [x] [GLM-4-9B-chat langchain access](./GLM-4/02-GLM-4-9B-chat%20langchain%20access.md) @谭逸珂
- [x] [GLM-4-9B-chat WebDemo deployment](./GLM-4/03-GLM-4-9B-Chat%20WebDemo.md) @何至轩
- [x] [GLM-4-9B-chat vLLM Deployment](./GLM-4/04-GLM-4-9B-Chat%20vLLM%20Deployment call.md) @王奕明
- [x] [GLM-4-9B-chat Lora fine-tuning](./GLM-4/05-GLM-4-9B-chat%20Lora%20fine-tuning.md) @肖鸿儒

- [Qwen 1.5](https://github.com/QwenLM/Qwen1.5.git)
- [x] [Qwen1.5-7B-chat FastApi deployment call](./Qwen1.5/01-Qwen1.5-7B-Chat%20FastApi%20 deployment call.md) @Yan Xin
- [x] [Qwen1.5-7B-chat langchain access](./Qwen1.5/02-Qwen1.5-7B-Chat%20 access langchain to build knowledge base assistant.md) @Yan Xin
- [x] [Qwen1.5-7B-chat WebDemo deployment](./Qwen1.5/03-Qwen1.5-7B-Chat%20WebDemo.md) @Yan Xin
- [x] [Qwen1.5-7B-chat Lora Fine-tuning](./Qwen1.5/04-Qwen1.5-7B-chat%20Lora%20fine-tuning.md) @不要葱姜姜
-[x] [Qwen1.5-72B-chat-GPTQ-Int4 deployment environment](./Qwen1.5/05-Qwen1.5-7B-Chat-GPTQ-Int4%20%20WebDemo.md) @byx020119
- [x] [Qwen1.5-MoE-chat Transformers deployment call](./Qwen1.5/06-Qwen1.5-MoE-A2.7B.md) @丁悦
- [x] [Qwen1.5-7B-chat vLLM reasoning deployment](./Qwen1.5/07-Qwen1.5-7B-Chat%20vLLM%20 reasoning deployment call.md) @高立业
- [x] [Qwen1.5-7B-chat Lora fine-tuning access to SwanLab experiment management platform](./Qwen1.5/08-Qwen1.5-7B-chat%20LoRA fine-tuning access to experiment management.md) @黄柏特

- [Google-Gemma](https://huggingface.co/google/gemma-7b-it)
- [x] [gemma-2b-it FastApi deployment call ](./Gemma/01-Gemma-2B-Instruct%20FastApi%20deployment call.md) @东东
- [x] [gemma-2b-it langchain access ](./Gemma/02-Gemma-2B-Instruct%20langchain%20Access.md) @东东
- [x] [gemma-2b-it WebDemo deployment](./Gemma/03-Gemma-2B-Instruct%20WebDemo%20Deployment.md) @东东
- [x] [gemma-2b-it Peft Lora fine-tuning](./Gemma/04-Gemma-2B-Instruct%20Lora fine-tuning.md) @东东

- [phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [x] [Phi-3-mini-4k-instruct FastApi deployment call](./phi-3/01-Phi-3-mini-4k-instruct%20FastApi%20Deployment call.md) @郑皓华
- [x] [Phi-3-mini-4k-instruct langchain Access](./phi-3/02-Phi-3-mini-4k-instruct%20langchain%20Access.md) @Zheng Haohua
-[x] [Phi-3-mini-4k-instructWebDemo deployment](./phi-3/03-Phi-3-mini-4k-instruct%20WebDemo deployment.md) @丁悦
- [x] [Phi-3-mini-4k-instruct Lora fine-tuning](./phi-3/04-Phi-3-mini-4k-Instruct%20Lora%20fine-tuning.md) @丁悦

- [CharacterGLM-6B](https://github.com/thu-coai/CharacterGLM-6B)
- [x] [CharacterGLM-6B Transformers deployment call](./CharacterGLM/01-CharacterGLM-6B%20Transformer deployment call.md) @孙健壮
- [x] [CharacterGLM-6B FastApi Deployment call](./CharacterGLM/02-CharacterGLM-6B%20FastApi deployment call.md) @Sun Jianzhuang
-[x] [CharacterGLM-6B webdemo deployment](./CharacterGLM/03-CharacterGLM-6B-chat.md) @孙健壮
- [x] [CharacterGLM-6B Lora fine-tuning](./CharacterGLM/04-CharacterGLM-6B%20Lora fine-tuning.md) @孙健壮

- [LLaMA3-8B-Instruct](https://github.com/meta-llama/llama3.git)
- [x] [LLaMA3-8B-Instruct FastApi deployment call](./LLaMA3/01-LLaMA3-8B-Instruct%20FastApi%20deployment call.md) @高立业
- [X] [LLaMA3-8B-Instruct langchain access](./LLaMA3/02-LLaMA3-8B-Instruct%20langchain%20access.md) @不要葱姜葱
- [x] [LLaMA3-8B-Instruct WebDemo Deployment](./LLaMA3/03-LLaMA3-8B-Instruct%20WebDemo%20Deployment.md) @不要葱姜姜
- [x] [LLaMA3-8B-Instruct Lora fine-tuning](./LLaMA3/04-LLaMA3-8B-Instruct%20Lora%20fine-tuning.md) @Gao Liye

- [XVERSE-7B-Chat](https://modelscope.cn/models/xverse/XVERSE-7B-Chat/summary)
- [x] [XVERSE-7B-Chat transformers deployment call](./XVERSE/01-XVERSE-7B-chat%20Transformers reasoning.md) @Guo Zhihang
- [x] [XVERSE-7B-Chat FastApi deployment call](./XVERSE/02-XVERSE-7B-chat%20FastAPI deployment.md) @Guo Zhihang
- [x] [XVERSE-7B-Chat langchain access](./XVERSE/03-XVERSE-7B-chat%20langchain%20access.md) @郭志航
- [x] [XVERSE-7B-Chat WebDemo deployment](./XVERSE/04-XVERSE-7B-chat%20WebDemo%20deployment.md) @郭志航
- [x] [XVERSE-7B-Chat Lora fine-tuning](./XVERSE/05-XVERSE-7B-Chat%20Lora%20fine-tuning.md) @郭志航

- [TransNormerLLM](https://github.com/OpenNLPLab/TransnormerLLM.git)
- [X] [TransNormerLLM-7B-Chat FastApi deployment call](./TransNormer/01-TransNormer-7B%20FastApi%20deployment call.md) @王茂霖
- [X] [TransNormerLLM-7B-Chat langchain access](./TransNormer/02-TransNormer-7B%20access langchain to build knowledge base assistant.md) @王茂霖
- [X] [TransNormerLLM-7B-Chat WebDemo Deployment](./TransNormer/03-TransNormer-7B%20WebDemo.md) @王茂霖
- [x] [TransNormerLLM-7B-Chat Lora fine-tuning](./TransNormer/04-TrasnNormer-7B%20Lora%20fine-tuning.md) @王茂霖

- [BlueLM Vivo Blue Heart Large Model](https://github.com/vivo-ai-lab/BlueLM.git)
- [x] [BlueLM-7B-Chat FatApi deployment call](./BlueLM/01-BlueLM-7B-Chat%20FastApi%20deployment.md) @郭志航
- [x] [BlueLM-7B-Chat langchain access](./BlueLM/02-BlueLM-7B-Chat%20langchain%20access.md) @郭志航
- [x] [BlueLM-7B-Chat WebDemo deployment](./BlueLM/03-BlueLM-7B-Chat%20WebDemo%20deployment.md) @郭志航
- [x] [BlueLM-7B-Chat Lora fine-tuning](./BlueLM/04-BlueLM-7B-Chat%20Lora%20fine-tuning.md) @郭志航

- [InternLM2](https://github.com/InternLM/InternLM)
- [x] [InternLM2-7B-chat FastApi deployment call](./InternLM2/01-InternLM2-7B-chat%20FastAPI deployment.md) @不要葱姜葱
- [x] [InternLM2-7B-chat langchain access](./InternLM2/02-InternLM2-7B-chat%20langchain%20 access.md) @不要葱姜葱
- [x] [InternLM2-7B-chat WebDemo deployment](./InternLM2/03-InternLM2-7B-chat%20WebDemo%20 deployment.md) @郑皓华
- [x] [InternLM2-7B-chat Xtuner Qlora Fine-tuning](./InternLM2/04-InternLM2-7B-chat%20Xtuner%20Qlora%20fine-tuning.md) @Zheng Haohua

- [DeepSeek](https://github.com/deepseek-ai/DeepSeek-LLM)
- [x] [DeepSeek-7B-chat FastApi deployment call](./DeepSeek/01-DeepSeek-7B-chat%20FastApi.md) @Don't use onions, ginger and garlic
- [x] [DeepSeek-7B-chat langchain access](./DeepSeek/02-DeepSeek-7B-chat%20langchain.md) @Don't use onions, ginger and garlic
- [x] [DeepSeek-7B-chat WebDemo](./DeepSeek/03-DeepSeek-7B-chat%20WebDemo.md) @Don't use onions, ginger and garlic
- [x] [DeepSeek-7B-chat Lora fine-tuning](./DeepSeek/04-DeepSeek-7B-chat%20Lora%20fine-tuning.md) @Don't use onions, ginger and garlic
- [x] [DeepSeek-7B-chat 4bits quantization Qlora Fine-tuning](./DeepSeek/05-DeepSeek-7B-chat%204bits quantization%20Qlora%20fine-tuning.md) @不要葱姜姜
- [x] [DeepSeek-MoE-16b-chat Transformers deployment call](./DeepSeek/06-DeepSeek-MoE-16b-chat%20Transformer deployment call.md) @Kailigithub
- [x] [DeepSeek-MoE-16b-chat FastApi deployment call](./DeepSeek/06-DeepSeek-MoE-16b-chat%20FastApi.md) @Kailigithub
- [x] [DeepSeek-coder-6.7b finetune colab](./DeepSeek/07-deepseek_fine_tune.ipynb) @Swiftie
- [x] [Deepseek-coder-6.7b webdemo colab](./DeepSeek/08-deepseek_web_demo.ipynb) @Swiftie

- [MiniCPM](https://github.com/OpenBMB/MiniCPM.git)
- [x] [MiniCPM-2B-chat transformers deployment call](./MiniCPM/MiniCPM-2B-chat%20transformers%20 Deployment call.md) @Kailigithub 
- [x] [MiniCPM-2B-chat FastApi deployment call](./MiniCPM/MiniCPM-2B-chat%20FastApi%20 deployment call.md) @Kailigithub 
- [x] [MiniCPM-2B-chat langchain access](./MiniCPM/MiniCPM-2B-chat%20langchain access.md) @不要葱姜姜 
- [x] [MiniCPM-2B-chat webdemo deployment](./MiniCPM/MiniCPM-2B-chat%20WebDemo deployment.md) @Kailigithub 
- [x] [MiniCPM-2B-chat Lora && Full Fine-tuning](./MiniCPM/MiniCPM-2B-chat%20Lora%20&&%20Full%20fine-tuning.md) @不要葱姜姜

- [Qwen-Audio](https://github.com/QwenLM/Qwen-Audio.git)
- [x] [Qwen-Audio FastApi Department[Deployment call](./Qwen-Audio/01-Qwen-Audio-chat%20FastApi.md) @陈思州
- [x] [Qwen-Audio WebDemo](./Qwen-Audio/02-Qwen-Audio-chat%20WebDemo.md) @陈思州

- [Qwen](https://github.com/QwenLM/Qwen.git)
- [x] [Qwen-7B-chat Transformers deployment call](./Qwen/01-Qwen-7B-Chat%20Transformers deployment call.md) @李娇娇
- [x] [Qwen-7B-chat FastApi deployment call](./Qwen/02-Qwen-7B-Chat%20FastApi%20deployment call.md) @李娇娇
- [x] [Qwen-7B-chat WebDemo](./Qwen/03-Qwen-7B-Chat%20WebDemo.md) @李娇娇
- [x] [Qwen-7B-chat Lora fine-tuning](./Qwen/04-Qwen-7B-Chat%20Lora%20fine-tuning.md) @不要葱姜姜
- [x] [Qwen-7B-chat ptuning fine-tuning](./Qwen/05-Qwen-7B-Chat%20Ptuning%20fine-tuning.md) @肖鸿儒
- [x] [Qwen-7B-chat full fine-tuning](./Qwen/06-Qwen-7B-chat%20full fine-tuning.md) @不要葱姜姜
- [x] [Qwen-7B-Chat access langchain to build knowledge base assistant](./Qwen/07-Qwen-7B-Chat%20access langchain to build knowledge base assistant.md) @李娇娇
- [x] [Qwen-7B-chat low-precision training](./Qwen/08-Qwen-7B-Chat%20Lora%20low-precision fine-tuning.md) @肖鸿儒
- [x] [Qwen-1_8B-chat CPU deployment](./Qwen/09-Qwen-1_8B-chat%20CPU%20deployment%20.md) @漫步

- [Yi Zero One Everything](https://github.com/01-ai/Yi.git)
- [x] [Yi-6B-chat FastApi deployment call](./Yi/01-Yi-6B-Chat%20FastApi%20Deployment call.md) @李柯辰
- [x] [Yi-6B-chat langchain access](./Yi/02-Yi-6B-Chat%20Access langchain to build knowledge base assistant.md) @李柯辰
- [x] [Yi-6B-chat WebDemo](./Yi/03-Yi-6B-chat%20WebDemo.md) @肖鸿儒
- [x] [Yi-6B-chat Lora fine-tuning](./Yi/04-Yi-6B-Chat%20Lora%20fine-tuning.md) @李娇娇

- [Baichuan Baichuan Intelligence](https://www.baichuan-ai.com/home)
- [x] [Baichuan2-7B-chat FastApi Deployment call](./BaiChuan/01-Baichuan2-7B-chat%2BFastApi%2B%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md) @Hui Jiahao
-[x] [Baichuan2-7B-chat WebDemo](./BaiChuan/02-Baichuan-7B-chat%2BWebDemo.md) @惠佳豪
- [x] [Baichuan2-7B-chat access to LangChain framework](./BaiChuan/03-Baichuan2-7B-chat%E6%8E%A5%E5%85%A5LangChain%E6%A1%86%E6%9E%B6.md) @惠佳豪
- [x] [Baichuan2-7B-chat Lora fine-tuning](./BaiChuan/04-Baichuan2-7B-chat%2Blora%2B%E5%BE%AE%E8%B0%83.md) @惠佳豪

- [InternLM](https://github.com/InternLM/InternLM.git)
- [x] [InternLM-Chat-7B Transformers Deployment call](./InternLM/01-InternLM-Chat-7B%20Transformers%20Deployment call.md) @Xiao Luo
-[x] [InternLM-Chat-7B FastApi deployment call](InternLM/02-internLM-Chat-7B%20FastApi.md) @不要葱姜姜
- [x] [InternLM-Chat-7B WebDemo](InternLM/03-InternLM-Chat-7B.md) @不要葱姜姜
- [x] [Lagent+InternLM-Chat-7B-V1.1 WebDemo](InternLM/04-Lagent+InternLM-Chat-7B-V1.1.md) @不要葱姜姜
- [x] [Puyu Lingbi Graphics and Texts Understanding & Creation WebDemo](InternLM/05-Puyu Lingbi Graphics and Texts Understanding & Creation.md) @不要葱姜姜
- [x] [InternLM-Chat-7B Access to LangChain Framework](InternLM/06-InternLM Access to LangChain to Build Knowledge Base Assistant.md) @Logan Zou

- [Atom (llama2)](https://hf-mirror.com/FlagAlpha/Atom-7B-Chat) - [x] [Atom-7B-chat WebDemo](./Atom/01-Atom-7B-chat-WebDemo.md) @Kailigithub - [x] [Atom-7B-chat Lora fine-tuning](./Atom/02-Atom-7B-Chat%20Lora%20fine-tuning.md) @Logan Zou
- [x] [Atom-7B-Chat access langchain to build a knowledge base assistant](./Atom/03-Atom-7B-Chat%20access langchain to build a knowledge base assistant.md) @陈思州
- [x] [Atom-7B-chat full fine-tuning](./Atom/04-Atom-7B-chat%20full fine-tuning.md) @Logan Zou

- [ChatGLM3](https://github.com/THUDM/ChatGLM3.git)
- [x] [ChatGLM3-6B Transformers deployment call](./ChatGLM/01-ChatGLM3-6B%20Transformer deployment call.md) @丁悦
- [x] [ChatGLM3-6B FastApi deployment call](./ChatGLM/02-ChatGLM3-6B%20FastApi deployment call.md) @丁悦
- [x] [ChatGLM3-6B chat WebDemo](ChatGLM/03-ChatGLM3-6B-chat.md) @不要葱姜姜
- [x] [ChatGLM3-6B Code Interpreter WebDemo](ChatGLM/04-ChatGLM3-6B-Code-Interpreter.md) @不要葱姜姜
- [x] [ChatGLM3-6B access to LangChain framework](ChatGLM/05-ChatGLM3-6B access to LangChain to build a knowledge base assistant.md) @Logan Zou
- [x] [ChatGLM3-6B Lora fine-tuning](ChatGLM/06-ChatGLM3-6B-Lora fine-tuning.md) @肖鸿儒

### General environment configuration

- [x] [pip, conda source change](./General-Setting/01-pip, conda source change.md) @不要葱姜姜
- [x] [AutoDL open port](./General-Setting/02-AutoDL open port.md) @不要葱姜姜

- Model download
- [x] [hugging face](./General-Setting/03-Model download.md)@不要葱姜姜
- [x] [hugging face](./General-Setting/03-Model Download.md) Mirror download @不要葱姜姜
- [x] [modelscope](./General-Setting/03-Model Download.md) @不要葱姜姜
- [x] [git-lfs](./General-Setting/03-Model Download.md) @不要葱姜姜
- [x] [Openxlab](./General-Setting/03-Model Download.md)
- Issue && PR
- [x] [Issue Submission](./General-Setting/04-Issue&PR&update.md) @肖鸿儒
- [x] [PR Submission](./General-Setting/04-Issue&PR&update.md) @肖鸿儒
- [x] [Fork update](./General-Setting/04-Issue&PR&update.md) @肖鸿儒

## Acknowledgements

### Core Contributors

- [Song Zhixue (Don't use onions, ginger and garlic) - Project Leader](https://github.com/KMnO4-zx) (Datawhale member-Henan Polytechnic University)
- [Zou Yuheng-Project Leader](https://github.com/logan-zou) (Datawhale member-University of International Business and Economics)
- [Xiao Hongru](https://github.com/Hongru0306) (Datawhale member-Tongji University)
- [Guo Zhihang](https://github.com/acwwt) (Content creator)
- [Zhang Fan](https://github.com/zhangfanTJU) (Content creator-Datawhale member)
- [Kailigithub](https://github.com/Kailigithub) (Datawhale member)
- [Li Jiaojiao](https://github.com/Aphasia0515) (Datawhale member)
- [Ding Yue](https://github.com/dingyue772) (Datawhale-Jingying teaching assistant)
- [Hui Jiahao](https://github.com/L4HeyXiao) (Datawhale-Publicity Ambassador)
- [Zheng Haohua](https://github.com/BaiYu96) (Content creator)
- [Wang Maolin](https://github.com/mlw67) (Content creator-Datawhale member)
- [Sun Jianzhuang](https://github.com/Caleb-Sun-jz) (Content creator-University of International Business and Economics)
- [Dongdong](https://github.com/LucaChen) (Content creator-Google Developer Machine Learning Technology Expert)
- [Jiang Shufan](https://github.com/Tsumugii24) (Content creator-Jingying Assistant Teacher)
- [Chen Sizhou](https://github.com/jjyaoao) (Datawhale member)
- [Shanbuphy](https://github.com/sanbuphy) (Datawhale member)
- [Yan Xin](https://github.com/thomas-yanxin) (Datawhale member)
- [Li Kechen](https://github.com/Joe-2002) (Datawhale member)
- [Swiftie](https://github.com/cswangxiaowei) (Xiaomi NLP algorithm=Law engineer)
- [Huang Baite](https://github.com/KashiwaByte) (Content creator-Xi'an University of Electronic Science and Technology)
- [Zhang Youdong](https://github.com/AXYZdong) (Content creator-Datawhale member)
- [Yu Yang](https://github.com/YangYu-NUAA) (Content creator-Datawhale member)
- [Xiao Luo](https://github.com/lyj11111111) (Content creator-Datawhale member)
- [Tan Yike](https://github.com/LikeGiver) (Content creator-University of International Business and Economics)
- [Wang Yiming](https://github.com/Bald0Wang) (Content creator-Datawhale member)
- [He Zhixuan](https://github.com/pod2c) (Content creator-Jingying teaching assistant)
- [Kang Jingqi](https://github.com/jodie-kang) (Content creator-Datawhale member)
- [Sanshui](https://github.com/sssanssss) (Content creator - Jingying teaching assistant)
-[September](https://github.com/chg0901) (Content creator-Datawhale intended member)
- [Kaijun Deng](https://github.com/Kedreamix) (Content creator-Datawhale member)

> Note: Ranking is based on contribution

### Others

- Special thanks to [@Sm1les](https://github.com/Sm1les) for the help and support of this project
- Some lora code and explanation reference repository: https://github.com/zyds/transformers-code.git
- If you have any ideas, please contact us DataWhale also welcomes everyone to raise issues
- Special thanks to the following students who contributed to the tutorial!

<div align=center style="margin-top: 30px;">
<a href="https://github.com/datawhalechina/self-llm/graphs/contributors">
<img src="https://contrib.rocks/image?repo=datawhalechina/self-llm" /> </a> </div> ### Star History <div align=center style="margin-top: 30px;"> <img src="./images/star-history-202473 .png"/> </div>