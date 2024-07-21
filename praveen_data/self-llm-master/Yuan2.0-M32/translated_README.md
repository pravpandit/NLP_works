<div align="center">
<h1>
Yuan2.0 M32 Large Model
</h1>
</div>

## 1. Model Introduction

Inspur Information **"Yuan2.0 M32" Large Model (abbreviated as Yuan2.0-M32)** adopts sparse mixed expert architecture (MoE), takes Yuan2.0-2B model as the base model, and realizes the collaborative work and task scheduling between 32 experts (Experts*32) through innovative gating network (Attention Router), which significantly reduces the demand for model inference computing power and brings stronger model accuracy and inference performance; Yuan2.0-M32 has been evaluated in code generation, mathematical problem solving, scientific question answering and comprehensive knowledge ability in multiple mainstream industry evaluations. The results show that Yuan2.0-M32 has demonstrated relatively advanced performance in multiple task evaluations, and the test accuracy of MATH (mathematical solution) and ARC-C (scientific question answering) exceeds that of LLaMA3-70 billion model.

**Yuan2.0-M32 large model** Basic information is as follows:

+ **Model parameter quantity: ** 40B <br>
+ **Number of experts: ** 32 <br>
+ **Number of activated experts: ** 2 <br>
+ **Activation parameter quantity: ** 3.7B <br> 
+ **Training data quantity: ** 2000B tokens <br>
+ **Support sequence length:** 16K <br>

<div align=center>
<img src=images/yuan2.0-m32-0.jpg >
<p>Fig.1: Yuan2.0-M32 architecture diagram</p>
</div>

<div align=center>
<img src=images/yuan2.0-m32-1.jpg >
<p>Fig.2: Yuan2.0-M32 industry mainstream evaluation task performance</p>
</div>

Project address: https://github.com/IEIT-Yuan/Yuan2.0-M32

Official report: https://mp.weixin.qq.com/s/WEVyYq9BkTTlO6EAfiCf6w

Technical report: https://arxiv.org/abs/2405.17976

## 2. Model Download
Yuan2.0-M32 provides a variety of model formats, the download links are shown in the following table:

| Model | Sequence Length | Model Format | Download Link |
| :----------: | :------: | :-------: |:---------------------------: | | Yuan2.0- M32 | 16K | Megatron | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32/) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32) \| [ Baidu Netdisk](https://pan.baidu.com/s/1K0LVU5NxeEujtYczF_T-Rg?pwd=cupw) \| [Shizhi AI](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2 -M32) | Yuan2.0-M32-HF | 16K | HuggingFace | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-hf) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-hf) \| [Baidu Netdisk](https://pan.baidu.com/s/1FrbVKji7IrhpwABYSIsV-A?pwd=q6uh)\| [Intelligent AI](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-hf)
| Yuan2.0-M32-GGUF | 16K | GGUF | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-gguf/summary) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-gguf) \| [Baidu Netdisk](https://pan.baidu.com/s/1BWQaz-jeZ1Fe69CqYtjS9A?pwd=f4qc) \| [Shizhi AI](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-gguf) | Yuan2.0-M32-GGUF-INT4 | 16K | GGUF | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-gguf-int4/summary) \| [HuggingFace](https://huggingface.co/IEITYuan/ Yuan2-M32-gguf-int4) \| [Baidu Netdisk](https://pan.baidu.com/s/1FM8xPpkhOrRcAfe7-zUgWQ?pwd=e6ag) \| [Shizhi AI](https://www. wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-gguf-int4) | Yuan2.0-M32-HF-INT4 | 16K | HuggingFace | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2 -M32-HF-INT4/summary) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-hf-int4) \| Baidu Netdisk \| [Early Intelligence AI](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-hf-int4/)

## 3. Tutorial Introduction

Among the multiple models introduced in the previous section, Yuan2-M32-HF-INT4 is a model that is quantized from the original Yuan2-M32-HF through auto-gptq.

Through model quantization, the requirements for video memory and hard disk for deploying Yuan2-M32-HF-INT4 will be significantly reduced.

The memory usage of Yuan2-M32-HF-INT4 deployed with 3090 is shown in the figure below:

<div align=center>
<img src=images/gpu.png >
<p>Fig.2: Yuan2-M32-HF-INT4 memory usage</p>
</div>

Therefore, this tutorial is based on Yuan2-M32-HF-INT4 and introduces the following content:

- 01-Yuan2.0-M32 FastApi deployment call
- 02-Yuan2.0-M32 Langchain access
- 03-Yuan2.0-M32 WebDemo deployment