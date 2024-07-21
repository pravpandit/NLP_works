<div align="center">
<h1>
Source 2.0 Big Model
</h1>
</div>

## 1. Model Introduction

Source 2.0 is a new generation of basic language big model released by Inspur Information, including Source 2.0-102B, Source 2.0-51B and Source 2.0-2B. Source 2.0 is based on Source 1.0, using more diverse high-quality pre-training data and instruction fine-tuning data sets to enable the model to have stronger understanding capabilities in different aspects such as semantics, mathematics, reasoning, code, and knowledge.

In terms of algorithms, Source 2.0 proposes and adopts a new type of attention algorithm structure: Localized Filtering-based Attention (LFA). LFA can better learn the local and global language features of natural language by first learning the correlation between adjacent words and then calculating the global correlation. It has a more accurate and humane understanding of the associated semantics of natural language, which improves the natural language expression ability of the model and thus improves the model accuracy.

<div align=center>
<img src=images/yuan2.0-0.png >
<p>Fig.1: Yuan 2.0 Architecture and LFA</p>
</div>

<div align=center>
<img src=images/yuan2.0-1.jpg >
<p>Fig.2: Yuan 2.0 performance on mainstream industry evaluation tasks</p>
</div>

Project address: https://github.com/IEIT-Yuan/Yuan-2.0

Official report: https://mp.weixin.qq.com/s/rjnsUS83TT7aEN3r2i0IPQ

Paper link: https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/docs/Yuan2.0_paper.pdf

## 2. Model download
Yuan2.0 provides a variety of model formats, the download links are shown in the following table:

| Model | Sequence length |Download link |
|:-------------------------------------------------------------------:| :------: |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -------------------------------------------------- -------------------------------------------------- ----------------------------:| | Source 2.0-102B-hf | 4K | [ModelScope](https://modelscope. cn/models/YuanLLM/Yuan2.0-102B-hf/summary) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-102B-hf) \| [OpenXlab](https://openxlab. org.cn/models/detail/YuanLLM/Yuan2-102B-hf) \| [Baidu Cloud Disk](https://pan.baidu.com/s/1O4GkPSTPu5nwHk4v9byt7A?pwd=pq74#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-102B-hf) | | Source 2.0-51B-hf | 4K | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-51B-hf/summary) \| [HuggingFace](https://huggingface.co/ IEITYuan/Yuan2.0-51B-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-51B-hf) \| [Baidu Cloud Disk](https://pan.baidu.com/s/1-qw30ZuyrMfraFtkLgDg0A?pwd=v2nd#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-51B-hf) | | Source 2.0-2B-hf | 8K | [ModelScope ](https://modelscope.cn/models/YuanLLM/Yuan2.0-2B-hf/summary) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-2B-hf) \| [OpenXlab ](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-2B-hf) \| [Baidu Netdisk](https://pan.baidu.com/s/1nt-03OAnjtZwhiVywj3xGw?pwd= nqef#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-2B-hf) | | Source 2.0-2B-Janus-hf | 8K | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-2B-Janus -hf/files) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-2B-Janus-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM /Yuan2-2B-Janus-hf) \| [Baidu Netdisk](https://pan.baidu.com/s/1f7l-rSVlYAij33htR51TEg?pwd=hkep ) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-2B-Janus-hf) || Source 2.0-2B-Februa-hf | 8K | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-2B-Februa-hf) \| [HuggingFace](https://huggingface.co/ IEITYuan/Yuan2-2B-Februa-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-2B-Februa-hf) \| [Baidu Netdisk](https: //pan.baidu.com/s/1f7l-rSVlYAij33htR51TEg?pwd=hkep ) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-2B-Februa-hf) | | Source 2.0-2B-Mars-hf <sup><font color="#FFFF00">*New*</font><br /></sup> | 8K | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-2B-Mars-hf) \| [HuggingFace](https ://huggingface.co/IEITYuan/Yuan2-2B-Mars-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-2B-Mars-hf) \| [ Baidu Netdisk](https://pan.baidu.com/s/1f7l-rSVlYAij33htR51TEg?pwd=hkep) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-2B -Mars-hf) | Note: Source 2.0-2B-Mars-hf, Source 2.0-2B-Februa-hf, Source 2.0-2B-Janus-hf are all iterative versions of Source 2.0-2B-hf.

## 3. Tutorial introduction is in the previous section IntroducedAmong multiple models, source 2.0-2B-Mars-hf is the latest version of the 2B parameter model. The requirements for video memory and hard disk for fine-tuning and deploying source 2.0-2B-Mars-hf are relatively low.

Therefore, this tutorial is based on source 2.0-2B-Mars-hf and introduces the following content:

- 01-Yuan2.0-2B FastApi deployment call
- 02-Yuan2.0-2B Langchain access
- 03-Yuan2.0-2B WebDemo deployment
- 04-Yuan2.0-2B vLLM deployment call
- 05-Yuan2.0-2B Lora fine-tuning