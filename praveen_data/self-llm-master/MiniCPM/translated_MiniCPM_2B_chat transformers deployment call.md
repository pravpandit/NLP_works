# MiniCPM-2B-chat transformers deployment call

## MiniCPM-2B-chat introduction

MiniCPM is a series of large end-side models jointly open-sourced by Mianbi Intelligence and the Natural Language Processing Laboratory of Tsinghua University. The main language model MiniCPM-2B has only 2.4 billion (2.4B) non-word embedding parameters.

After SFT, MiniCPM is similar to Mistral-7B (with better Chinese, mathematics, and coding capabilities) on the public comprehensive evaluation set, and its overall performance exceeds Llama2-13B, MPT-30B, Falcon-40B and other models.
After DPO, MiniCPM-2B also surpassed many representative open source large models such as Llama2-70B-Chat, Vicuna-33B, Mistral-7B-Instruct-v0.1, and Zephyr-7B-alpha on the current evaluation set MTBench, which is closest to user experience.
Based on MiniCPM-2B, the end-side multimodal large model MiniCPM-V is built. The overall performance is the best among the models of the same scale, surpassing the existing multimodal large models built based on Phi-2, and achieving performance comparable to or even better than the 9.6B Qwen-VL-Chat on some evaluation sets..
After Int4 quantization, MiniCPM can be deployed on mobile phones for inference, and the streaming output speed is slightly higher than the speed of human speech. MiniCPM-V also directly runs through the deployment of multimodal large models on mobile phones.
A 1080/2080 can be used for efficient parameter fine-tuning, and a 3090/4090 can be used for full parameter fine-tuning. One machine can continuously train MiniCPM, and the secondary development cost is low.

## Environment preparation
Rent a **single card 3090 or other 24G** graphics card machine in the autodl platform. As shown in the figure below, select PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1
Next, open the JupyterLab of the server just rented, image and open the terminal in it to start environment configuration, model download and run demonstration. 
![Alt ​​text](images/image-1.png)

Next, open the JupyterLab server you just rented, and open the terminal to start environment configuration, model download, and run the demo.

pip source change and install dependent packages

```shell
# Upgrade pip
python -m pip install --upgrade pip
# Change the pypi source to speed up the installation of the library
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install modelscope transformers sentencepiece accelerate

MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

> Note: flash-attn installation will be slow, about ten minutes.

## Model download

Use the `snapshot_download` function in `modelscope` to download the model. The first parameter is the model name, and the parameter `cache_dir` is the download path of the model.

Create a new `download.py` file in the `/root/autodl-tmp` path and enter the following content in it. Remember to save the file after pasting the code, as shown in the figure below. And run `python /root/autodl-tmp/download.py` to download. The model size is 10 GB, and it takes about 5~10 minutes to download the model.

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('OpenBMB/MiniCPM-2B-sft-fp32', cache_dir='/root/autodl-tmp', revision='master')
```

## Code preparation

Create a new trains.py file in the /root/autodl-tmp path and enter the following content in it
```python
from transformers import AutoModelForCausalLM, AutoTokenizer # Import the required classes from the transformers library
import torch # Import the torch library for deep learning related operations

torch.manual_seed(0) # Set the random seed to ensure the reproducibility of the results

# Define the model path
path = '/root/autodl-tmp/OpenBMB/MiniCPM-2B-sft-fp32'

# Load the tokenizer from the model path,
tokenizer = AutoTokenizer.from_pretrained(path)

# Load the model from the model path, set it to use bfloat16 precision to optimize performance, and deploy the model to a CUDA-enabled GPU, trust_remote_code=True allows loading remote code
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)

# Chat with the model, ask questions and set generation parameters such as temperature, top_p value and repetition_penalty (repetition penalty factor)
responds, history = model.chat(tokenizer, "Which mountain is the highest in Shandong Province? Is it higher or lower than Huangshan? What is the difference?", temperature=0.5, top_p=0.8, repetition_penalty=1.02)

# Display the generated answer
print(responds)
```
### Deployment

Enter the following command in the terminal to run trains.py, which implements the Transformer of MiniCPM-2B-chats deployment call

```shell
cd /root/autodl-tmp
python trains.py
```
Observe the command line and wait for the model to load and generate a dialogue, as shown in the following figure
![image](images/image-2.png)