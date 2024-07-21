# CharacterGLM-6B Transformers deployment call

## Environment preparation

Rent a 3090 or other 24G video memory graphics card machine in the autodl platform. As shown in the figure below, select PyTorch-->2.0.0-->3.8(ubuntu20.04)-->11.8

![image](https://github.com/suncaleb1/self-llm/assets/155936975/fc4c6323-d338-4d66-a244-bbefe7da3746)

Next, open JupyterLab on the server you just rented, and open the terminal in it to start environment configuration, model download and run demo.

pip change source and install dependent packages

```python
#Upgrade pip
python -m pip install --upgrade pip
#Change pypi source to accelerate library installation
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope
pip install transformers
pip installsentencepiece
```

## Model download

Use the snapshot_download function in modelscope to download the model. The first parameter is the model name, and the parameter cache_dir is the download path of the model.

Create a new download.py file in the /root/autodl-tmp path and enter the following content in it. Remember to save the file after pasting the code, as shown in the figure below. And run python /root/autodl-tmp/download.py to execute the download. The model size is 12 GB. It takes about 10~15 minutes to download the model.

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('THUCoAI/CharacterGLM-6B', cache_dir='/root/autodl-tmp', revision='master')
```

## Code preparation

```python
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
# Load the model using the local path where it was downloaded
model_dir = '/root/autodl-tmp/THUCoAI/CharacterGLM-6B'
# Load the tokenizer, load locally, trust_remote_code=True setting allows model weights and related code to be downloaded from the network
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# Model loading, local loading, using the AutoModelForCausalLM class
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
# Move the model to the GPU for acceleration (if there is a GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Generate a conversation using the model's evaluation modemodel.eval()
session_meta = {'user_info': 'I am Lu Xingchen, a male, a well-known director, and Su Mengyuan's co-director. I am good at shooting music-themed movies. Su Mengyuan respects me and regards me as a mentor and friend. ', 'bot_info': 'Su Mengyuan, whose real name is Su Yuanxin, is a popular domestic female singer and actress. After participating in a talent show, she quickly became famous and entered the entertainment industry with her unique voice and outstanding stage charm. She looks beautiful, but her real charm lies in her talent and diligence. Su Mengyuan is an outstanding student who graduated from the Conservatory of Music. She is good at creation and has many popular original songs. In addition to her achievements in music, she is also keen on charity, actively participates in public welfare activities, and conveys positive energy with practical actions. At work, she is very dedicated to her work. When filming, she always devotes herself to the role, which has won praise from industry insiders and the love of fans. Although in the entertainment industry, she always maintains a low-key and humble attitude, and is deeply respected by her peers. When expressing, Su Mengyuan likes to use "we" and "together" to emphasize team spirit. ', 'bot_name': 'Su Mengyuan', 'user_name': 'Lu Xingchen'}
# First round of dialogue
response, history = model.chat(tokenizer, session_meta,"Hello, Xiao Su", history=[])
print(response)
# 第二轮对话
response, history = model.chat(tokenizer, session_meta,"最近对音乐有什么新的想法吗", history=history)
print(response)
# 第三轮对话
response, history = model.chat(tokenizer,session_meta, "那我们商量一下下一部音乐电影的拍摄，好嘛？", history=history)
print(response)
```

## 部署

在终端输入以下命令运行trans.py，即实现CharacterGLM-6B的Transformers部署调用

```python
cd /root/autodl-tmp
python trans.py
```

观察命令行中loading checkpoint表示模型正在加载，等待模型加载完成产生对话，如下图所示

![image](https://github.com/suncaleb1/self-llm/assets/155936975/f9d65275-fa89-4039-95c5-7cc0615753e2)