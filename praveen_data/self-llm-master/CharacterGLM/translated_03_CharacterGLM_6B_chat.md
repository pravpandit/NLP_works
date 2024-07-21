# CharacterGLM-6B-chat

## Environment preparation

Rent a 3090 or other 24G graphics card machine in the autodl platform. As shown in the figure below, select PyTorch-->2.0.0-->3.8(ubuntu20.04)-->11.8

![image](https://github.com/suncaleb1/self-llm/assets/155936975/0dddbee9-df80-4033-9568-185ea585f261)

Next, open JupyterLab on the server you just rented, and open the terminal in it to start environment configuration, model download and run demo.

pip change source and install dependent packages

```python
# Upgrade pip
python -m pip install --upgrade pip
# Change pypi source to accelerate library installation
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope
pip install transformers
```

## Model download

Use modeThe snapshot_download function in lscope downloads the model. The first parameter is the model name, and the parameter cache_dir is the download path of the model.

Create a new download.py file in the /root/autodl-tmp path and enter the following content in it. Remember to save the file after pasting the code, as shown in the figure below. And run python /root/autodl-tmp/download.py to execute the download. The model size is 12 GB, and it takes about 10~15 minutes to download the model.

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('THUCoAI/CharacterGLM-6B', cache_dir='/root/autodl-tmp', revision='master')
```

## Code preparation

First, clone the code and open the academic image acceleration that comes with the autodl platform. For detailed use of academic image acceleration, please see:
https://www.autodl.com/docs/network_turbo/

```python
source /etc/network_turbo
```

Then switch the path and clone the code.

```python
cd /root/autodl-tmp
git clone https://github.com/thu-coai/CharacterGLM-6B
```

## Demo run

Modify the code path and change The model in line 20 of /root/autodl-tmp/CharacterGLM-6B/basic_demo/web_demo_streamlit.py is replaced with the local /root/autodl-tmp/THUCoAI/CharacterGLM-6B

![image](https://github.com/suncaleb1/self-llm/assets/155936975/1edc97a2-3d6e-43e3-b176-644b756b615f)

Modify the requirements.txt file and delete torch from it. Torch is already in the environment and does not need to be installed again. Then execute the following command:

```python
cd /root/autodl-tmp/CharacterGLM-6B
pip install -r requirements.txt
```

Run the following command in the terminal to start the inference service. Try to cd to the basic_demo folder to prevent the character.json file from being found

```python
cd /root/autodl-tmp/CharacterGLM-6B/basic_demo
streamlit run ./web_demo2.py --server.address 127.0.0.1 --server.port 6006
```

![image](https://github.com/suncaleb1/self-llm/assets/155936975/2fff8bd4-6d4b-449f-81ee-dc9e42b8ceb8)

After mapping the autodl port to the local http://localhost:6006, you can see the demo interface. For specific mapping steps, refer to the document /02-AutoDL open port.md in the General-Setting folder.

Open the http://localhost:6006 interface in the browser, modelLoad and use it, as shown below.

![image](https://github.com/suncaleb1/self-llm/assets/155936975/ac7a9887-4628-4539-9297-caccfb523530)

## Command line operation

Modify the code path and replace the model path in /root/autodl-tmp/CharacterGLM-6B/basic_demo/cli_demo.py with the local /root/autodl-tmp/THUCoAI/CharacterGLM-6B

Run the following command in the terminal to start the inference service

```python
cd /root/autodl-tmp/CharacterGLM-6B/basic_demo
python ./cli_demo.py ``` ![image](https://github.com/suncaleb1/self-llm/assets/155936975/1eb29dd5-8bae-458f-908f-f7388ae248c0)