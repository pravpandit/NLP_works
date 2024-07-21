# CharacterGLM-6B FastApi deployment call

## Environment preparation

Rent a 3090 or other 24G graphics card machine in the autodl platform. As shown in the figure below, select PyTorch-->2.0.0-->3.8(ubuntu20.04)-->11.8

![image](https://github.com/suncaleb1/self-llm/assets/155936975/2992ca12-7566-4916-94a6-1367df1a0d35)

Next, open JupyterLab on the server you just rented, and open the terminal in it to start environment configuration, model download and run demo.

pip change source and install dependent packages

```python
# Upgrade pip
python -m pip install --upgrade pip
# Change pypi source to accelerate library installation
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install fastapi==0.104.1
pip install uvicorn==0.24.0.post1pip install requests==2.25.1
pip install modelscope==1.9.5
pip install transformers==4.37.2
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1

```

## Model download

Use the snapshot_download function in modelscope to download the model. The first parameter is the model name, and the parameter cache_dir is the download path of the model.

Create a new download.py file in the /root/autodl-tmp path and enter the following content in it. Remember to save the file after pasting the code, as shown in the figure below. And run python /root/autodl-tmp/download.py to download. The model size is 12 GB. It takes about 10~15 minutes to download the model.

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('THUCoAI/CharacterGLM-6B', cache_dir='/root/autodl-tmp', revision='master')
```

## Code preparation

Create a new api.py file in the /root/autodl-tmp path and enter the following content in it. Remember to save the file after pasting the code. The following code has very detailed comments. If you have any questions, please raise an issue.

```python
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import datetime
import torch

# Set device parameters
DEVICE = "cuda" # Use CUDA
DEVICE_ID = "0" # CUDA device ID, empty if not set
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE # Combine CUDA device information

# Clean up GPU memory function
def torch_gc():
if torch.cuda.is_available(): # Check if CUDA is available
with torch.cuda.device(CUDA_DEVICE): # Specify CUDA device
torch.cuda.empty_cache() # Clear CUDA cache
torch.cuda.ipc_collect() # Collect CUDA memory fragments

# Create FastAPI application
app = FastAPI()

# Endpoint for handling POST requests
@app.post("/")
async def create_item(request: Request):
global model, tokenizer # Declare global variables to use models and tokenizers inside functions
json_post_raw = await request.json() # Get JSON data for POST request
json_post = json.dumps(json_post_raw) # Convert JSON data to a string
json_post_list = json.loads(json_post) # Convert a string to a Python object
prompt = json_post_list.get('prompt') # Get the prompt in the request
history = json_post_list.get('history') # Get the history in the request
max_length = json_post_list.get('max_length') # Get the maximum length in the request
top_p = json_post_list.get('top_p') # Get the top_p parameter in the request
temperature = json_post_list.get('temperature') # Get the temperature parameter in the request
session_meta = {'user_info': 'I am Lu Xingchen, a male, a well-known director, and a co-director of Su Mengyuan. I am good at shooting music-themed movies. Su Mengyuan's attitude towards me is respect, and regard me as a mentor and helpful friend. ', 'bot_info': 'Su Mengyuan, whose real name is Su Yuanxin, is a popular domestic female singer and actress. After participating in a talent show, she quickly became famous and entered the entertainment industry with her unique voice and outstanding stage charm. She looks beautiful, but her real charm lies in her talent and diligence. Su Mengyuan is an outstanding student who graduated from the Conservatory of Music. She is good at creation and has many popular original songs. In addition to her achievements in music, she is also keen on charity, actively participates in public welfare activities, and conveys positive energy with practical actions. At work, she is very dedicated to her work. When filming, she always devotes herself to the role, which has won praise from industry insiders and the love of fans. Although in the entertainment industry, she always maintains a low-key and humble attitude, and is deeply respected by her peers. When expressing, Su Mengyuan likes to use "we" and "together" to emphasize team spirit. ', 'bot_name': 'Su Mengyuan', 'user_name': 'Lu Xingchen'}
# Call the model to generate a dialogue
response, history = model.chat(
tokenizer,
session_meta,
prompt,
history=history,
max_length=max_length if max_lengthength else 2048, # If the maximum length is not provided, 2048 is used by default
top_p=top_p if top_p else 0.7, # If the top_p parameter is not provided, 0.7 is used by default
temperature=temperature if temperature else 0.95 # If the temperature parameter is not provided, 0.95 is used by default
)
now = datetime.datetime.now() # Get the current time
time = now.strftime("%Y-%m-%d %H:%M:%S") # Format the time as a string
# Build response JSON
answer = {
"response": response,
"history": history,
"status": 200,
"time": time
}
# Build log information
log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
print(log) # print log
torch_gc() # perform GPU memory cleanup
return answer # return response

# main function entry
if __name__ == '__main__':
# load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/THUCoAI/CharacterGLM-6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/THUCoAI/CharacterGLM-6B", trust_remote_code=True).to(torch.bfloat16).cuda()
model.eval() # set the model to evaluation mode
# start FastAPI application
# use port 6006 to map the autodl port to thisto use the api locally
uvicorn.run(app, host='0.0.0.0', port=6006, workers=1) # Start the application on the specified port and host
```

## Api deployment call

Enter the following command in the terminal to start the api service

```python

cd /root/autodl-tmp
python api.py

```

By default, it is deployed on port 6006 and is called via the POST method. You can use curl to call it, as shown below:

```python

curl -X POST "http://127.0.0.1:6006" \
-H 'Content-Type: application/json' \
-d '{"prompt": "你好", "history": []}'

```

The call example result is shown in the figure below

![image](https://github.com/suncaleb1/self-llm/assets/155936975/c5568fe4-ae2a-4679-b795-f3fe17458310)

You can also use the request in Pythons library, create a new api-requests.py file, and write the following code:

```python

import requests
import json

def get_completion(prompt):
headers = {'Content-Type': 'application/json'}
data = {"prompt": prompt, "history": []}
response = requests.post(url='http://127.0.0.1:6006', headers=headers, data=json.dumps(data))
return response.json()['response']

if __name__ == '__main__':
print(get_completion('Who are you?'))

```

Open a new terminal and enter the following command

```python
cd /root/autodl-tmp
python api-requests.py

```

The return value and result are shown below

```python
{
'response': 'Hi, hello, my name is Su Mengyuan. (Smiling and walking towards the other party)', 
'history': [['Who are you? ', 'Hi, hello, my name is Su Mengyuan. (Smiling and walking towards the other party)']], 
'status': 200, 
'time': '2024-03-05 22:44:35'
}
```

![image](https://github.com/suncaleb1/self-llm/assets/155936975/f14d739e-addf-4b1b-bcf3-da4d714130fe)