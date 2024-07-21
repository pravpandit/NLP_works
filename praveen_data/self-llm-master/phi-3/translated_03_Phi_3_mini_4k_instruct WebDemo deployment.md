# Phi-3-mini-4k-instruct WebDemo deployment

## Environment preparation

Rent a 3090 or other 24G graphics card machine in the autodl platform. As shown in the figure below, select `PyTorch-->2.1.0-->3.10(ubuntu20.04)-->12.1`

![alt text](./assets/03-1.png)
Next, open JupyterLab on the server you just rented, and open the terminal in it to start environment configuration, model download and run demo.

pip changes the source to speed up downloading and installing dependent packages

```shell
# Upgrade pip
python -m pip install --upgrade pip
# Change the pypi source to speed up library installation
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.11.0
pip install langchain==0.1.15
pip install "transformers>=4.40.0" accelerate tiktoken einops scipy transformers_stream_generator==0.1.16
pip install streamlit
``` 
> Considering that some students may encounter some problems in configuring the environment, we have prepared an environment image of Phi-3 on the AutoDL platform, which is applicable to all deployment environments of this warehouse. Click the link below and create an Autodl example directly.
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/phi-3-mini-4k-instruct-webdemo***

## Model download

Use the snapshot_download function in modelscope to download the model. The first parameter is the model name, and the parameter cache_dir is the download path of the model.

Create a new model_download.py file in the /root/autodl-tmp path and enter the following content in it. Remember to save the file after pasting the code, as shown in the figure below. And run `python /root/autodl-tmp/model_download.py` to execute the download. The model size is 15 GB. DownloadIt takes about 2 minutes to load the model.

```python 
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('LLM-Research/Phi-3-mini-4k-instruct', cache_dir='/root/autodl-tmp', revision='master')
```

## Code preparation

Create a new file `chatBot.py` in the `/root/autodl-tmp` path and enter the following content in it. Remember to save the file after pasting the code. The following code has very detailed comments. If you have any questions, please raise an issue.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

# Create a title and a link in the sidebar
with st.sidebar:
st.markdown("## Phi-3 LLM")
"[Open Source Large Model Usage Guide self-llm](https://github.com/datawhalechina/self-llm.git)"

# Create a title and a subtitle
st.title("💬 Phi-3 Chatbot")
st.caption("🚀 A streamlit chatbot powered by Self-LLM")

# Define the model path
mode_name_or_path = '/root/autodl-tmp/LLM-Research/Phi-3-mini-4k-instruct'

# Define a function to get the model and tokenizer
@st.cache_resource
def get_model():
# Get the tokenizer from the pre-trained model
tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# Get the model from the pre-trained model and set the model parameters
model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

return tokenizer, model

def bulid_input(prompt, history=[]):
system_format='<s><|system|>\n{content}<|end|>\n'
user_format='<|user|>\n{content}<|end|>\n'
assistant_format='<|assistant|>\n{content}<|end|>\n'
history.append({'role':'user','content':prompt})
prompt_str = ''
# Splice historical dialogue
for item in history:if item['role']=='user':
prompt_str+=user_format.format(content=item['content'])
else:
prompt_str+=assistant_format.format(content=item['content'])
return prompt_str + '<|assistant|>\n'

# Load Phi-3 model and tokenizer
tokenizer, model = get_model()

# If there is no "messages" in session_state, create a list containing default messages
if "messages" not in st.session_state:
st.session_state["messages"] = []

# Traverse all messages in session_state and display them on the chat interface
for msg in st.session_state.messages:
st.chat_message(msg["role"]).write(msg["content"])

# If the user enters content in the chat input box, do the following
if prompt := st.chat_input():

# Display the user's input on the chat interface
st.chat_message("user").write(prompt)

# Build input
input_str = bulid_input(prompt=prompt, history=st.session_state["messages"])
input_ids = tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').cuda()
outputs = model.generate(
input_ids=input_ids, max_new_tokens=512, do_sample=True,
top_p=0.9, temperature=0.5, repetition_penalty=1.1, eos_token_id=tokenizer.encode('<|endoftext|>')[0]
)
outputs = outputs.tolist()[0][len(input_ids[0]):]

response = tokenizer.decode(outputs)

response = response.split('<|end|>')[0]
st.session_state.messages.append({"role": "assistant", "content": response})
# Display the output of the model on the chat interface
st.chat_message("assistant").write(response)
# Output the content in the current session_state for subsequent debugging
print(st.session_state)
```

## Run demo

Run the following command in the terminal to start the streamlit service and map the port according to the instructions of `autodl`Shoot to the local machine, then open the link http://localhost:6006/ in the browser to see the chat interface.

```bash
streamlit run /root/autodl-tmp/chatBot.py --server.address 127.0.0.1 --server.port 6006
```

Port mapping: Replace `xxxx` with the port number corresponding to your container instance
```bash
ssh -CNg -L 6006:127.0.0.1:6006 root@connect.westc.gpuhub.com -p xxxx
```