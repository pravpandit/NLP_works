# Yuan2.0-M32 WebDemo deployment

## Environment preparation

Rent an RTX 3090/24G graphics card machine in the Autodl platform. As shown in the figure below, select PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1.

![Open machine configuration selection](images/01-1.png)

Next, we open JupyterLab on the server we just rented, as shown in the figure below.

![Open JupyterLab](images/01-2.png)

Then open the terminal to start environment configuration, model download and run the demonstration. 

![Open terminal](images/01-3.png)

## Environment configuration

Yuan2-M32-HF-INT4 is a model quantized from the original Yuan2-M32-HF by auto-gptq.

Through model quantization, the requirements for video memory and hard disk for deploying Yuan2-M32-HF-INT4 will be significantly reduced.

Note: Since the pip version of auto-gptq does not currently support Yuan2.0 M32, it needs to be compiled and installed

```shell
# Upgrade pip
python -m pip install --upgrade pip

# Replace pipInstallation of ypi source acceleration library
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Pull Yuan2.0-M32 project
git clone https://github.com/IEIT-Yuan/Yuan2.0-M32.git

# Enter AutoGPTQ
cd Yuan2.0-M32/3rd_party/AutoGPTQ

# Install autogptq
pip install --no-build-isolation -e .

# Install einops modelscope streamlit
pip install einops modelscope streamlit==1.24.0
```

> Considering that some students may encounter some problems in configuring the environment, we have prepared a mirror of Yuan2.0-M32 on the AutoDL platform. Click the link below and directly create an Autodl example.
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/Yuan2.0-M32*** ##Model download 

Use the snapshot_download function in modelscope to download the model. The first parameter is the model name, and the parameter cache_dir is the model download path.

Here you can first enter the autodl platform and initialize the file storage in the corresponding area of ​​the machine. The file storage path is '/root/autodl-fs'.

The files in this storage will not be lost when the machine is shut down, which can avoid the model from being downloaded twice.

![autodl-fs](images/autodl-fs.png)

Then run the following code to execute the model download.

```python
from modelscope import snapshot_download
model_dir = snapshot_download('YuanLLM/Yuan2-M32-HF-INT4', cache_dir='/root/autodl-fs')
``` 

## Model merging

The downloaded model is multiple files, which need to be merged.

```shell cat /root/autodl-fs/YuanLLM/Yuan2-M32-HF-INT4/gptq_model-4bit-128g.safetensors*> /root/autodl-fs/YuanLLM/Yuan2-M32-HF-INT4/gptq_model-4bit-128g.safetensors
```

## Code preparation

Create a new file `chatBot.py` in the `/root/autodl-tmp` path and enter the following content in it. Remember to save the file after pasting the code. The following code has very detailed comments. If you have any questions, please raise an issue.

The chatBot.py code is as follows

```python
# Import the required libraries
from auto_gptq import AutoGPTQForCausalLM
from transformers import LlamaTokenizer
import torch
import streamlit as st

# Create a title and a link in the sidebar
with st.sidebar:
st.markdown("## Yuan2.0-M32 LLM")
"[Open Source Large Model Usage Guide self-llm](https://github.com/datawhalechina/self-llm.git)"
# CreateCreate a slider to select the maximum length, ranging from 0 to 1024, with a default value of 512
max_length = st.slider("max_length", 0, 1024, 512, step=1)

# Create a title and a subtitle
st.title("💬 Yuan2.0-M32 Chatbot")
st.caption("🚀 A streamlit chatbot powered by Self-LLM")

# Define the model path
path = '/root/autodl-fs/YuanLLM/Yuan2-M32-HF-INT4'

# Define a function to get the model and tokenizer
@st.cache_resource
def get_model():
print("Creat tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','< commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True) print("Creat model ...") model = AutoGPTQForCausalLM.from_quantized(path, trust_remote_code=True).cuda() return tokenizer, model # Load model and tokenizer tokenizer, model = get_model() # If there is no "messages" in session_state, create a list containing the default messages
if "messages" not in st.session_state:
st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Traverse all messages in session_state and display them on the chat interface
for msg in st.session_state.messages:
st.chat_message(msg["role"]).write(msg["content"])

# If the user enters content in the chat input box, do the following
if prompt := st.chat_input():
# Add the user's input to the messages list in session_state
st.session_state.messages.append({"role": "user", "content": prompt})

# Display the user's input on the chat interface
st.chat_message("user").write(prompt)

# Call the model
input_str = "<n>".join(msg["content"] for msg in st.session_state.messages) + "<sep>"
inputs = tokenizer(input_str, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=256)
output = tokenizer.decode(outputs[0])
response = output.split("<sep>")[-1].replace("<eod>", '')

# Add the output of the model to the messages list in session_state
st.session_state.messages.append({"role": "assistant", "content": response})

# Display the output of the model on the chat interface
st.chat_message("assistant").write(response)

# print(st.session_state)
```

# Configure vscode ssh

Copy the machine ssh login command

![](images/03-0.png)

Paste it to the .ssh/config of the local computer and modify it to the following format

![](images/03-1.png)

Then connect to this ssh and select linx

![](images/03-2.png)

Copy the password and enter it, press Enter to log in to the machine

## Run demo

Run the following command in the terminal to start the streamlit service

```shell
streamlit run chatBot.py --server.address 127.0.0.1 --server.port 6006
```

![](images/03-3.png)

Click to open in the browser to see the chat interface.

The running effect is as follows:

![](images/03-4.png)