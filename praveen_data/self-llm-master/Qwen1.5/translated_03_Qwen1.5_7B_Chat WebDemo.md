# Qwen1.5-4B-Chat WebDemo Deployment

## Qwen1.5 Introduction

Qwen1.5 is a beta version of Qwen2. Qwen1.5 is a transformer-based decoder-only language model that has been pre-trained on a large amount of data. Compared with the previously released Qwen, the improvements of Qwen1.5 include 6 model sizes, including 0.5B, 1.8B, 4B, 7B, 14B, and 72B; the performance of the Chat model in terms of human preferences has been significantly improved; both the base model and the chat model support multiple languages; all sizes of models stably support 32K context length without trust_remote_code.

## Environment preparation
Rent a 3090 or other 24G graphics card machine on the autodl platform. As shown in the figure below, select PyTorch-->2.0.0-->3.8(ubuntu20.04)-->11.8 (versions above 11.3 are acceptable)
Next, open JupyterLab on the server you just rented, image and open the terminal to start environment configuration, model download and run demonstration. 
![Alt ​​text](images/Qwen2-Web1.png)
Pip source change and installation of dependent packages
```
# Upgrade pip
python -mpip install --upgrade pip
# Change the pypi source acceleration library installation
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.9.5
pip install "transformers>=4.37.0"
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
pip install transformers_stream_generator==0.0.4
```

> Considering that some students may encounter some problems in configuring the environment, we have prepared an environment image of Qwen1.5 on the AutoDL platform, which is applicable to all deployment environments of this warehouse except Qwen-GPTQ and vllm. Click the link below and create an Autodl example directly.
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/self-llm-Qwen1.5***

## Model download
Use the snapshot_download function in modelscope to download the model. The first parameter is the model name, and the parameter cache_dir is the download path of the model.

Create a new download.py file in the /root/autodl-tmp path and enter the following content in it. Remember to save the file after pasting the code, as shown in the figure below. And run python /root/autodl-tmp/download.py to execute the download. It takes about 2 minutes to download the model.

```
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
from modelscope import GenerationConfig
model_dir = snapshot_download('qwen/Qwen1.5-7B-Chat', cache_dir='/root/autodl-tmp', revision='master')
```

## Code preparation

Create a new `chat in the `/root/autodl-tmp` pathBot.py` file and enter the following content in it. Remember to save the file after pasting the code. The code below has very detailed comments. If you don't understand anything, please raise an issue.

```python
# Import the required libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import streamlit as st

# Create a title and a link in the sidebar
with st.sidebar:
st.markdown("## Qwen1.5 LLM")
"[Open Source Large Model Eating Guide self-llm](https://github.com/datawhalechina/self-llm.git)"
# Create a slider for selecting the maximum length, ranging from 0 to 1024, with a default value of 512
max_length = st.slider("max_length", 0, 1024, 512, step=1)

# Create a title and a subtitle
st.title("💬 Qwen1.5 Chatbot")st.caption("🚀 A streamlit chatbot powered by Self-LLM")

# Define model path
mode_name_or_path = '/root/autodl-tmp/qwen/Qwen1.5-7B-Chat'

# Define a function to get the model and tokenizer
@st.cache_resource
def get_model():
# Get tokenizer from pre-trained model
tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False)
# Get model from pre-trained model and set model parameters
model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")

return tokenizer, model

# Load Qwen1.5-4B-Chat model and tokenizerenizer
tokenizer, model = get_model()

# If there is no "messages" in session_state, create a list containing default messages
if "messages" not in st.session_state:
st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Traverse all messages in session_state and display them on the chat interface
for msg in st.session_state.messages:
st.chat_message(msg["role"]).write(msg["content"])

# If the user enters content in the chat input box, do the following
if prompt := st.chat_input():
# Add the user's input to the messages list in session_state
st.session_state.messages.append({"role": "user", "content":prompt}) # Display the user's input on the chat interface st.chat_message("user").write(prompt) # Build input input_ids = tokenizer.apply_chat_template(st.session_state.messages,tokenize=False,add_generation_prompt=True) model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda') generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512) generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip (model_inputs.input_ids, generated_ids) ]response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# Add the output of the model to the messages list in session_state
st.session_state.messages.append({"role": "assistant", "content": response})
# Display the output of the model on the chat interface
st.chat_message("assistant").write(response)
# print(st.session_state)
```

## Run demo

Run the following command in the terminal to start the streamlit service, map the port to the local according to the instructions of `autodl`, and then open the link http://localhost:6006/ in the browser to see the chat interface.

```bash
streamlit run /root/autodl-tmp/chatBot.py --server.address 127.0.0.1 --server.port 6006
```

As shown below:

![Alt ​​text](images/Qwen2-Web2.png)