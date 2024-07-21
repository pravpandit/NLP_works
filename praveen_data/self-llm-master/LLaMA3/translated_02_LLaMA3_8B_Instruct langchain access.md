# LLaMA3-8B-Instruct langchain access

## Environment preparation

Rent a 3090 or other 24G video memory graphics card machine in the Autodl platform, as shown in the following figure, select the image `PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1`

Next, open JupyterLab on the server you just rented, and open the terminal in it to start environment configuration, model download and run demonstration. 

![Open machine configuration selection](images/autodl_config.png)

pip changes the source to speed up downloading and installing dependent packages

```shell
# Upgrade pip
python -m pip install --upgrade pip
# Change the pypi source to speed up the installation of the library
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.11.0
pip install langchain==0.1.15
pip install "transformers>=4.40.0" accelerate tiktoken einops scipy transformers_stream_generator==0.1.16
pip install -U huggingface_hub
``` 

> Considering that some students may encounter some problems in configuring the environment, we have prepared an environment image of LLaMA3 on the AutoDL platform, which is applicable to all deployment environments of this repository. Click the link below and create an Autodl example directly.
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/self-llm-LLaMA3***

## Model download

Use the snapshot_download function in modelscope to download the model. The first parameter is the model name, and the parameter cache_dir is the download path of the model.

Create a new model_download.py file in the /root/autodl-tmp path and enter the following content in it. Remember to save the file after pasting the code, as shown in the figure below. And run `python /root/autodl-tmp/model_download.py` to download the model. The model size is 15 GB.It takes about 2 minutes to download the model.

```python 
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
```

## Code preparation

To build LLM applications conveniently, we need to customize an LLM class based on the locally deployed LLaMA3_LLM and connect LLaMA3 to the LangChain framework. After completing the customized LLM class, the LangChain interface can be called in a completely consistent way without considering the inconsistency of the underlying model call.

Customizing the LLM class based on the local deployment of LLaMA3 is not complicated. We only need to inherit a subclass from the LangChain.llms.base.LLM class and rewrite the constructor and _call function:

```python
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLaMA3_LLM(LLM):
# Customize LLM class based on local llama3
tokenizer: AutoTokenizer = None
model: AutoModelForCausalLM = None

def __init__(self, mode_name_or_path :str):

super().__init__()
print("Loading model from local...")
self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False) self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16, device_map="auto") self.tokenizer.pad_token = self.tokenizer.eos_token print("Complete loading of local model") def bulid_input(self, prompt, history=[]): user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>' assistant_format='<|start_header_id|>assistant<| end_header_id|>\n\n{content}<|eot_id|>' history.append({'role':'user','content':prompt}) prompt_str = '' # Splice historical dialogue for item in history: if item['role']=='user': prompt_str+=user_format.format(content=item[ 'content']) else: prompt_str+=assistant_format.format(content=item['content']) return prompt_str def _call(self, prompt : str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun ] = None, **kwargs:Any): input_str = self.bulid_input(prompt=prompt) input_ids = self.tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').to(self.model.device) outputs = self.model.generate( input_ids=input_ids, max_new_tokens=512, do_sample=True, top_p=0.9, temperature=0.5, repetition_penalty=1.1, eos_token_id=self.tokenizer.encode('<|eot_id|>')[0] ) outputs = outputs.tolist( )[0][len(input_ids[0]):] response = self.tokenizer.decode(outputs).strip().replace('<|eot_id|>', "").replace('<|start_header_id|>assistant<|end_header_id|>\n\n', '').strip()
return response

@property
def _llm_type(self) -> str:
return "LLaMA3_LLM"
```

In the above class definition, we rewrite the constructor and _call function respectively: for the constructor, we load the locally deployed LLaMA3 model at the beginning of the object instantiation to avoid the long time caused by reloading the model for each call; the _call function is the core function of the LLM class, and LangChain will call this function to call LLM. In this function, we call the generate method of the instantiated model to call the model and return the call result.

In the overall project, we encapsulate the above code as LLM.py, and then directly introduce the custom LLM class from this file.

```python
fromLLM import LLaMA3_LLM llm = LLaMA3_LLM(mode_name_or_path = "/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct") llm("Who are you") ```` ![alt text](. /images/image-2.png)