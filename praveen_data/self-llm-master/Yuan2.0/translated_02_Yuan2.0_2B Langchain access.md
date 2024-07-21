# Yuan2.0-2B connects to LangChain to build a knowledge base assistant 

## Environment preparation

Rent an RTX 3090/24G graphics card machine on the Autodl platform. As shown in the figure below, select PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1.

![Open machine configuration selection](images/01-1.png)

Next, we open JupyterLab on the server we just rented, as shown in the figure below.

![Open JupyterLab](images/01-2.png)

Then open the terminal to start environment configuration, model download and run the demonstration. 

![Open terminal](images/01-3.png)

## Environment configuration

pip source change to speed up downloading and installing dependent packages

```shell
# Upgrade pip
python -m pip install --upgrade pip

# Change pypi source to speed up library installation
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install langchain modelscope
pipinstall langchain modelscope
``` 

> Considering that some students may encounter some problems in configuring the environment, we have prepared a Yuan2.0 image on the AutoDL platform. Click the link below and directly create an Autodl example.
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/Yuan2.0***

## Model download 

Use the snapshot_download function in modelscope to download the model. The first parameter is the model name, and the parameter cache_dir is the download path of the model.

Here you can first enter the autodl platform and initialize the file storage in the corresponding area of ​​the machine. The file storage path is '/root/autodl-fs'.

The files in this storage will not be lost when the machine is shut down, which can avoid the model from being downloaded twice.

![autodl-fs](images/autodl-fs.png)

Then run the following code to execute the model download. The model size is 4.5GB, and it takes about 5 minutes to download.

```python from modelscope import snapshot_download model_dir =snapshot_download('YuanLLM/Yuan2-2B-Mars-hf', cache_dir='/root/autodl-fs')
```

## Code preparation

To build LLM applications conveniently, we need to customize an LLM class based on the locally deployed Yuan2 and connect Yuan2 to the LangChain framework.

After completing the customized LLM class, the LangChain interface can be called in a completely consistent manner without considering the inconsistency of the underlying model call.

Customizing the LLM class based on the locally deployed Yuan2 is not complicated. We only need to inherit a subclass from the LangChain.llms.base.LLM class and rewrite the constructor and _call function:

```python
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import LlamaTokenizer, AutoModelForCausalLM
import torch

class Yuan2_LLM(LLM):
# Customize LLM class based on local Yuan2
tokenizer: LlamaTokenizer = None
model: AutoModelForCausalLM = None

def __init__(self, mode_name_or_path:str):
super().__init__()

# Load pre-trained tokenizer and model
print("Creat tokenizer...")
self.tokenizer = LlamaTokenizer.from_pretrained(mode_name_or_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text> ','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True) print("Creat model...") self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16 , trust_remote_code=True).cuda() def _call(self, prompt : str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None,
**kwargs: Any):

prompt += "<sep>"
inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
outputs = self.model.generate(inputs,do_sample=False,max_length=4000)
output = self.tokenizer.decode(outputs[0])
response = output.split("<sep>")[-1]

return response

@property
def _llm_type(self) -> str:
return "Yuan2_LLM"
```

In the above class definition, we rewrite the constructor and _call function respectively: For the constructor, we load the locally deployed one at the beginning of the object instantiation.Yuan2 model, so as to avoid the long time of reloading the model for each call; _call function is the core function of LLM class, LangChain will call this function to call LLM, in which we call the generate method of the instantiated model to call the model and return the call result.

In the overall project, we encapsulate the above code as LLM.py, and then directly introduce the custom LLM class from this file.

## Call

Then you can use it like any other langchain large model function.

```python
from LLM import Yuan2_LLM
llm = Yuan2_LLM('/root/autodl-fs/YuanLLM/Yuan2-2B-Mars-hf')
print(llm("Who are you"))
```

![alt text](./images/02-0.png)