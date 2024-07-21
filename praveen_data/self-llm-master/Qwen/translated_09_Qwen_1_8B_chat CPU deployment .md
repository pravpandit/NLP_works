# 09-Qwen-1_8B-chat CPU Deployment

## Overview

This article describes the process of deploying the Qwen 1.8B model on Intel devices. You need a machine with at least 16GB of memory to complete this task. We will use Intel's BigDL library for inference of large models to complete the entire process.

Bigdl-llm is an acceleration library for running LLM (Large Language Model) on Intel devices. It uses INT4/FP4/INT8/FP8 precision quantization and architecture-specific optimization to achieve low resource usage and high-speed inference capabilities of large models on Intel CPUs and GPUs (applicable to any PyTorch model).

For the sake of generality, this article only involves CPU-related code. If you want to learn how to deploy large models on Intel GPUs, you can refer to the [official documentation](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html).

## Environment configuration

Before we start, we need to prepare bigdl-llm and the related operating environment for subsequent deployment. We recommend that you use Python 3.9Perform the following operations.

If you find that the download speed is too slow, you can try to change the default mirror source: `pip config set global.index-url https://pypi.doubanio.com/simple`

```python
%pip install --pre --upgrade bigdl-llm[all] 
%pip install gradio 
%pip install hf-transfer
%pip install transformers_stream_generator einops
%pip install tiktoken
```

## Model download

First, we use huggingface-cli to obtain the qwen-1.8B model, which takes a long time and needs to wait for a while; here, considering the domestic download restrictions, we add environment variables to speed up the download.

```python
import os

# Set environment variables
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Download model
os.system('huggingface-cli download --resume-download qwen/Qwen-1_8B-Chat --local-dir qwen18chat_src')
```

## Save quantized model

In order to achieve low resource consumption inference of large language models, we first need to quantize the model to int4 precision, and then serialize and save it in the corresponding local folder for repeated loading and inference; using the `save_low_bit` api, we can easily achieve this step.

```python
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import os
if __name__ == '__main__':
model_path = os.path.join(os.getcwd(),"qwen18chat_src")
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.save_low_bit('qwen18chat_int4')
tokenizer.save_pretrained('qwen18chat_int4')
```

## Load quantized model

After saving the int4 model file, we can load it into memory for further reasoning; if you cannot export the quantized model on your local machine, you can also save the model in a machine with larger memory and then transfer it to a small memory end device for running. Most commonly used home PCs can meet the resource requirements for the actual operation of the int4 model.

```python
import torch
import time
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

QWEN_PROMPT_FORMAT = "<human>{prompt} <bot>"
load_path = "qwen18chat_int4"
model = AutoModelForCausalLM.load_low_bit(load_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
input_str = "Tell me a story about a young man who struggled to start a business and finally succeeded"
with torch.inference_mode():
prompt = QWEN_PROMPT_FORMAT.format(prompt=input_str)
input_ids = tokenizer.encode(prompt, return_tensors="pt")
st = time.time()
output = model.generate(input_ids,
max_new_tokens=512)
end = time.time()
output_str = tokenizer.decode(output[0], skip_special_tokens=True)
print(f'Inference time: {end-st} s')
print('-'*20, 'Prompt', '-'*20)
print(prompt)
print('-'*20, 'Output', '-'*20)
print(output_str)
```

## gradio-demo experience

In order to get a better multi-round dialogue experience, a simple `gradio` demo interface is also provided for debugging. You can modify the built-in `system` information or even fine-tune the model to make the local model closer to the large model requirements you envision.

```python
import gradio as gr
import time
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

QWEN_PROMPT_FORMAT = "<human>{prompt} <bot>" load_path = "qwen18chat_int4" model = AutoModelForCausalLM.load_low_bit(load_path, trust_remote_code=True) tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True) def add_text(history, text): _, history = model.chat(tokenizer, text, history=history) return history, gr.Textbox(value="", interactive=False) def bot(history): response = history[-1][1] history[-1][1] = "" for character in response: history[-1][1] += character time.sleep(0.05) yield history with gr.Blocks() as demo: chatbot = gr.Chatbot( [], elem_id="chatbot", bubble_full_width=False, ) with gr.Row(): txt = gr.Textbox( scale=4, show_label=False, placeholder="Enter text and press enter", container=False, ) txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then( bot, chatbot, chatbot, api_name="bot_response" )txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

demo.queue()
demo.launch()
```

Using Intel's large language model inference framework, we can achieve high-performance inference of large models on Intel devices. Only 2G memory is required to achieve smooth dialogue with the local large model. Let's experience it together.