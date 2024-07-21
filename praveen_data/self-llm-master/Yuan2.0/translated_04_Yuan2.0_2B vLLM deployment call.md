# Yuan 2.0 inference service deployment based on vLLM

## 1. Configure vLLM environment
Environment requirements: torch2.1.2 cuda12.1

vLLM environment configuration is mainly divided into the following two steps: pull Yuan-2.0 project and install vllm runtime environment

Note: Since the pip version vllm does not currently support Yuan 2.0, it needs to be compiled and installed

### Step 1. Pull Yuan-2.0 project

```shell
# Pull project
git clone https://github.com/IEIT-Yuan/Yuan-2.0.git
```

### Step 2. Install vLLM runtime environment

```shell
# Enter vLLM project
cd Yuan-2.0/3rdparty/vllm

# Install dependencies
pip install -r requirements.txt

# Install setuptools
# vllm has requirements for the version of setuptools, refer to https://github.com/vllm-project/vllm/issues/4961
vim pyproject.toml # Modify to setuptools == 69.5.1
pipinstall setuptools == 69.5.1

# Install vllm
pip install -e .
```

## 2. Yuan2.0-2B model reasoning and deployment based on vLLM

The following is an example of how to use the vLLM reasoning framework to reason and deploy the Yuan2.0-2B model

### Step 1. Model download

Use the snapshot_download function in modelscope to download the model. The first parameter is the model name, and the parameter cache_dir is the model download path.

Here you can first enter the autodl platform and initialize the file storage in the corresponding area of ​​the machine. The file storage path is '/root/autodl-fs'.

The files in this storage will not be lost when the machine is shut down, which can avoid the model from being downloaded twice.

![autodl-fs](images/autodl-fs.png)

Then run the following code to execute the model download. The model size is 4.5GB, and it takes about 5 minutes to download.

```python from modelscope import snapshot_download model_dir = snapshot_download('YuanLLM/Yuan2-2B-Mars-hf', cache_dir='/root/autodl-fs')
```

### Step 2. Reasoning Yuan2.0-2B based on vllm

To reason Yuan2.0-2B based on vllm, you first need to load the model and then perform reasoning

#### 1. Load the model

```python
from vllm import LLM, SamplingParams
import time

# Configuration parameters
sampling_params = SamplingParams(max_tokens=300, temperature=1, top_p=0, top_k=1, min_p=0.0, length_penalty=1.0, repetition_penalty=1.0, stop="<eod>", )

# Load model
llm = LLM(model="/root/autodl-fs/YuanLLM/Yuan2-2B-Mars-hf", trust_remote_code=True)
```

#### 2. Reasoning
Reasoning supports single prompt and multiple prompts

##### Option 1. Single prompt reasoningReasoning

```python
prompts = ["Give me a python code to print helloword <sep>"]

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()

for output in outputs:
prompt = output.prompt
generated_text = output.outputs[0].text
print("Prompt:", prompt)
print("Generated text:", generated_text)
print()

print("inference_time:", (end_time - start_time))
```

##### Option 2. Multiple prompt reasoning

```python
prompts = ["Give me a python code to print helloword <sep>", "Give me a c++ code to print helloword <sep>>"]

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()

for output in outputs:
prompt = output.prompt
generated_text = output.outputs[0].text
print("Prompt:", prompt)
print("Generated text:", generated_text)
print()

print("inference_time:", (end_time - start_time))
```

### Step 3. Deploy Yuan2.0-2B based on vllm.entrypoints.api_server
The steps to deploy Yuan2.0-2B based on api_server include initiating and calling the inference service

#### 1. Service initiation

```shell 
# Please run the following command in the command line, not directly in jupyter!python
python -m vllm.entrypoints.api_server --model=/root/autodl-fs/YuanLLM/Yuan2-2B-Mars-hf --trust-remote-code
```

![](images/05-0.png)

#### 2. Service call
There are two ways to call a service: the first is to call it directly through the command line; the second is to call it in batches by running a script.

##### Option 1. Call the service based on the command line

```shell
!curl http://localhost:8000/generate -d '{"prompt": "Give me a python code to print helloword <sep>", "use_beam_search": false, "n": 1, "temperature": 1, "top_p": 0, "top_k": 1, "max_tokens":256, "stop": "<eod>"}'
```

##### Option 2. Call the service based on the command script

```python
import requests
import json

prompt = "Give me a python code to print helloword<sep>" raw_json_data = { "prompt": prompt, "logprobs": 1, "max_tokens": 256, "temperature": 1, "use_beam_search": False, "top_p ": 0, "top_k": 1, "stop": "<eod>", } json_data = json.dumps(raw_json_data) headers = { "Content-Type": "application/json", } response = requests.post (f'http://localhost:8000/generate', data=json_data, headers=headers)
output = response.text
output = json.loads(output)
print(output)
```

### Step 4. Deploy Yuan2.0-2B based on vllm.entrypoints.openai.api_server
The steps to deploy Yuan2.0-2B based on openai's api_server are similar to those in step 3. The methods of initiating and calling services are as follows:

#### 1. Service initiation

```shell 
# Please run the following command in the command line, not directly in jupyter!python
python -m vllm.entrypoints.openai.api_server --model=/root/autodl-fs/YuanLLM/Yuan2-2B-Mars-hf --trust-remote-code
```

![](images/05-1.png)

#### 2. Service call
There are two ways to call a service: the first is to call it directly through the command line; the second is to call it in batches by running a script.

##### Option 1. Call the service based on the command line

```shell
!curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/root/autodl-fs/YuanLLM/Yuan2-2B-Mars-hf", "prompt": "Give me a python code to print helloword <sep>", "max_tokens": 300, "temperature": 1, "top_p": 0, "top_k": 1, "stop": "<eod>"}'
```

##### Option 2. Calling services based on command scripts

```python
import requests
import json

prompt = "Give me a python code to print helloword <sep>"
raw_json_data = {
"model": "/root/autodl-fs/YuanLLM/Yuan2-2B-Mars-hf",
"prompt": prompt,
"max_tokens": 256, "temperature": 1, "use_beam_search": False, "top_p": 0, "top_k": 1, "stop": "<eod>", } json_data = json.dumps(raw_json_data, ensure_ascii=True) headers = { "Content-Type": "application/json", } response = requests.post(f'http://localhost:8000/v1/completions', data=json_data, headers=headers) output = response.text output = json.loads(output) print(output) ```