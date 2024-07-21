# XVERSE-MoE-A4.2B Transformers deployment call
## XVERSE-MoE-A4.2B introduction
**XVERSE-MoE-A4.2B** is a large language model (Large Language Model) independently developed by Shenzhen Yuanxiang Technology that supports multiple languages. It uses a mixed expert model (MoE, Mixture-of-experts) architecture. The total parameter scale of the model is 25.8 billion, and the actual number of activated parameters is 4.2 billion. The open source model this time is the base model **XVERSE-MoE-A4.2B**, and its main features are as follows:

- **Model structure**: XVERSE-MoE-A4.2B is a decoder-only Transformer architecture, which expands the FFN layer of the dense model to an expert layer. Unlike the traditional MoE, where the size of each expert is the same as the standard FFN (such as Mixtral 8x7B), it uses more fine-grained experts, each of which is the size of the standard FFN. 1/4, and set up two categories of shared experts (Shared Expert) and non-shared experts (Non-shared Expert). Shared experts are always activated during calculations, and non-shared experts are selectively activated through Routers.
- **Training data**: 2.7 trillionThe model is fully trained with high-quality and diverse data of tokens, including more than 40 languages ​​such as Chinese, English, Russian, and Spanish. By finely setting the sampling ratio of different types of data, the performance of Chinese and English is excellent, and the effects of other languages ​​can also be taken into account; the model is trained using 8K training samples.
- **Training framework**: For the unique expert routing and weight calculation logic in the MoE model, in-depth customization and optimization are carried out, and a set of efficient fusion operators are developed to improve computing efficiency. At the same time, in order to solve the challenges of memory usage and large communication volume of the MoE model, the overlap processing method of computing, communication and CPU-Offload is designed to improve the overall throughput.

The model size, architecture, and learning rate of **XVERSE-MoE-A4.2B** are as follows:

| total params | activated params | n_layers | d_model | n_heads | d_ff | n_non_shared_experts | n_shared_experts | top_k | lr |
| :----------: | :--------------: | :------: | :-----: | :-----: | :--: | :--: | :------------------: | :--------------: | :---: | :----: |
| 25.8B | 4.2B | 28 | 2560 | 32 | 1728 | 64 | 2 | 6 | 3.5e−4 |

However, the XVERSE repository has not updated more practical cases, so it is still necessary for everyone to enrich it. I will share more cases when I have time.

For relevant reports on the XVERSE-MoE-A4.2B model, please see: [Yuanxiang's first MoE large model open source: 4.2B activation parameters, the effect is comparable to the 13B model](https://mp.weixin.qq.com/s/U_ihKmhRD6Xc0cZ8hMJ1SQ)

## Talk about video memory calculation
The consideration of video memory calculation will vary with different model types and tasks

The Transformers deployment call here is an inference task, so only model parameters, KV Cache, intermediate results and input data need to be considered. The model here is the MoE model, considering the complete model parameters (25.8B); using bf16 loading, and then considering the intermediate results and inputData and KV Cache, etc., require about `2x1.2x25.8` video memory, so we will choose three cards with a total of 72G video memory later. The video memory requirement is still quite large. You can try it according to your own conditions.

For a more complete video memory calculation, refer to this blog: [[Transformer Basic Series] Manual Video Memory Usage](https://zhuanlan.zhihu.com/p/648924115)
## Environment Preparation
Rent a **three-card 3090 and other 24G (total 72G) video memory** machine on the autodl platform, as shown in the figure below, select PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1
Next, open the JupyterLab of the server you just rented, image and open the terminal in it to start environment configuration, model download and run demonstration. 
![Alt ​​text](images/1.png)
pip source change and installation of dependent packages
```shell
# Because it involves accessing github, it is best to open the academic mirror acceleration of autodl
source /etc/network_turbo
# Upgrade pip
python -m pip install --upgrade pip
# Change the pypi source to accelerate the installation of the library
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# Install the new version containing XVERSE-MoE from the transformers github repository
# If you can't install it, you can use pip install git+https://github.moeyy.xyz/https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/transformers
# Install the required python packages
pip install modelscope sentencepiece accelerate fastapi uvicorn requests streamlit transformers_stream_generator
# Install flash-attention
# This also doesn't work, use pip install https://github.moeyy.xyz/https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
## Model download
Use ModelScope to download the model
```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('xverse/XVERSE-MoE-A4.2B', cache_dir='/root/autodl-tmp', revision='master')
```
## CodePreparation
Create a new trains.py file in the /root/autodl-tmp path and enter the following content in it
```python
import torch # Import the torch library for deep learning related operations
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig # The three classes are used to load the tokenizer, load the causal language model, and load the generation configuration respectively

# Set the model path to the model path just downloaded
model_name = "/root/autodl-tmp/xverse/XVERSE-MoE-A4.2B"

# Load the language model, set the data type to bfloat16, that is, mixed precision format to optimize performance and reduce video memory usage, and set the inference device to `auto` to automatically select the best device for inference. If there is no available GPU, it may fall back to the CPU
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# Load the tokenizer
tokenizer =AutoTokenizer.from_pretrained(model_name)

# Define input string
prompt = "Give me a short introduction to large language model."
messages = [
{"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": prompt}
]
# Use the apply_chat_template method of the tokenizer to process messages and convert the format
text = tokenizer.apply_chat_template(
messages,
tokenize=False,
add_generation_prompt=True # Add generation prompt before the message
)
# Convert the text in the text variable to the model input format and specify that the returned tensor is a PyTorch tensor ("pt")
model_inputs = tokenizer([text], return_tensors="pt").to(device)
# Generate text using the model's generate method
generated_ids = model.generate(
model_inputs.input_ids,
max_new_tokens=512
)
# Extract the newly generated tokens from the generated IDs in addition to the original input
generated_ids = [
output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
# Convert the generated token IDs back to text using the tokenizer's batch_decode method
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# Display the generated answer
print(response)
```