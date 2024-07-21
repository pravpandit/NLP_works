### Model download

#### hugging face

Use the `huggingface-cli` command line tool provided by `huggingface`. Installation dependencies:

```shell
pip install -U huggingface_hub
```

Then create a new python file, fill in the following code, and run it.

- resume-download: breakpoint resume
- local-dir: local storage path. (In Linux environment, you need to fill in the absolute path)

```python
import os

# Download model
os.system('huggingface-cli download --resume-download internlm/internlm-chat-7b --local-dir your_path')
```

#### hugging face mirror download

Same as using hugginge face to download, just fill in the mirror address. Use the `huggingface-cli` command line tool provided by `huggingface`. Installation dependencies:

```shell
pip install -U huggingface_hub
```Then create a new python file, fill in the following code, and run it.

- resume-download: breakpoint resume
- local-dir: local storage path. (Absolute path is required in Linux environment)

```python
import os

# Set environment variables
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Download model
os.system('huggingface-cli download --resume-download internlm/internlm-chat-7b --local-dir your_path')

```

For more information about mirror usage, please visit [HF Mirror](https://hf-mirror.com/) for more information.

#### modelscope

Use the `snapshot_download` function in `modelscope` to download the model. The first parameter is the model name, and the parameter `cache_dir` is the download path of the model.

Note: `cache_dir` is best to be an absolute path.

Installation dependencies:

```shell
pip install modelscope
pip install transformers
```

Create a new python file in the current directory, fill in the following code, and run it.

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='your path', revision='master')
```

#### git-lfs

Go to the [git-lfs](https://git-lfs.com/) website to download the installation package, and then install `git-lfs`. After installation, enter `git lfs install` in the terminal, and then you can use `git-lfs` to download the model. Of course, this method requires you to have a little bit of **Magic**.

```shell
git clone https://huggingface.co/internlm/internlm-7b
```#### Openxlab

Openxlab can directly download the model weight file by specifying the address of the model repository, the name of the file to be downloaded, the location where the file needs to be downloaded, etc.

To download the model using the python script, you must first install the dependencies. The installation code is as follows: `pip install -U openxlab` After the installation is complete, use the download function to import the model in the model center.

```python
from openxlab.model import download
download(model_repo='OpenLMLab/InternLM-7b', model_name='InternLM-7b', output='your local path')
```