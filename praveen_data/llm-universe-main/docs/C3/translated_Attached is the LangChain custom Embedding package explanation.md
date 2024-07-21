# Embedding package explanation
The corresponding source code of this article is here (https://github.com/datawhalechina/llm-universe/blob/main/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93/%E9%99%84LangChain%E8%87%AA%E5%AE%9A%E4%B9%89Embedding%E5%B0%81%E8%A3%85%E8%AE%B2%E8%A7%A3.ipynb). If you need to reproduce, you can download and run the source code. 
LangChain provides an efficient development framework for developing custom applications based on LLM, which makes it easy for developers to quickly stimulate the powerful capabilities of LLM and build LLM applications. LangChain also supports Embeddings of multiple large models, and has built-in calling interfaces for Embeddings of large models such as OpenAI and LLAMA. However, LangChain does not have all large models built-in. It provides strong scalability by allowing users to customize Embeddings types.

In this section, we take Zhipu AI as an example to describe how to customize E based on LangChain.mbeddings.

This section involves relatively more technical details of LangChain and large model calls. Students with energy can learn deployment. If you don’t have energy, you can directly use the subsequent code to support calls.

To implement custom Embeddings, you need to define a custom class that inherits from the Embeddings base class of LangChain, and then define two functions: ① embed_query method, which is used to embed a single string (query); ② embed_documents method, which is used to embed a string list (documents).

First, we import the required third-party libraries:

```python
from __future__ import annotations

import logging
from typing import Dict, List, Any

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)
```

Here we define a custom Embeddings class that inherits from the Embeddings class:

```python
class ZhipuAIEmbeddings(BaseModel, Embeddings):
"""`Zhipuai Embeddings` embedding models."""

client: Any
"""`zhipuai.ZhipuAI"""
```

In Python, root_validator is a decorator function in the Pydantic module for custom data validation. root_validator is used to perform custom validation on the entire data model before validating the entire data model to ensure that all data conforms to the expected data structure.

root_validator receives a function as a parameter that contains the logic to be validated. The function should return a dictionary containing the validated data. If the validation fails, a ValueError exception is thrown.

Here we just need to configure `ZHIPUAI_API_KEY` in the `.env` file, and `zhipuai.ZhipuAI` will automatically obtain `ZHIPUAI_API_KEY`Y`。

```python
@root_validator()
def validate_environment(cls, values: Dict) -> Dict:
"""
Instantiate ZhipuAI as values["client"]

Args:

values ​​(Dict): A dictionary containing configuration information, which must contain the client field.
Returns:

values ​​(Dict): A dictionary containing configuration information. If there is a zhipuai library in the environment, the instantiated ZhipuAI class will be returned; otherwise, an error 'ModuleNotFoundError: No module named 'zhipuai'' will be reported.
"""
from zhipuai import ZhipuAI
values["client"] = ZhipuAI()
return values
```

`embed_query` is a method for calculating embedding for a single text (str). Here we override this method and call `ZhipuAI` instantiated when validating the environment to call the remoteAPI and returns the embedding result.

```python
def embed_query(self, text: str) -> List[float]:
"""
Generates the embedding of the input text.

Args:
texts (str): The text to generate the embedding.

Return:
embeddings (List[float]): The embedding of the input text, a list of floating point values.
"""
embeddings = self.client.embeddings.create(
model="embedding-2",
input=text
)
return embeddings.data[0].embedding
```

embed_documents is a method for calculating the embedding of a string list (List[str]). For this type of input, we take a loop to calculate the embedding of the substrings in the list one by one and return it.

```python
def embed_documents(self, texts: List[str]) -> List[List[float]]:
"""
Generates embedding for the input text list.
Args:
texts (List[str]): The text list to generate embedding for.

Returns:
List[List[float]]: The embedding list for each document in the input list. Each embedding is represented as a list of floating point values.
"""
return [self.embed_query(text) for text in texts]
```

For `embed_query`, you can add some content processing before requesting embedding. For example, if the text is very long, we can consider segmenting the text to prevent it from exceeding the maximum token limit. These are all possible. It depends on everyone's subjective initiative to improve it. Here is just a simple demo.

Through the above steps, we can define the calling method of embedding based on LangChain and Zhipu AI. We encapsulate this code inzhipuai_embedding.py file.

The corresponding source code of this article is here (https://github.com/datawhalechina/llm-universe/blob/main/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93/%E9%99%84LangChain%E8%87%AA%E5%AE%9A%E4%B9%89Embedding%E5%B0%81%E8%A3%85%E8%AE%B2%E8%A7%A3.ipynb). If you need to reproduce, you can download and run the source code.