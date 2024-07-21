# Using Embedding API

The source code for this article is here (https://github.com/datawhalechina/llm-universe/blob/main/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93/2.%E4%BD%BF%E7%94%A8%20Embedding%20API.ipynb). If you need to reproduce, you can download and run the source code.
## 1. Using OpenAI API
GPT has a packaged interface, so we can simply package it. Currently, there are three GPT embedding modes, and the performance is as follows:
|Model | Pages per dollar | [MTEB](https://github.com/embeddings-benchmark/mteb) score | [MIRACL](https://github.com/project-miracl/miracl) score|
| --- | --- | --- | --- |
|text-embedding-3-large|9,615|54.9|64.6|
|text-embedding-3-small|62,500|62.3|44.0|
|text-embedding-ada-002|12,500|61.0|31.4|
* MTEB score is the average score of eight tasks such as classification, clustering, and pairing of the embedding model.
* MIRACL score is the average score of the embedding model in the retrieval task. 

From the above three embedding models, we can see that `text-embedding-3-large` has the best performance and the most expensive price. We can use it when the application we build needs better performance and the cost is sufficient; `text-embedding-3-small` has better performance and price. We can choose this model when our budget is limited; and `text-embedding-ada-002` is the previous generation model of OpenAI. It is not as good as the previous two in terms of performance or price, so it is not recommended.

```python
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Read local/project environment variables.
# find_dotenv() finds and locates the path of the .env file
# load_dotenv() reads the .env file and loads the environment variables in it into the current running environment 
# If you set global environment variables, this line of code will have no effect.
_ = load_dotenv(find_dotenv())

# If you need to access through a proxy port, you need to configure as follows
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

def openai_embedding(text: str, model: str=None):
# Get environment variable OPENAI_API_KEY
api_key=os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=api_key)

# embedding model：'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'
if model == None:
model="text-embedding-3-small"

response = client.embeddings.create(
input=text,
model=model
)
return response

response = openai_embedding(text='Input text to generate embedding, string form.')
```

The data returned by the API is in `json` format. In addition to the `object` vector type, there are also data `data` for storing data, `model` for embedding model, and `usage` for this token usage, as shown below:
```json
{
"object": "list",
"data": [
{
"object": "embedding",
"index": 0,
"embedding": [
-0.006929283495992422,
... (omitted)
-4.547132266452536e-05,
],
}
],
"model": "text-embedding-3-small",
"usage": {
"prompt_tokens": 5,
"total_tokens": 5
}
}
```
We can call the object of response to get the type of embedding.

```python
print(f'The returned embedding type is: {response.object}')
```

The returned embedding type is: list

The embedding is stored in data, and we can check the length of the embedding and the generated embedding.

```python
print(f'embedding length is: {len(response.data[0].embedding)}')
print(f'embedding (first 10) is: {response.data[0].embedding[:10]}')
```

embedding length is: 1536
embedding (first 10) is：[0.03884002938866615, 0.013516489416360855, -0.0024250170681625605, -0.01655769906938076, 0.024130908772349358, -0.017382603138685226, 0.04206013306975365, 0.011498954147100449, -0.028245486319065094, -0.00674333656206727]

We can also check the model and token usage of this embedding.

```python
print(f'This embedding model is: {response.model}')
print(f'This token usage is: {response.usage}')
```

This embedding model is: text-embedding-3-small
This token usage is: Usage(prompt_tokens=12, total_tokens=12)

## 2. Use Wenxin Qianfan API
Embedding-V1 is based on BaiduThe text representation model of Wenxin Large Model technology, Access token is the credential for calling the interface. When using Embedding-V1, you should first obtain Access token with API Key and Secret Key, and then use Access token to call the interface to embed text. At the same time, Qianfan Large Model Platform also supports embedding models such as bge-large-zh.

```python
import requests
import json

def wenxin_embedding(text: str):
# Get environment variables wenxin_api_key, wenxin_secret_key
api_key = os.environ['QIANFAN_AK']
secret_key = os.environ['QIANFAN_SK']

# Use API Key and Secret Key to https://aip.baidubce.com/oauth/2.0/token to obtain Access token
url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={0}&client_secret={1}".format(api_key, secret_key) payload = json.dumps("") headers = { 'Content-Type': 'application/json', 'Accept': ' application/json' } response = requests.request("POST", url, headers=headers, data=payload) # Embedding text url = "https://aip.baidubce.com/rpc/2.0 through the obtained Access token /ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=" + str(response.json().get("access_token"))input = []
input.append(text)
payload = json.dumps({
"input": input
})
headers = {
'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

return json.loads(response.text)
# text should be List(string)
text = "Input text to generate embedding, in string form."

response = wenxin_embedding(text=text)
```

Embedding-V1 has a separate id for each embedding, and a timestamp to record the time of the embedding.

```python
print('This embedding id is: {}'.format(response['id']))
print('ThisThe timestamp of this embedding generation is: {}'.format(response['created']))
```

The embedding id of this time is: as-hvbgfuk29u
The timestamp of this embedding generation is: 1711435238

Similarly, we can also get the embedding type and embedding from the response.

```python
print('The returned embedding type is: {}'.format(response['object']))
print('The embedding length is: {}'.format(len(response['data'][0]['embedding'])))
print('embedding (first 10) is: {}'.format(response['data'][0]['embedding'][:10]))
```

The returned embedding type is: embedding_list
The embedding length is: 384
Embedding (first 10) is: [0.060567744076251984, 0.020958080887794495, 0.053234219551086426, 0.02243831567466259, -0.024505289271473885, -0.09820500761270523, 0.04375714063644409, -0.009092536754906178, -0.020122773945331573, 0.015808865427970886]

## 3. Use iFlytek Spark API

### Not yet open

## 4. Use Zhipu API
Zhipu has a packaged SDK, we can call it.

```python from zhipuai import ZhipuAI def zhipu_embedding(text: str): api_key = os.environ['ZHIPUAI_API_KEY'] client = ZhipuAI(api_key=api_key) response = client.embeddings.create( model="embedding-2", input=text, ) return response

text = 'Input text to generate embedding, in string form. '
response = zhipu_embedding(text=text)
```

response is of type `zhipuai.types.embeddings.EmbeddingsResponded`. We can call `object`, `data`, `model`, `usage` to view the embedding type, embedding, embedding model and usage of response.

```python
print(f'response type is: {type(response)}')
print(f'embedding type is: {response.object}')
print(f'the generated embedding model is: {response.model}')
print(f'the generated embedding length is: {len(response.data[0].embedding)}')
print(f'embedding (first 10) is: {response.data[0].embedding[:10]}')
```

Response type: <class 'zhipuai.types.embeddings.EmbeddingsResponded'>
Embedding type: list
The model for generating embeddings: embedding-2
The length of generated embeddings: 1024
Embedding (first 10) is: [0.017892399802803993, 0.0644201710820198, -0.009342825971543789, 0.02707476168870926, 0.004067837726324797, -0.05597858875989914, -0.04223804175853729, -0.03003198653459549, -0.016357755288481712, 0.06777040660381317]

The source code for this article is [here](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93/2.%E4%BD%BF%E7%94%A8%20Embedding%20API.ipynb). If you need to reproduce, you can download and run the source code.