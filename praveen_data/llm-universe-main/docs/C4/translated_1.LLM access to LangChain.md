# 4.1 Connect LLM to LangChain

The source files involved in the text can be obtained from the following path:
> - [LLM Connect to LangChain.ipynb](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C4%20%E6%9E%84%E5%BB%BA%20RAG%20%E5%BA%94%E7%94%A8/1.LLM%20%E6%8E%A5%E5%85%A5%20LangChain.ipynb)
> - [wenxin_llm.py](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C4%20%E6%9E%84%E5%BB%BA%20RAG%20%E5%BA%94%E7%94%A8/wenxin_llm.py) > - [zhipuai_llm.py](https://github.com/datawhalechina/llm-unichina verse/blob/main/notebook/C4%20%E6%9E%84%E5%BB%BA%20RAG%20%E5%BA%94%E7%94%A8/zhipuai_llm.py)

---

LangChain provides an efficient development framework for developing custom applications based on LLM, which allows developers to quickly stimulate the powerful capabilities of LLM and build LLM applications. LangChain also supports a variety of large models, and has built-in calling interfaces for large models such as OpenAI and LLAMA. However, LangChain does not have all large models built in. It provides powerful scalability by allowing users to customize LLM types.

## 1. Call ChatGPT based on LangChain

LangChain provides encapsulation for a variety of large models. The LangChain interface can easily call ChatGPT and integrate it in personal applications built on the LangChain framework. Here we briefly describe how to use the LangChain interface to call ChatGPT.

Note that calling ChatGPT based on the LangChain interface also requires configuring your personal key, and the configuration method is the same as above.

### 1.1 Models
Import `OpenAI`'s dialogue model `ChatOpenAI` from `langchain.chat_models`.In addition to OpenAI, `langchain.chat_models` also integrates other dialogue models. For more details, please refer to the [Langchain official document](https://api.python.langchain.com/en/latest/langchain_api_reference.html#module-langchain.chat_models).

```python
import os
import openai
from dotenv import load_dotenv, find_dotenv

# Read local/project environment variables.

# find_dotenv() finds and locates the path of the .env file
# load_dotenv() reads the .env file and loads the environment variables in it into the current running environment
# If you set the global environment variable, this line of code has no effect.
_ = load_dotenv(find_dotenv())

# Get the environment variable OPENAI_API_KEY
openai_api_key = os.environ['OPENAI_API_KEY']
```

langchain is not installed-openai, please run the following code first!

```python
from langchain_openai import ChatOpenAI
```

Next, you need to instantiate a ChatOpenAI class. You can pass in hyperparameters to control the answer when instantiating, such as the `temperature` parameter.

```python
# Here we set the parameter temperature to 0.0 to reduce the randomness of the generated answer.
# If you want to get a different and novel answer every time, you can try to adjust this parameter.
llm = ChatOpenAI(temperature=0.0)
llm
```

ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x000001B17F799BD0>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x000001B17F79BA60>, temperature=0.0, openai_api_key=SecretStr('**********'), openai_api_base='https://api.chatgptid.net/v1', openai_proxy='')

The cell above assumes that your OpenAI API key is set in the environment variable. If you want to manually specify the API key, use the following code:

```python
llm = ChatOpenAI(temperature=0, openai_api_key="YOUR_API_KEY")
```

As you can see, the ChatGPT-3.5 model is called by default. In addition, several commonly used hyperparameter settings include:

· model_name: The model to be used, the default is 'gpt-3.5-turbo', and the parameter settings are consistent with the OpenAI native interface parameter settings.

· temperature: Temperature coefficient, the value is the same as the native interface.

· openai_api_key: OpenAI API key, if you do not use environment variables to set the API Key, you can also set it during instantiation.

openai_proxy: Set the proxy. If you do not use environment variables to set the proxy, you can also set it in the instance· streaming: whether to use streaming, that is, output the model answer word by word, the default is False, which is not described here.

· max_tokens: the maximum number of tokens output by the model, the meaning and value are the same as above.

Once we initialize the `LLM` of your choice, we can try to use it! Let's ask "Please introduce yourself!"

```python
output = llm.invoke("Please introduce yourself!")
```

```python
output
```

AIMessage(content='Hello, I am an intelligent assistant, focusing on providing users with various services and help. I can answer questions, provide information, solve problems, and help users complete work and life more efficiently. If you have any questions or need help, please feel free to let me know, I will do my best to help you. Thank you for your use!', response_metadata={'token_usage': {'completion_tokens': 104, 'prompt_tokens': 20, 'total_tokens': 124}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_b28b39ffa8', 'finish_reason': 'stop', 'logprobs': None})

### 1.2 Prompt

When we develop large model applications, most of the time we don't pass user input directly to the LLM. Usually, they add the user input to a larger text, called a `prompt template`, which provides additional context about the specific task at hand.
PromptTemplates help with this! They bundle all the logic from user input to a fully formatted prompt. This can be started very simply - for example, the prompt to generate the above string is:

We need to construct a personalized Template first:

```python
from langchain_core.prompts import ChatPromptTemplate

# Here we ask the model to translate the given text into Chinese
prompt = """Please translate the text separated by three backticks into English!\
text: ```{text}```
"""
```

Next, let's take a look at the complete prompt template constructed:

```python
text = "I carry luggage that is heavier than my body,\
Swimmed to the bottom of the Nile River, \
After several flashes of lightning, I saw a bunch of circles of light, \
Not sure if it was here. \
"
prompt.format(text=text)
```

'Please translate the text separated by three backticks into English! text: ```I swam to the bottom of the Nile with luggage heavier than my body, and after several lightnings, I saw a bunch of halos, not sure if it was here.```\n'

We know that the interface of the chat model is based on messages, not raw text. PromptTemplates can also be used to generate message lists. In this example, `prompt` contains not only the input content information, but also the information of each `message` (role, position in the list, etc.). Usually, a `ChatPromptTemplate` is a list of `ChatMessageTemplate`. Each `ChatMessageTemplate` contains instructions for formatting the chat message (its role and content).

Let's take a look at an example:

```python
from langchain.prompts.chat import ChatPromptTemplate

template = "You are a translation assistant that can help me translate {input_language} into {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
("system", template),
("human", human_template),
])

text = "I carried a bag heavier than my body, \
swam into the bottom of the Nile, \
passed a few lightning bolts and saw a bunch of light circles, \
not sure if it was here. \
"
messages = chat_prompt.format_messages(input_language="Chinese", output_language="English", text=text)
messages
```

[SystemMessage(content='You are a translation assistant who can help me translate Chinese into English.'),
HumanMessage(content='I carried a bag heavier than my body, swam into the bottom of the Nile, passed a few lightning bolts and saw a bunch of light circles, not sure if it was here.')]

Next, let's call the defined `llm` and `messages` to output the answer:

```python
output = llm.invoke(messages)
output
```

AIMessage(content='I carried luggage heavier than my body and dived into the bottom of the Nile River. After passing through several flashes of lightning, I saw a pile of halos, not sure if this is the place.')

### 1.3 Output parser

OutputParsers convert the raw output of the language model into a format that can be used downstream. There are several main types of OutputParsers, including:
- Convert LLM text to structured information (such as JSON)
- Convert ChatMessage to string
- Convert extra information returned by calls other than messages (such as OpenAI function calls) to string

Finally, we pass the model output to `output_parser`, which is a `BaseOutputParser`, which means it accepts **strings or BaseMessages as input**. StrOutputParser in particular simply converts any input to a string.

```python
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
output_parser.invoke(output)
```

'I carried luggage heavier than my body and dived into the bottom of the Nile River. After passing through several flashes of lightning, I saw a pile of halos, not sure if this is the place.'

From the above results, we can see that we successfully parsed the output of type `ChatMessage` into a `string` through the output parser

### 1.4 Complete Process

We can now combine all of this into a chain. The chain will take input variables, pass those variables to the prompt template to create a prompt, pass the prompt to the language model, and then pass the output through the (optional) output parser. Next, we will use the LCEL syntax to quickly implement a chain. Let's see it in action!

```python
chain = chat_prompt | llm | output_parser
chain.invoke({"input_language":"中文", "output_language":"英文","text": text})

```

'I carried luggage heavier than my body and dived into the bottom of the Nile River. After passing through several flashes of lightning, I saw a pile of halos, not sure if this is the place.'

Test another example:

```python
text = 'I carried luggage heavier than my body and dived into the bottom of the Nile River. After passing through several flashes of lightning, I saw a pile of halos, not sure if this is the place.'
chain.invoke({"input_language":"英文", "output_language":"中文","text": text})
```

'I carried luggage heavier than my body and dived into the bottom of the Nile River. After passing through several flashes of lightning, I saw a pile of halos, not sure if this is the destination. '

> What is LCEL? 
LCEL (LangChain Expression Language, Langchain's expression language), LCEL is a new syntax and an important addition to the LangChain toolkit. It has many advantages, making it easier for us to handle LangChain and agents. 

- LCEL provides asynchronous, batch and stream processing support, allowing code to be quickly ported to different servers.- LCEL has fallback measures to solve the problem of LLM format output.
- LCEL increases the parallelism of LLM and improves efficiency.
- LCEL has built-in logging, which helps to understand the operation of complex chains and agents even if the agents become complex.

Usage example:

`chain = prompt | model | output_parser`

In the above code, we use LCEL to piece together different components into a chain, in which user input is passed to the prompt template, and then the prompt template output is passed to the model, and then the model output is passed to the output parser. The symbol | is similar to the Unix pipe operator, which links different components together and uses the output of one component as the input of the next component.

## 2. Use LangChain to call Baidu Wenxin Yiyan

We can also call Baidu Wenxin Da Model through the LangChain framework to connect the Wenxin model to our application framework.

### 2.1 Customize LLM to access langchain
In the old version, LangChain does not directly support Wenxin calls. We need to customize an LLM that supports Wenxin model calls. In order to show users how to customize LLM, we briefly describe this method in "Appendix 1. LangChain Customized LLM". You can also refer to [Source Document](https://python.langchain.com/v0.1/docs/modules/model_io/llms/custom_llm/).

Here, we can directly call the customized Wenxin_LLM. For details on how to encapsulate Wenxin_LLM, see `wenxin_llm.py`.

**Note: The following code needs to download our encapsulated code [wenxin_llm.py](./wenxin_llm.py) to the same directory as this Notebook before it can be used directly. Because the new version of LangChain can directly call the Wenxin Qianfan API, we recommend using the next part of the code to call the Wenxin Yiyan model**

```python
# Need to download source code
from wenxin_llm import Wenxin_LLM
```

We hope to store the secret key directly in the .env file like calling ChatGPT, and load it into the environment variable, so as to hide the specific details of the secret key and ensure security. Therefore, we need to configure `QIANFAN_AK` and `QIANFAN_SK` in the .env file and load them using the following code:

```python
from dotenv import find_dotenv, load_dotenv
import os
# Read local/project environment variables.
# find_dotenv() finds and locates the path of the .env file
# load_dotenv() reads the .env file and loads the environment variables in it into the current running environment
# If you set global environment variables, this line of code has no effect.
_ = load_dotenv(find_dotenv())

# Get environment variable API_KEY
wenxin_api_key = os.environ["QIANFAN_AK"]
wenxin_secret_key = os.environ["QIANFAN_SK"]
```

```python
llm = Wenxin_LLM(api_key=wenxin_api_key, secret_key=wenxin_secret_key, system="You are an assistant!")
```

```python
llm.invoke("Hello, please introduce yourself!")
```

[INFO] [03-31 22:12:53] openapi_requestor.py:316 [t:27812]:requesting llm api endpoint: /chat/eb-instant

1
2

'Hello! I'm an assistant who helps you with tasks. I'm quick to respond, efficient, and adaptable, and I'm committed to providing you with quality service. Whatever help you need, I'll do my best to meet your needs. '

```python
# Or use
llm(prompt="Hello, please introduce yourself!")
```

[INFO] [03-31 22:12:41] openapi_requestor.py:316 [t:27812]: requesting llm api endpoint: /chat/eb-instant

1
2

'Hello! I'm an assistant who helps you with tasks. I'm quick to learn and process information, and I'll provide help and answer questions based on your needs. Whatever help you need, I'll do my best to provide support. '

Thus, we can add the Wenxin model to the LangChain architecture and implement the call to the Wenxin model in the application.

### 2.2 Directly call Wenxin in LangChainYiyan

We can also use the new version of LangChain to directly call the Wenxin Yiyan model.

```python

from dotenv import find_dotenv, load_dotenv
import os

# Read local/project environment variables.

# find_dotenv() finds and locates the path of the .env file

# load_dotenv() reads the .env file and loads the environment variables in it into the current running environment

# If you set the global environment variables, this line of code has no effect.
_ = load_dotenv(find_dotenv()) # Get environment variables API_KEY QIANFAN_AK = os.environ["QIANFAN_AK"] QIANFAN_SK = os.environ["QIANFAN_SK"] `` ``python # Install required dependencies %pip install -qU langchain langchain-community `` `` imppython from langchain_community.llms ort QianfanLLMEndpoint llm = QianfanLLMEndpoint(streaming=True) res = llm("Hello, please introduce yourself!") print(res) ```` d:\Miniconda\miniconda3\envs\llm2\lib\site-packages\ langchain_core\_api\deprecation.py:117: LangChainDeprecationWarning: The function `__call__` was deprecated in LangChain 0.1.7 and will be removed in 0.2.0. Use invoke instead. warn_deprecated( [INFO] [03-31 22:40: 14] openapi_requestor.py:316 [t:3684]: requesting llm api endpoint: /chat/eb-instant [INFO] [03-31 22:40:14] oauth.py:207 [t:3684]: trying to refresh access_token for ak `MxBM7W***`
[INFO] [03-31 22:40:15] oauth.py:220 [t:3684]: sucessfully refresh access_token

Hello! My name is Wenxin Yiyan, and my English name is ERNIE Bot. I am an artificial intelligence language model that can assist you with a wide range of tasks and provide information on various topics, such as answering questions, providing definitions and explanations and suggestions, and providing contextual knowledge and dialogue management. If you have any questions or need help, feel free to ask me and I will do my best to answer.

## 3. Use LangChain to call iFlytek Spark

We can also call iFlytek Spark large model through LangChain framework. For more information, please refer to [SparkLLM](https://python.langchain.com/docs/integrations/llms/sparkllm)

We hope to store the secret key directly in the .env file like calling ChatGPT, and load it into the environment variable, so as to hide the specific details of the secret key and ensure securityTherefore, we need to configure `IFLYTEK_SPARK_APP_ID`, `IFLYTEK_SPARK_API_KEY` and `IFLYTEK_SPARK_API_SECRET` in the .env file and load it using the following code:

```python
from dotenv import find_dotenv, load_dotenv
import os

# Read local/project environment variables.

# find_dotenv() finds and locates the path of the .env file
# load_dotenv() reads the .env file and loads the environment variables in it into the current running environment
# If you set global environment variables, this line of code has no effect.
_ = load_dotenv(find_dotenv())

# Get environment variable API_KEY
IFLYTEK_SPARK_APP_ID = os.environ["IFLYTEK_SPARK_APP_ID"]
IFLYTEK_SPARK_API_KEY = os.environ["IFLYTEK_SPARK_API_KEY"]
IFLYTEK_SPARK_API_SECRET = os.environ["IFLYTEK_SPARK_API_SECRET"]
```

```python
def gen_spark_params(model):
'''
Construct Spark model request parameters
'''

spark_url_tpl = "wss://spark-api.xf-yun.com/{}/chat"
model_params_dict = {
# v1.5 version
"v1.5": {
"domain": "general", # Used to configure large model version
"spark_url": spark_url_tpl.format("v1.1") # Service address of cloud environment
},
# v2.0 version
"v2.0": {
"domain": "generalv2", # Used to configure large model version
"spark_url": spark_url_tpl.format("v2.1") # Service address of cloud environment},
# v3.0 version
"v3.0": {
"domain": "generalv3", # Used to configure the large model version
"spark_url": spark_url_tpl.format("v3.1") # Service address of the cloud environment
},
# v3.5 version
"v3.5": {
"domain": "generalv3.5", # Used to configure the large model version
"spark_url": spark_url_tpl.format("v3.5") # Service address of the cloud environment
}
}
return model_params_dict[model]
```

```python
from langchain_community.llms import SparkLLM

spark_api_url = gen_spark_params(model="v1.5")["spark_url"]

#Load the model (default v3.0)
llm = SparkLLM(spark_api_url = spark_api_url) #Specify v1.5 version
```

```python
res = llm("Hello, please introduce yourself!")
print(res)
```

Hello, I am the cognitive intelligence model developed by iFlytek, my name is iFlytek Spark Cognitive Model. I can communicate with humans naturally, answer questions, and efficiently complete cognitive intelligence needs in various fields.

So we can add the Spark model to the LangChain architecture and call the Wenxin model in the application.

## 4. Use LangChain to call Zhipu GLM

We can also call Zhipu AI model through the LangChain framework to connect it to our application framework. Since the [ChatGLM](https://python.langchain.com/docs/integrations/llms/chatglm) provided in langchain is no longer available, we need to customize a LLM.

If you are using Zhipu GLM API, you need to encapsulate our code [zhipuai_llm.py](./zhipuai_llm.py) to the same directory as this Notebook, and then you can run the following code to use GLM in LangChain.

According to Zhipu’s official announcement, the following models will be deprecated soon. After these models are deprecated, they will be automatically routed to new models. Please note that before the deprecation date, update your model code to the latest version to ensure a smooth transition of services. For more information about the model, please visit [model](https://open.bigmodel.cn/dev/howuse/model)

| Model code | Deprecation date | Point to model |
| ---- | ---- | ---- |
|chatglm_pro|December 31, 2024|glm-4|
|chatglm_std|December 31, 2024|glm-3-turbo|
|chatglm_lite|December 31, 2024|glm-3-turbo|

### 4.1 Customize chatglm to access langchain

```python
# Need to download source code
from zhipuai_llm import ZhipuAILLM
``````python

from dotenv import find_dotenv, load_dotenv
import os

# Read local/project environment variables.

# find_dotenv() finds and locates the path of the .env file
# load_dotenv() reads the .env file and loads the environment variables in it into the current running environment
# If you set the global environment variables, this line of code has no effect.
_ = load_dotenv(find_dotenv())

# Get the environment variable API_KEY
api_key = os.environ["ZHIPUAI_API_KEY"] # Fill in the APIKey information obtained in the console
```

```python
zhipuai_model = ZhipuAILLM(model = "glm-4", temperature = 0.1, api_key = api_key) #model="glm-4-0520",
```

```python
zhipuai_model("Hello, please introduce yourself!")
```

'Hello! I am Zhipu Qingyan, a researcher at the KEG Laboratory andLanguage model co-trained by Zhipu AI in 2023. My goal is to help users solve problems by answering their questions. Since I am a computer program, I have no self-awareness and cannot perceive the world like a human. I can only answer questions by analyzing the information I have learned. '

---

#### The above-mentioned source file acquisition path:

> - [1.LLM access LangChain.ipynb](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C4%20%E6%9E%84%E5%BB%BA%20RAG%20%E5%BA%94%E7%94%A8/1.LLM%20%E6%8E%A5%E5%85%A5%20LangChain.ipynb)
> - [wenxin_llm.py](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C4%20%E6%9E%84%E5%BB%BA%20RAG%20%E5%BA%94%E7%94%A8/wenxin_llm.py)
> - [zhipuai_llm.py](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C4%20%E6%9E%84%E5%BB%BA%20RAG%20%E5%BA%94%E7%94 %A8/zhipuai_llm.py)