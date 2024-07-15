In this course, we will see a lot of code like `from llama import BasicModelRunner`. Many students may think that they need to install the `llama` library. In fact, <b>we need to install the `lamini` library</b>, and `llama` is just a subset of the `lamini` library. The following are the installation and usage instructions of the `lamini` library.

## Installation
The installation of the `lamini` library is very simple. Just execute the following command:

`pip install lamini`

## Registration
Next, we need to go to [lamini official website](https://www.lamini.ai/) to register an account to obtain an api key to use all the functions of the `lamini` library.
![lamini official website](../../figures/Finetuning%20Large%20Language%20Models/lamini official website.png)

You can use Google Mail (default) or other email addresses to register an account. After registration, click `Account` in the upper left corner of the official website to see your api key and remaining balance.

![lamini official website](../../figures/Finetuning%20Large%20Language%20Models/lamini官网_apikey.png)

## Usage
### 1. Default method
`lamini` needs to create a configuration file `~/.powerml/configure_llama.yaml` in your user directory by default, and then write the configuration information as follows:

```
production:
key: "<YOUR-KEY-HERE>"
```

The user directory is generally `C:\Users\Administrator` in Windows system, and `~/` in Linux/maxOS.

### 2. Simple method
Since the default method is more troublesome, we provide a more convenient method for everyone. When we need to use the `LLMEngine` or `BasicModelRunner` class of `llama`, we can directly write `production.key` into the class parameter `config`, for example:

```
llm = LLMEngine(
id="example_llm",
config={"production.key": "<YOUR-KEY-HERE>"}
)
```

Or:

```
non_finetuned =BasicModelRunner("meta-llama/Llama-2-7b-hf", 
config={"production.key": "<YOUR-KEY-HERE>"})

```

Just replace `<YOUR-KEY-HERE>` with our api key on the `lamini` official website.

If you are worried about the code leaking the api key, you can use a configuration file to store `production.key` in the same way as ChatGPT, which I will not elaborate on here.