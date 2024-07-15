# Chapter 4 Model Chains

Chains usually combine a large language model (LLM) with a prompt, based on which we can perform a series of operations on text or data. Chains can accept multiple inputs at once. For example, we can create a chain that accepts user input, formats it using a prompt template, and then passes the formatted response to the LLM. We can build more complex chains by combining multiple chains together, or by combining chains with other components.

## 1. Large Language Model Chain

The Large Language Model Chain (LLMChain) is a simple but very powerful chain, and is the basis for many of the chains we will introduce later.

### 1.1 Initialize the language model

```python
import warnings
warnings.filterwarnings('ignore')

from langchain.chat_models import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate 
from langchain.chains import LLMChain 

# Here we set the parameter temperature to 0.0 to reduceLess randomness in generated answers.
# If you want to get different and novel answers each time, you can try adjusting this parameter.
llm = ChatOpenAI(temperature=0.0) 
```

### 1.2 Initialize prompt template
Initialize prompt, this prompt will accept a variable called product. The prompt will ask LLM to generate the best name to describe the company that makes the product

```python
prompt = ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")
```

### 1.3 Build a large language model chain

Combine the large language model (LLM) and prompt (Prompt) into a chain. This large language model chain is very simple and allows us to run through the prompts and combine them into the large language model in a sequential manner.

```python
chain = LLMChain(llm=llm, prompt=prompt)
```

### 1.4 Run the Large Language Model Chain

So if we have a product called "Queen Size Sheet Set", we can run it through this chain by using `chain.run`

```python
product = "King Size Sheet Set"
chain.run(product)
```

'"Luxury Bed Spindle"'

You can enter any product description and see what the chain will output.

## 2. Simple Sequential Chains

Sequential Chains are chains that execute their links in a predefined order. Specifically, we will use a Simple Sequential Chain, which is the simplest type of sequential chain, where each step has an input/output and the output of one step is the input of the next step.

```python
from langchain.chains import SimpleSequentialChain
llm = ChatOpenAI(temperature=0.9)
```

### 2.1 Create two child chains

```python
# Prompt Template 1: This prompt will accept a product and return the best name to describe the company
first_prompt = ChatPromptTemplate.from_template(
"What is the best name to describe a company that makes {product}"
)
chain_one = LLMChain(llm=llm,prompt=first_prompt)

# Prompt Template 2: Accepts a company name and outputs a 20-word description of the company
second_prompt = ChatPromptTemplate.from_template(
"Write a 20-word description for the following \
Company: {company_name}"
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)
```

### 2.2 Build a Simple Sequential Chain
Now we can combine the two LLMChains so that we can create the company name and description in one step

```python
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
verbose=True)
```

Give an input and run the above chain

### 2.3 Run a Simple Sequential Chain

```python
product = "King Size Sheet Set"
overall_simple_chain.run(product)
```

> Entering new SimpleSequentialChain chain...
优床制造公司
优床制造公司 is a company that specializes in producing high-quality bedding.

> Finished chain.

'优床制造公司 is a company that specializes in producing high-quality bedding. '

## 3. Sequential Chain

When there is only one input and one output, a simple sequential chain (SimpleSequentialChain) can be implemented. When there are multiple inputs or multiple outputs, we need to use a sequential chain (SequentialChain) to implement it.

```python
import pandas as pd
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI #Import OpenAI model
from langchain.prompts import ChatPromptTemplate #Import chat prompt template
from langchain.chains import LLMChain #Import LLM chain.

llm = ChatOpenAI(temperature=0.9)
```

Next we will create a series of chains and use them one by one

### 3.1 Create four subchains

```python
#Subchain 1
# prompt template 1: Translate into English (translate the following review into English)
first_prompt = ChatPromptTemplate.from_template(
"Translate the following review into English:"
"\n\n{Review}"
)
# chain 1: Input: Review Output: English Review
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")

#Subchain 2
# prompt template 2: Summarize the following review in one sentence
second_prompt = ChatPromptTemplate.from_template(
"Please summarize the following review in one sentenceiew:"
"\n\n{English_Review}"
)
# chain 2: Input: English Review Output: Summary
chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

# subchain 3
# prompt template 3: What language is used in the following review
third_prompt = ChatPromptTemplate.from_template(
"What language is used in the following review:\n\n{Review}"
)
# chain 3: Input: Review Output: Language
chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")

# subchain 4
# prompt template 4: Use a specific language to write a follow-up reply to the following summary
fourth_prompt = ChatPromptTemplate.from_template(
"Use a specific language to write a follow-up reply to the following summary:"
"\n\nSummary: {summary}\n\nLanguage: {language}"
)
# chain 4: Input: summary, language Output: follow-up reply
chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")
```

### 3.2 Combine four subchains

```python
#Input: review
#Output: English review, summary, follow-up reply
overall_chain = SequentialChain(
chains=[chain_one, chain_two, chain_three, chain_four],
input_variables=["Review"],
output_variables=["English_Review", "summary","followup_message"],
verbose=True
)
```

Let's select a review and pass it through the entire chain. We can see that the original review is in French and the English review can be considered a translation.Next is the summary based on the English review, and the final output is the continuation of the original French text.

```python
df = pd.read_csv('../data/Data.csv')
review = df.Review[5]
overall_chain(review)
```

> Entering new SequentialChain chain...

> Finished chain.

{'Review': "I trouve le goût mediocre. La mousse ne tient pas, c'est bizarre. I've got mêmes dans le commerce et le goût est bien meilleur...\nVieux lot ou contrefaçon !?",
'English_Review': "I find the taste mediocre. The foam doesn't hold, it's weird. I buy thsame ones in stores and the taste is much better...\nOld batch or counterfeit!?",
'summary': "The reviewer finds the taste mediocre, the foam doesn't hold well, and suspects the product may be either an old batch or a counterfeit.",
'followup_message': "Follow-up reply (in French): I appreciate your understanding. I found this mediocre product to be very good and I didn't try it myself. I had no problems with the product and I had no idea what it was.The product is so old, so it is not a problem. There are no excuses for this experience and no problems to solve. You will be satisfied with it beforehand and the comments will be answered."}

## IV. Routing Chains

So far, we have learned about large language model chains and sequential chains. But what if we want to do something more complicated? A fairly common but basic operation is to route an input to a chain, depending on what exactly that input is. If you have multiple sub-chains, each specialized for a specific type of input, you can compose a routing chain that first decides which sub-chain to pass it to, and then passes it to that chain.

A router consists of two components:

- Routing Chain: The router chain itself, which is responsible for selecting the next chain to call
- Destination_chains: Chains that the router chain can route to

As a concrete example, let's look at where we route between different types of chains, where we have different prompts: 

```python
from langchain.chains.router import MultiPromptChain #Import multi-prompt chain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
llm = ChatOpenAI(temperature=0)
```

### 4.1 Define prompt templates

First, we define prompt templates that are suitable for different scenarios.

```python
# Chinese
#The first prompt is suitable for answering physics questions
physics_template = """You are a very smart physics expert. \
You are good at answering questions in a concise and easy-to-understand way. \
When you don't know the answer to a question, you admit that \
You don't know.

This is a question:
{input}"""

#The second prompt is suitable for answeringAnswer math questions
math_template = """You are a very good mathematician. \
You are good at answering math questions. \
You are so good because \
you are able to break down difficult problems into component parts, \
answer the components, and then combine them to answer broader questions.

This is a question:
{input}"""

#The third one is suitable for answering history questions
history_template = """You are a very good historian. \
You have excellent knowledge and understanding of people, events and contexts across a range of historical periods\
You have the ability to think, reflect, debate, discuss and evaluate the past. \
You respect historical evidence and have the ability to use it to support your interpretations and judgments.

This is a question:
{input}"""

#The fourth one is suitable for answering computer questions
computerscience_template = """ You are a successful computer science expert. \
You are creative, collaborative, \
forward-thinking, confident, problem-solving, \
understanding of theory and algorithms, and excellent communication skills. \
You are very good at answering programming questions. \
You are so good because you know \
How to solve problems by describing the solution in imperative steps that can be easily interpreted by machines, \
and you know how to choose a good balance between time complexity and space complexityBalanced solution.

This is also an input:
{input}"""
```

### 4.2 Name and describe the prompt templates

After defining these prompt templates, we can name each template and give a specific description. For example, the first description of physics is suitable for answering questions about physics. This information will be passed to the routing chain, and then the routing chain will decide when to use this subchain.

```python
# Chinese
prompt_infos = [
{
"name": "Physics",
"description": "Good at answering questions about physics",
"prompt template": physics_template
},
{
"name": "Mathematics",
"description": "Good at answering math questions",
"prompt template": math_template
},
{
"name": "History",
"description": "Good at answering history questions",
"prompt template": history_template
},
{
"name": "Computer Science",
"Description": "Good at answering computer science questions", 
"Prompt template": computerscience_template
}
]

```

LLMRouterChain (This chain uses LLM to determine how to route things)

Here, we need a **multi-prompt chain**. This is a specific type of chain used to route between multiple different prompt templates. But this is just one type of routing, we can also route between any type of chain.

The few classes we are going to implement here are the large model router chain. This class itself uses the language model to route between different child chains. This is where the description and name provided above will be used.

### 4.3 Create the corresponding destination chain based on the prompt template information
The destination chain is the chain called by the routing chain, and each destination chain is a language model chain

```python
destination_chains = {}
for p_info in prompt_infos:
name = p_info["name"]
prompt_template = p_info["prompt template"]
prompt = ChatPromptTemplate.from_template(template=prompt_template)
chain = LLMChain(llm=llm, prompt=prompt)
destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
```

### 4.4 Create a default destination chain
In addition to the destination chain, we also need a default destination chain. This is a chain that is called when the router cannot decide which child chain to use. In the example above, it might be called when the input question is not related to physics, mathematics, history, or computer science.

```python
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)
```

### 4.5 Define routing templates between different chains

This includes a description of the task to be completed and the specific format the output should be in.Note: Here is an example added based on the original tutorial, mainly because the "gpt-3.5-turbo" model is not well adapted to understand the meaning of the template, using "text-davinci-003" or "gpt-4-0613" can work well, so here are more example prompts to make it better to learn.

For example:

<< INPUT >>

"What is black body radiation?"

<< OUTPUT >>

```json
{{{{
"destination": string \ name of the prompt to use or "DEFAULT"
"next_inputs": string \ a potentially modified version of the original input
}}}}
```

```python
# Multi-prompt routing template
MULTI_PROMPT_ROUTER_TEMPLATE = """Give the language model a raw text input, \
Let it choose the model prompt that best suits the input. \
The system will provide you with the name of the available prompts and the description of the most suitable prompt to change. \
If you think that modifying the original input will eventually lead to a better language model,response, \
You can also modify the original input.

<< Format >>
Returns a markdown snippet with a JSON object in the following format:
```json
{{{{
"destination": string \ The prompt name to use or "DEFAULT"
"next_inputs": string \ The improved version of the original input
}}}}

Remember: "destination" must be one of the candidate prompt names specified below, \
or "DEFAULT" if the input doesn't quite fit any of the candidate prompts.
Remember: "next_inputs" can just be the original input if you think no modification is needed.

<< Candidate hints >>
{destinations}

<< Input >>
{{input}}

<< Output (remember to include ```json)>>

Example:
<< Input >>
"What is blackbody radiation?"
<< Output >>
```json
{{{{
"destination": string \ Hint name to use or use "DEFAULT"
"next_inputs": string \ Improved version of original input
}}}}

"""
```### 4.6 Building the Routing Chain
First, we create the complete router template by formatting the destinations defined above. This template can be used for many different types of destinations.
So here, you can add a different subject like English or Latin instead of just Physics, Math, History, and Computer Science.

Next, we create the prompt template from this template.

Finally, we create the routing chain by passing in the llm and the entire routing prompt. It is important to note that there is a routing output parsing here, which is important because it will help this link decide which sub-links to route between.

```python
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
destinations=destinations_str
)
router_prompt = PromptTemplate(
template=router_template,
input_variables=["input"],
output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

```

### 4.7 Create the overall chain

```python
#Multi-prompt chain
chain = MultiPromptChain(router_chain=router_chain, #l routing link
destination_chains=destination_chains, # target link
default_chain=default_chain, # default link
verbose=True 
)
```

### 4.8 Ask a question

If we ask a physics question, we want to see it routed to the physics link.

```python
chain.run("What is blackbody radiation?")
```

> Entering new MultiPromptChain chain...
Physics: {'input': 'What is blackbody radiation? '}
> Finished chain.

'Blackbody radiation refers to an idealized object that can completely absorb and radiate all electromagnetic radiation incident on it with the highest efficiency. The characteristic of this radiation is that its radiation intensity is related to the wavelength, and the radiation intensity at different wavelengths conforms to Planck's radiation law. Blackbody radiation has a wide range of applications in physics, such as studying thermodynamics, quantum mechanics, and cosmology. '

If we ask a math question, we hope to see it routed to the math link.

```python
chain.run("2+2 equals to what?")
```

> Entering new MultiPromptChain chain...
Math: {'input': '2+2 equals to what? '}

> Finished chain.

'2+2 equals 4. '

What happens if we pass a question that is not related to any sub-link?

Here, we asked a question about biology, and we can see that the link it chose is None. This means that it will be **passed to the default link, which itself is just a general call to the language model**. Luckily, language models know a lot about biology, so they can help us.

```python
chain.run("Why does everyAll cells contain DNA? ")
```

> Entering new MultiPromptChain chain...
Physics: {'input': 'Why does every cell in our body contain DNA?'}
> Finished chain.

'Every cell in our body contains DNA because DNA is the carrier of genetic information. DNA is a long chain of four bases (adenine, thymine, guanine and cytosine), which stores our genetic information, including our genes and genetic characteristics. Each cell needs this genetic information to perform its specific functions and tasks. So, DNA is present in every cell to ensure that our body functions properly.'

## English version prompt

**1. Large language model chain**

```python
from langchain.chat_models import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate 
from langchain.chains import LLMChain 

llm = ChatOpenAI(temperature=0.0) 

prompt = ChatPromptTemplate.from_template(
"What is the best name to describe \
a company that makes {product}?"
)

chain = LLMChain(llm=llm, prompt=prompt)

product = "Queen Size Sheet Set"
chain.run(product)
```

'Royal Comfort Linens'

**2. Simple Sequential Chain**

```python
from langchain.chains import SimpleSequentialChain
llm = ChatOpenAI(temperature=0.9)

first_prompt = ChatPromptTemplate.from_template(
"What is the best name to describe \
a company that makes {product}?"uct}?"
)
# Chain 1
chain_one = LLMChain(llm=llm, prompt=first_prompt)

second_prompt = ChatPromptTemplate.from_template(
"Write a 20 words description for the following \
company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)
product = "Queen Size Sheet Set"
overall_simple_chain.run(product)
```

> Entering new SimpleSequentialChain chain...
"Royal Comfort Beddings"
Royal Comfort Beddings is a reputable company that offers luxurious and comfortable bedding options fit for royalty.

> Finished chain.

'Royal Comfort Beddings is a reputable company that offers luxurious and comfortable bedding options fit for royalty.'

**3. Sequential chain**

```python
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate 
from langchain.chains import LLMChain 

llm = ChatOpenAI(temperature=0.9)

first_prompt = ChatPromptTemplate.from_template(
"Translate the following review to english:"
"\n\n{Review}"
)

chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")

second_prompt = ChatPromptTemplate.from_template(
"Can you summarize the following review in 1 sentence:"
"\n\n{English_Review}"
)

chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

third_prompt = ChatPromptTemplate.from_template(
"What language is the following review:\n\n{Review}"
)

chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")

fourth_prompt = ChatPromptTemplate.from_template(
"Write a follow up response to the following "
"summary in the specified language:"
"\n\nSummary: {summary}\n\nLanguage: {language}"
)

chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")

overall_chain = SequentialChain(
chains=[chain_one, chain_two, chain_three, chain_four],
input_variables=["Review"],
output_variables=["English_Review", "summary","followup_message"],
verbose=True
)

review = df.Review[5]
overall_chain(review)
```

> Entering new SequentialChain chain...

> Finished chain.

{'Review': "I've been working on my médiocre. I've been working on my business, and I've been working on my business...\nI've been working on my business, and I've been working on my business...\nçon!?",
'English_Review': "I find the taste poor. The foam doesn't hold, it's weird. I buy the same ones from the store and the taste is much better...\nOld batch or counterfeit!?",
'summary': 'The reviewer is disappointed with the poor taste and lack of foam in the product, suspecting it to be either an old batch or a counterfeit.',
'followup_message': "Reply to this review:\n\nCher(e) criticism,\n\nI really liked it because it was so delicious.I was so anxious about the product that I had to deal with it. I felt so frustrated and was in such a difficult situation.\n\nWhen I was away, I had to assure myself that I had to buy products of a genuine quality and that they were of high quality. I also knew that the product was very safe during the holiday.\n\nI was very careful, as it was possible to deal with the product in the event of an accident. I didn't have any inquiries, so I had to contact them directly, as they would not be able to help me with the situation.I have to admit that this is a problem. I have to pay close attention to the quality and to the product to ensure satisfaction.\n\nI have to admit that I have to comment on it, so I can know if I can continue to improve the product. I am not satisfied with it in advance and I have to answer any questions in order to determine the situation.\n\nI have always had the opportunity to experience a great product during my time.\n\nCordialement,\nL'équipe du service client"}

**4. Routing chain**

```python
from langchain.chains import LLMChain 
from langchain.chat_models import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate 
from langchain.chains.router import MultiPromptChain 
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
llm = ChatOpenAI(temperature=0.0) 

physics_template = """You are a very smart physics professor. \
You are great atanswering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""

math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""

computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexityy. 

Here is a question:
{input}"""

prompt_infos = [
{
"name": "physics", 
"description": "Good for answering questions about physics", 
"prompt_template": physics_template
},
{
"name": "math", 
"description": "Good for answering math questions", 
"prompt_template": math_template
},
{
"name": "History", 
"description": "Good for answering history questions", 
"prompt_template": history_template
},{
"name": "computer science", 
"description": "Good for answering computer science questions", 
"prompt_template": computerscience_template
}
]

destination_chains = {}
for p_info in prompt_infos:
name = p_info["name"]
prompt_template = p_info["prompt_template"]
prompt = ChatPromptTemplate.from_template(template=prompt_template)
chain = LLMChain(llm=llm, prompt=prompt)
destination_chains[name] = chain 

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
"destination": string \ name of the prompt to use or "DEFAULT"
"next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of thee candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
destinations=destinations_str
)
router_prompt = PromptTemplate(
template=router_template,
input_variables=["input"],
output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(router_chain=router_chain, 
destination_chains=destination_chains, 
default_chain=default_chain, verbose=True
)

print(chain.run("What is black body radiation?"))

print(chain.run("what is 2 + 2"))

print(chain.run("Why does every cell in our body contain DNA?"))
```

> Entering new MultiPromptChain chain...
physics: {'input': 'What is black body radiation?ation?'}
> Finished chain.
Black body radiation refers to the electromagnetic radiation emitted by an object that absorbs all incident radiation and reflects or transmits none. It is called "black body" because it absorbs all wavelengths of light, appearing black at room temperature. 

According to Planck's law, black body radiation is characterized by a continuous spectrum of wavelengths and intensities, which depend on the temperature of the object. As the temperature increases, the peak intensity of the radiation shifts to shorter wavelengths, resulting in a change in color from red to orange, yellow, white, and eventually blue at very high temperatures.

Black body radiation is a fundamental concept in physics and has significant applications in various fields, including astrophysics, thermodynamics, and quantum mechanics. It played a crucial role in the development of quantum theory and understanding the behavior of light and matter.

> Entering new MultiPromptChain chain...
math: {'input': 'what is 2 + 2'}
> Finished chain.
Thank you for your kind words! As a mathematician, I am happy to help with any math questions, no matter how simple or complex they may be.

The question you've asked is a basic addition problem: "What is 2 + 2?" To solve this, we can simply add the two numbers together:

2 + 2 = 4

Therefore, the answer to the question "What is 2 + 2?" is 4.

> Entering new MultiPromptChainomptChain chain...
None: {'input': 'Why does every cell in our body contain DNA?'}
> Finished chain.
Every cell in our body contains DNA because DNA is the genetic material that carries the instructions for the development, functioning, and reproduction of all living organisms. DNA contains the information necessary for the synthesis of proteins, which are essential for the structure and function of cells. It serves as a blueprint for the production of specific proteins that determinAdditionally, DNA is responsible for the transmission of genetic information from one generation to the next during reproduction. Therefore, every cell in our body contains DNA to ensure the proper functioning and continuity of life.