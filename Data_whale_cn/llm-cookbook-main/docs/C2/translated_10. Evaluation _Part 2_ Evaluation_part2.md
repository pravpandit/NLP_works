# Chapter 10 Evaluation (Part 2) - When there is no simple correct answer

In the previous chapter, we explored how to evaluate the performance of the LLM model when **there is a clear correct answer**, and we learned to write a function to verify whether the LLM correctly classified the listed products.

However, if we want to use LLM to generate text, not just for solving classification problems, how should we evaluate its answer accuracy? In this chapter, we will discuss how to evaluate the quality of LLM's output in this application scenario.

## 1. Run the question-answering system to get a complex answer

First, we run the question-answering system built in the previous chapter to get a complex answer that does not have a simple correct answer:

```python
import utils_zh

'''
Note: Due to the model's poor understanding of Chinese, the Chinese prompt may fail randomly. You can run it multiple times; students are also welcome to explore more stable Chinese prompts
'''
# User message
customer_msg = f"""
Tell me about the smartx pro phone and the fotosnap camera, the dslr one.
Also, what TVs do you have here? """

# Extract product names from questions
products_by_category = utils_zh.get_products_from_query(customer_msg)
# Convert product names to lists
category_and_product_list = utils_zh.read_string_to_list(products_by_category)
# Find information about products
product_info = utils_zh.get_mentioned_product_info(category_and_product_list)
# Generate answers from information
assistant_answer = utils_zh.answer_user_msg(user_msg=customer_msg, product_info=product_info)

print(assistant_answer) 
```

Information about SmartX Pro phone and FotoSnap DSLR camera:

1. SmartX Pro phone (model: SX-PP10) is a powerful smartphone with a 6.1-inch display, 128GB storage, 12MP dual camera and 52. FotoSnap DSLR Camera (Model: FS-DSLR200) ​​is a versatile DSLR camera with a 24.2MP sensor, 1080p video capture, 3-inch LCD screen and interchangeable lenses. Price: $599.99, warranty: 1 year.

Information about TVs:

We have the following TVs to choose from:

1. CineView 4K TV (Model: CV-4K55) - 55-inch display, 4K resolution, HDR support and smart TV features. Price: $599.99, warranty: 2 years.

2. CineView 8K TV (Model: CV-8K65) - 65-inch display, 8K resolution, HDR support and smart TV features. Price: $2999.99, warranty: 2 years.
3. CineView OLED TV (model: CV-OLED55) - 55-inch OLED display, 4K resolution, HDR and smart TV features. Price: $1499.99, warranty: 2 years.

Do you have any special requirements or other questions about the above products?

## 2. Use GPT to evaluate whether the answer is correct

We hope you can learn a design model from it, i.e. when you can specify a list of criteria to evaluate the LLM output, you can actually use another API call to evaluate your first LLM output.

```python
from tool import get_completion_from_messages

# Question, context
cust_prod_info = {
'customer_msg': customer_msg,
'context': product_info
}

def eval_with_rubric(test_set, assistant_answer):
"""
Evaluate the generated answer using the GPT API

Parameters:
test_set: test set
assistant_answer: assistant's answer
"""

cust_msg = test_set['customer_msg']
context = test_set['context']
completion = assistant_answer

# Persona
system_message = """\
You are an assistant evaluating how well a customer service agent answers a user's question by looking at the context the agent used.
"""

# Specific instructions
user_message = f"""\
You are evaluating submitted answers to questions based on the context the agent used. Here is the data:
[Start]
************
[User Question]: {cust_msg}
************
[Context used]: {context}
************
[Customer Agent's Answer]: {completion}
************
[End]

Please compare the factual content of the submitted answer to the context, ignoring differences in style, grammar, or punctuation.
Answer the following questions:
Is the assistant's response based solely on the context provided? (yes or no)
Does the answer include information that is not provided in the context? (yes or no)
Are there any inconsistencies between the response and the context? (yes or no)
Count how many questions the user asked. (output a number)
For each question asked by the user, is there a corresponding answer?
Question 1: (yes or no)
Question 2: (yes or no)
...
Question N: (yes or no)
Of the number of questions asked, how many were answered in the answer? (output a number)
"""

messages = [
{'role': 'system', 'content': system_message},
{'role': 'user', 'content': user_message}
]

response = get_completion_from_messages(messages)
return response

evaluation_output = eval_with_rubric(cust_prod_info, assistant_answer)
print(evaluation_output)
```

The assistant's response is based only on the context provided. Yes
The answer does not contain information that is not provided in the context. Yes
There are no inconsistencies between the response and the context. Yes
The user asked 2 questions.
For each question asked by the user, there is a corresponding answer.
Question 1: Yes
Question 2: Yes
Of the number of questions asked, 2 were answered in the answer.

## III. Evaluating the gap between generated answers and standard answers

In classic natural language processing techniques, there are some traditional metrics for measuring the similarity between LLM output and output written by human experts. For example, the BLUE score can be used to measure the similarity between two paragraphs of text.

There is actually a better way, which is to use Prompt. You can specify Prompt and use Prompt to compare the degree of match between the customer service agent response automatically generated by LLM and the ideal response of human.

```python
'''Validation set based on Chinese prompt'''
test_set_ideal = {
'customer_msg': """\
Tell me about the Smartx Pro phone and FotoSnap DSLR camera, the dslr one.\nAlso, what kind of TV do you have here? """,
'ideal_answer':"""\
The SmartX Pro phone is a powerful smartphone with a 6.1-inch display, 128GB storage, 12MP dual cameras, and 5G network support. It costs $899.99 and has a 1-year warranty.
FotThe oSnap DSLR camera is a versatile SLR camera with a 24.2MP sensor, 1080p video capture, a 3-inch LCD screen, and interchangeable lenses. It costs $599.99 and has a 1-year warranty.

We have the following TVs to choose from:

1. CineView 4K TV (Model: CV-4K55) - 55-inch display, 4K resolution, HDR support and smart TV features. It costs $599.99 and has a 2-year warranty.

2. CineView 8K TV (Model: CV-8K65) - 65-inch display, 8K resolution, HDR support and smart TV features. It costs $2999.99 and has a 2-year warranty.

3. CineView OLED TV (Model: CV-OLED55) - 55-inch OLED display, 4K resolution, HDR support and smart TV features. It costs $1499.99 and has a 2-year warranty.
"""
}
```

We first defined a validation set in the above text, which includes a user instruction and a standard answer.

Then we can implement an evaluation function, which uses LLM's understanding ability to require LLM to evaluate whether the generated answer is consistent with the standard answer.

```python
def eval_vs_ideal(test_set, assistant_answer):"""
Evaluate whether a response matches an ideal answer

Parameters:
test_set: test set
assistant_answer: assistant's response
"""
cust_msg = test_set['customer_msg']
ideal = test_set['ideal_answer']
completion = assistant_answer

system_message = """\
You are an assistant evaluating the quality of a customer service agent's response to a user's question by comparing it to an ideal (expert) answer.
Please output a single letter (A, B, C, D, E) and nothing else.
"""

user_message = f"""\
You are comparing a submitted answer to an expert answer for a given question. The data is as follows:
[Start]
************
[Question]: {cust_msg}
************
[Expert answer]: {ideal}
************
[Submitted answer]: {[case]: {completion}
************
[end]

Compare the factual content of the submitted answer with the expert answer, focusing on the content and ignoring differences in style, grammar, or punctuation.
Your focus should be on whether the answer is correct in content; minor differences in content are acceptable.
The submitted answer may be a subset, a superset, or conflict with the expert answer. Identify which applies and answer the question by selecting one of the following options:
(A) The submitted answer is a subset of the expert answer and is identical with it.
(B) The submitted answer is a superset of the expert answer and is identical with it.
(C) The submitted answer contains exactly the same details as the expert answer.
(D) The submitted answer disagrees with the expert answer.
(E) There are differences in the answers, but these differences are not significant from a factual perspective.
Options: ABCDE
"""

messages = [
{'role': 'system', 'content': system_message},
{'role': 'user', 'content': user_message}
]

response = get_competion_from_messages(messages)
return response
```

This rubric comes from the OpenAI open source evaluation framework, which is a great framework that contains many evaluation methods, both contributed by OpenAI developers and the wider open source community.

In this rubric, we ask the LLM to compare the information content of the submission to the expert answer, ignoring differences in style, grammar, and punctuation, but the key is that we ask it to compare and output a score from A to E, depending on whether the submission is a subset, superset, or exact match of the expert answer, which may mean that it made up or fabricated some additional facts.

The LLM will choose the most appropriate description.

The answer generated by LLM is:

```python
print(assistant_answer)
```

Information about SmartX Pro phone and FotoSnap DSLR camera:

1. SmartX Pro phone (model: SX-PP10) is a powerful smartphone with a 6.1-inch display, 128GB storage, 12MP dual camera and 5G network support. It is priced at $899.99 and has a 1-year warranty.

2.The FotoSnap DSLR Camera (Model: FS-DSLR200) ​​is a versatile DSLR camera with a 24.2MP sensor, 1080p video capture, a 3" LCD screen, and interchangeable lenses. It is priced at $599.99 and comes with a 1-year warranty.

Information about the TVs:

We have the following TVs to choose from:

1. CineView 4K TV (Model: CV-4K55) - 55" display, 4K resolution, HDR support, and Smart TV features. It is priced at $599.99 and comes with a 2-year warranty.

2. CineView 8K TV (Model: CV-8K65) - 65" display, 8K resolution, HDR support, and Smart TV features. It is priced at $2999.99 and comes with a 2-year warranty.

3. CineView OLED TV (Model: CV-OLED55) - 55" OLED display, 4K resolution, HDR support, and Smart TV features. The price is $1499.99 and the warranty period is 2 years.

Do you have any further questions about the above products or need to know about other products?

```python
eval_vs_ideal(test_set_ideal, assistant_answer)

```'C'

For this generated answer, GPT judges that the generated content is consistent with the standard answer

```python
assistant_answer_2 = "life is like a box of chocolates"

eval_vs_ideal(test_set_ideal, assistant_answer_2)

```

'D'

For the obviously abnormal answer, GPT judges it as inconsistent

I hope you learned two design patterns from this chapter.

1. Even if there is no ideal answer provided by an expert, as long as an evaluation criterion can be established, one LLM can be used to evaluate the output of another LLM.

2. If you can provide an ideal answer provided by an expert, it can help your LLM better compare whether a specific assistant output is similar to the ideal answer provided by the expert.

I hope this can help you evaluate the output of your LLM system so that you can continuously monitor the performance of the system during development and use these tools to continuously evaluate and improve the performance of the system.

## 4. English version

**1. Ask questions to the question-answering system**

```python
import utils_en

# User message
customer_msg = f"""
tell me about the smartx pro phone and the fotosnap camera, the dslr one.
Also, what TVs or TV related products do you have?"""

# Extract product names from the question
products_by_category = utils_en.get_products_from_query(customer_msg)
# Convert product names to lists
category_and_product_list = utils_en.read_string_to_list(products_by_category)
# Find information about products
product_info = utils_en.get_mentioned_product_info(category_and_product_list)
# Generate answers from information
assistant_answer = utils_en.answer_user_msg(user_msg=customer_msg, product_info=product_info)
```

```python
print( ...
```python
print(assistant_answer = utils_en.answer_user_msg(user_msg=customer_msg, product_info=product_info)


```tant_answer) 
```

Sure! Let me provide you with some information about the SmartX ProPhone and the FotoSnap DSLR Camera.

The SmartX ProPhone is a powerful smartphone with advanced camera features. It has a 6.1-inch display, 128GB storage, a 12MP dual camera, and supports 5G connectivity. The SmartX ProPhone is priced at $899.99 and comes with a 1-year warranty.

The FotoSnap DSLR Camera is a versatile camera that allows you to capture stunning photos and videos. It featuresa 24.2MP sensor, 1080p video recording, a 3-inch LCD screen, and supports interchangeable lenses. The FotoSnap DSLR Camera is priced at $599.99 and also comes with a 1-year warranty.

As for TVs and TV-related products, we have a range of options available. Some of our popular TV models include the CineView 4K TV, CineView 8K TV, and CineView OLED TV. We also have home theater systems like the SoundMax Home Theater and SoundMax Soundbar. Could you please let me know your specific requirements or preferences so that I can assist you better?

**2. Use GPT to evaluate**

```python
# Question, context
cust_prod_info = {
'customer_msg': customer_msg,
'context': product_info
}
```

```python
def eval_with_rubric(test_set, assistant_answer):
"""
Use the GPT API to evaluate the generated answer

Parameters:
test_set: test set
assistant_answer: assistant's reply
"""

cust_msg = test_set['customer_msg']
context = test_set['context']
completion = assistant_answer

# Ask GPT to evaluate the correctness of the answer as an assistant
system_message = """\
You are an assistant that evaluates how well the customer service agent \
answers a user question by looking at the context that the customer service \
agent is using to generate its response. 
"""

# Specific instructions
user_message = f"""\
You are evaluating a submitted answer to a question based on the context \
that the agent uses to answer the question.
Here is the data:
[BEGIN DATA]
************
[Question]: {cust_msg}
************
[Context]: {context}
************
[Submission]: {completion}
************
[END DATA]

Compare the factual content of the submitted answer with the context. \
Ignore any differences in style, grammar, or punctuation.
Answer the following questions:
- Is the Assistant response based only on the context provided? (Y or N)
- Does the answer include information that is not provided in the context? (Y or N)
- Is there any disagreement between the response and the context? (Y or N)
- Count how many questions the user asked. (output a number)
- For each question that the user asked, is there a corresponding answer to it?
Question 1: (Y or N)
Question 2: (Y or N)
...
Question N: (Y or N)
- Of the number of questions asked, how many of these questions were addressed by the answer? (output a number)
"""

messages = [
{'role': 'system', 'content': system_message},
{'role': 'user', 'content': user_message}
]

response = get_completion_from_messages(messages)
return response
```

```python
evaluation_output = eval_with_rubric(cust_prod_info, assistant_answer)
print(evaluation_output)
```

- Is the Assistant response based only on the context provided? (Y or N)
Y

- Does the answer include information that is not provided in the context? (Y or N)
N

- Is there any disagreement between the response and the context? (Y or N)
N

- Count how many questions the user asked. (output a number)
2

- For each question that the user asked, is there a corresponding answer to it?
Question 1: Y
Question 2: Y

- Of the number of questions asked, how many of these questions were addressed by the answer? (output a number)
2

**3. Evaluate the gap between the generated answer and the standard answer**

```python
test_set_ideal = {
'customer_msg': """\
tell me about the smartx pro phone and the fotosnap camera, the dslr one.
Also, what TVs or TV related products do you have?""",
'ideal_answer':"""\
Of course! The SmartX ProPhone is a powerful \
smartphone with advanced camera features. \
For instance, it has a 12MP dual camera. \
Other features include 5G wireless and 128GB storage. \
It also has a 6.1-inch display. The price is $899.99.

The FotoSnap DSLR Camera is great for \
capturing stunning photos and videos. \
Some features include 1080p video, \
3-inch LCD, a 24.2MP sensor, \
and interchangeable lenses. \
The price is 599.99.

For TVs and TV related products, we offer 3 TVs \

All TVs offer HDR and Smart TV.

The CineView 4K TV has vibrant colors and smart features. \
Some of these features include a 55-inch display, \
'4K resolution. It's priced at 599.

The CineView 8K TV is a stunning 8K TV. \
Some features include a 65-inch display and \
8K resolution. It's priced at 2999.99

The CineView OLED TV lets you experience vibrant colors. \
Some features include a 55-inch display and 4K resolution. \
It's priced at 1499.99.

We also offer 2 home theater products, both of which include bluetooth.\
The SoundMax Home Theater is a powerful home theater system for \
an immmersive audio experience.
Its features include 5.1 channel, 1000W output, and wireless subwoofer.
It's priced at 399.99.

The SoundMax Soundbar is a sleek and powerful soundbar.
It's features include 2.1 channel, 300W output, and wireless subwoofer.
It's priced at 199.99

Are there any additional questions you may have about these products \
that you mentioned here?
Or may you have other questions I canhelp you with?
"""
}
```

```python
def eval_vs_ideal(test_set, assistant_answer):
"""
Evaluate whether the response matches the ideal answer

Parameters:
test_set: test set
assistant_answer: assistant's response
"""
cust_msg = test_set['customer_msg']
ideal = test_set['ideal_answer']
completion = assistant_answer

system_message = """\
You are an assistant that evaluates how well the customer service agent \
answers a user question by comparing the response to the ideal (expert) response
Output a single letter and nothing else. 
"""

user_message = f"""\
You are comparing a submitted answer to an expert answer on a given question. Here is the data:
[BEGIN DATA]
************
[Question]: {cust_msg}
************
[Expert]: {ideal}
************
[Submission]: {completion}
************
[END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies. 
Answer the question by selecting one of the following options:
(A) The submitted answer is a subset of the expert answer and is fully consistent with it.
(B) The submitted answer is a superset of the expert answer and is fully consistent with it.
(C) The submitted answer contains all the same details as the expert answer.
(D) There is a disagreement between the submitted answer and the expert answer.
(E) The answers differ, but these differences don't matter from the perspective of factuality.
choice_strings: ABCDE
"""

messages = [
{'role': 'system', 'content': system_message},
{'role': 'user', 'content': user_message}
]

response = get_completion_from_messages(messages)
return response
```

```python
print(assistant_answer)
```

Sure! Let me provide you with some information about the SmartX ProPhone and theFotoSnap DSLR Camera.

The SmartX ProPhone is a powerful smartphone with advanced camera features. It has a 6.1-inch display, 128GB storage, a 12MP dual camera, and supports 5G connectivity. The SmartX ProPhone is priced at $899.99 and comes with a 1-year warranty.

The FotoSnap DSLR Camera is a versatile camera that allows you to capture stunning photos and videos. It features a 24.2MP sensor, 1080p video recording, a 3-inch LCD screen, and supports interchangeable lenses. The FotoSnap DSLR Camera is priced at $599.99 and also comes with a 1-year warranty.

As for TVs and TV-related products, we have a range of options available. Some of our popular TV models include the CineView 4K TV, CineView 8K TV, and CineView OLED TV. We also have home theater systems like the SoundMax Home Theater and SoundMax Soundbar. Could you please let me know your specific requirements or preferences so that I can assist you better?

```python
# Due to the update of the model, it is no longer possible to correctly judge on the original Prompt
eval_vs_ideal(test_set_ideal, assistant_answer)
```

'D'

```python
assistant_answer_2 = "life is like a box of chocolates"
```

```python
eval_vs_ideal(test_set_ideal, assistant_answer_2)
# For obviously abnormal answers, GPT judges them as inconsistent
```

'D'