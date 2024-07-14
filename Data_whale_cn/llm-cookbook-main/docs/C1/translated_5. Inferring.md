# Chapter 5 Inference

In this chapter, we will lead you through a story to understand how to infer sentiment and topics from product reviews and news articles.

Let's imagine that you are a data analyst at a startup, and your task is to extract key sentiment and topics from various product reviews and news articles. These tasks include label extraction, entity extraction, and understanding the sentiment of text. In the traditional machine learning process, you need to collect labeled datasets, train models, determine how to deploy models in the cloud, and perform inference. Although this approach may produce good results, it takes a lot of time and effort to complete this entire process. Moreover, each task, such as sentiment analysis, entity extraction, etc., requires training and deploying a separate model.

However, just when you are ready to do the heavy work, you discover large language models (LLMs). An obvious advantage of LLMs is that for many of these tasks, you only need to write a prompt to start generating results, which greatly reduces your workload. This discovery is like finding a magic key that speeds up application development a lot. What’s most exciting is that you can use just one model and one API to perform many different tasks, without having to worry about how to train and deploy many different models.

Let’s start learning this chapter and explore how to use LLM to speed up our work process and improve our work efficiency.

##1. Sentiment Inference

### 1.1 Sentiment Analysis

Let's take a review of a lamp on an e-commerce platform as an example. Through this example, we will learn how to classify the sentiment of the review into two categories (positive/negative).

```python
lamp_review = """
I need a nice bedroom lamp with extra storage and a reasonable price.\
I received it very quickly. During the shipping process, our lamp cord broke, but the company was happy to send a new one.\
It arrived a few days later. The lamp was easy to assemble. I found a part missing, so I contacted their customer service and they sent me the missing part very quickly!\
In my opinion, Lumina is an excellent company that cares about its customers and products!
"""
```

Next, we will try to write a prompt to classify the sentiment of this product review. If we want the system to parse the sentiment of this review, just write a prompt like "What is the sentiment of the following product review?", plus some standard separators and comment text, etc.

Then, we run this program once. The results show that the sentiment of this product review is positive, which seems to be quite accurate. Although the lamp is not perfect, the customer seems to be quite satisfied with it. This company seems to be very serious about customer experience and product quality, so it seems to be a correct judgment to assign a positive sentiment to the review..

```python
from tool import get_completion

prompt = f"""
What is the sentiment of the following product review separated by three backticks?

Review text: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)
```

The sentiment is positive.

If you want to give a more concise answer so that it is easier to process later, you can add another command to the above prompt: *Answer with one word: "positive" or "negative"*. This will only print the word "positive", which makes the output more uniform and convenient for subsequent processing.

```python
prompt = f"""
What is the sentiment of the following product review separated by three backticks?

Answer with one word: "positive" or "negative".

Review text: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)
```

Positive

### 1.2 Identifying sentiment types

Next, we will continue to use the previous lamp review, but this time we will try anew prompt. We want the model to identify the sentiment expressed by the author of the review and organize these sentiments into a list of no more than five items.

```python
# Chinese
prompt = f"""
Identify the sentiment expressed by the author of the following review. Contains no more than five items. Format the answer as a comma-separated list of words.

Review text: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)
```

Satisfied, grateful, appreciated, trusted, satisfied

Large language models are very good at extracting specific things from a piece of text. In the example above, the sentiment expressed by the review helps to understand how customers view a specific product.

### 1.3 Identifying anger

For many businesses, it is critical to understand the anger of customers. This leads to a classification problem: Is the author of the following review angry? Because if someone is really emotional, it may mean that extra attention is needed, because every angry customer is an opportunity to improve service and an opportunity to improve the company's reputation. At this time, the customer support or customer service team should intervene, contact the customer, understand the specific situation, and then solve their problem.

```python
# Chinese
prompt =f"""
Does the author of the following review express anger? Comments are separated by three backticks. Give a yes or no answer.

Comment text: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)
```

No

In the example above, the customer is not angry. Note that if you want to build all these classifiers using regular supervised learning, it is impossible to do this in a few minutes. We encourage you to try changing some of these prompts, perhaps asking if the customer expressed joy, or asking if there are any missing parts, and see if you can get the prompt to make different inferences about this lamp review.

## 2. Information Extraction

### 2.1 Product Information Extraction

Information extraction is an important part of natural language processing (NLP), which helps us extract specific, relevant information from text. We will dig deeper into the rich information in customer reviews. In the following example, we will ask the model to identify two key elements: the product purchased and the manufacturer of the product.

Imagine if you were trying to analyze numerous reviews on an online e-commerce site. Knowing what products are mentioned in the reviews, who made them, and the associated positive or negative sentiment would greatly help you track the sentiment of a particular product or manufacturer in the minds of users.trend.

In the following example, we ask the model to present the response as a JSON object, where the key is the product and brand.

```python
# Chinese
prompt = f"""
Identify the following items from the review text:
- The item purchased by the reviewer
- The company that manufactured the item

The review text is delimited by three backticks. Format your response as a JSON object with "item" and "brand" as the keys.

If the information does not exist, use "unknown" as the value.

Keep your response as short as possible.

Review text: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)
```

{
"item": "bedroom lamp",
"brand": "Lumina"
}

As shown above, it will say that this item is a bedroom lamp and the brand is Luminar. You can easily load it into a Python dictionary and then do other processing on this output.

### 2.2 Comprehensive sentiment inference and information extraction

In the above section, we used three to four prompts to extract information such as "emotional tendency", "anger", "item type" and "brand" in the comments.However, we can actually design a single prompt to extract all of this information at once.

```python
# Chinese
prompt = f"""
Identify the following items from the review text:
- Sentiment (positive or negative)
- Did the reviewer express anger? (yes or no)
- Item purchased by the reviewer
- The company that manufactured the item

Reviews are delimited by three backticks. Format your response as a JSON object with "sentiment", "angry", "item type", and "brand" as keys.
If the information does not exist, use "unknown" as the value.
Keep your response as short as possible.
Format the "angry" value as a boolean.

Review text: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)
```

{
"sentiment": "positive",
"angry": false,
"item type": "bedroom lamp",
"brand": "Lumina"
}

In this example, we instruct LLM to format the "angry" case as a boolean and output JSON format. You can try various variations of the formatting pattern.ization, or experiment with completely different reviews to see if the LLM can still accurately extract these.

## 3. Topic Inference

Another cool application of large language models is to infer topics. Suppose we have a long text, how can we tell what the main idea of ​​this text is? What topics does it cover? Let's take a closer look at the following fictional newspaper report.

```python
# Chinese
story = """
In a recent government survey, public sector employees were asked to rate their satisfaction with their department.
The survey results showed that NASA was the most popular department, with a satisfaction rate of 95%.

John Smith, a NASA employee, commented on the findings, saying:
"I'm not surprised that NASA ranked first. It's a great place to work with amazing people and incredible opportunities. I'm proud to be a part of such an innovative organization."

NASA's management team also welcomed the results, with Director Tom Johnson saying:
"We're happy to hear that our employees are satisfied with their work at NASA.
We have a talented, committed team who work tirelessly to achieve our goals, and it’s great to see their hard work pay off.”

The survey also showed that the Social Security Administration had the lowest satisfaction rating, with only 45% of employees sayingindicating that they are satisfied with their jobs.
The government has pledged to address the issues raised by employees in the survey and work to improve job satisfaction in all departments.
"""
```

### 3.1 Infer discussion topics

The above is a fictional newspaper article about government employees' feelings about their work unit. We can ask the big language model to identify the five topics discussed in it and summarize each topic in one or two words. The output will be presented as a comma-delimited Python list.

```python
# Chinese
prompt = f"""
Identify the five topics discussed in the given text below.

Each topic is summarized in 1-2 words.

Please output a parseable Python list where each element is a string showing a topic.

Given text: ```{story}```
"""
response = get_completion(prompt)
print(response)
```

['NASA', 'Satisfaction', 'Comments', 'Management Team', 'Social Security Administration']

### 3.2 Making news alerts for specific topics

Suppose we have a news website or similar platform, and this is the topic we are interested in: NASA, local government, engineering, employee satisfaction, federal government, etc. We want to analyze a news article and understand what topics it contains. We can use such a Prompt: Determine whether each item in the following list of topics is a topic in the following text. Give a list of answers in the form of 0 or 1.

```python
# Chinese
prompt = f"""
Determine whether each item in the list of topics is a topic in the given text,

Give the answer in the form of a list, each element is a Json object, the key is the corresponding topic, and the value is the corresponding 0 or 1.

List of topics: NASA, local government, engineering, employee satisfaction, federal government

Given text: ```{story}```
"""
response = get_completion(prompt)
print(response)
```

[
{"NASA": 1},
{"Local Government": 1},
{"Engineering": 0},
{"Employee Satisfaction": 1},
{"Federal Government": 1}
]

From the output, this `story` is related to "NASA", "Employee Satisfaction", "Federal Government", "Local Government", but not to "Engineering". This capability is called zero-shot learning in the field of machine learning. This is because we do not provide any labeled training data, only Prompt , it can determine which topics are covered in the news article.

If we want to create a news alert, we can also apply this process to processing news. Suppose I am very interested in the work of "NASA", then you can build a system like this: whenever there is news related to "NASA", the system will output an alert.

```python
result_lst = eval(response)
topic_dict = {list(i.keys())[0] : list(i.values())[0] for i in result_lst}
print(topic_dict)
if topic_dict['NASA'] == 1:
print("Alert: New news about NASA")
```

{'NASA': 1, 'Local Government': 1, 'Engineering': 0, 'Employee Satisfaction': 1, 'Federal Government': 1}
Alert: New news about NASA

That's our comprehensive introduction to inference. In just a few minutes, we have been able to build multiple systems for text reasoning, a task that previously took machine learning experts days or even weeks to complete. This change is undoubtedly exciting because no matter what you are doing,Whether you are an experienced machine learning developer or a beginner, you can use the input prompt to quickly start complex natural language processing tasks.

## English version

**1.1 Sentiment Analysis**

```python
lamp_review = """
Needed a nice lamp for my bedroom, and this one had \
additional storage and not too high of a price point. \
Got it fast. The string to our lamp broke during the \
transit and the company happily sent over a new one. \
Came within a few days as well. It was easy to put \
together. I had a missing part, so I contacted their \
support and they very quickly got me the missing piece! \
Lumina seems to me to be a great company that cares \
about their customers and products!!
"""
```

```python
prompt = f"""
What is the sentiment of the following product review, 
which is delimited with triple backticks?

Review text: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)
```

The sentiment of the product review is positive.

```python
prompt = f"""
What is the sentiment of the following product review, 
which is delimited with triple backticks?

Give youranswer as a single word, either "positive" \
or "negative".

Review text: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)
```

positive

**1.2 Identify emotion types**

```python
prompt = f"""
Identify a list of emotions that the writer of the \
following review is expressing. Include no more than \
five items in the list. Format your answer as a list of \
lower-case words separated by commas.

Review text: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)```

satisfied, pleased, grateful, impressed, happy

**1.3 Identify anger**

```python
prompt = f"""
Is the writer of the following review expressing anger?\
The review is delimited with triple backticks. \
Give your answer as either yes or no.

Review text: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)
```

No

**2.1 Product information extraction**

```python
prompt = f"""
Identify the following items from the review text: 
- Item purchased by reviewer
- Company that made the item

Thereview is delimited with triple backticks. \
Format your response as a JSON object with \
"Item" and "Brand" as the keys. 
If the information isn't present, use "unknown" \
as the value.
Make your response as short as possible.

Review text: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)
```

{
"Item": "lamp",
"Brand": "Lumina"
}

**2.2 Comprehensive sentiment inference and information extraction**

```python
prompt = f"""
Identify the following items from the review text: 
- Sentiment (positive or negative)
- Is the reviewer expressing anger? (true or false)
- Item purchased by reviewer
- Company that made the item

The review is delimited with triple backticks. \
Format your response as a JSON object with \
"Sentiment", "Anger", "Item" and "Brand" as the keys.
If the information isn't present, use "unknown" \
as the value.
Make your response as short as possible.
Format the Anger value as a boolean.

Review text: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)
```

{
"Sentiment": "positive",
"Anger": false,
"Item": "lamp",
"Brand": "Lumina"
}

**3.1 Inferring discussion topics**

```python
story = """
In a recent survey conducted by the government, 
public sector employees were asked to rate their level 
of satisfaction with the department they work at. 
The results revealed that NASA was the most popular 
department with a satisfaction rating of 95%.

One NASA employee, John Smith, commented on the findings, 
stating, "I'm not surprisedised that NASA came out on top. 
It's a great place to work with amazing people and 
incredible opportunities. I'm proud to be a part of 
such an innovative organization."

The results were also welcomed by NASA's management team, 
with Director Tom Johnson stating, "We are thrilled to 
hear that our employees are satisfied with their work at NASA. 
We have a talented and dedicated team who work tirelessly 
to achieve our goals, and it's fantastic to see that their 
hard work is paying off."

ThThe survey also revealed that the Social Security Administration had the lowest satisfaction rating, with only 45% of employees indicating they were satisfied with their job. The government has pledged to address the concerns raised by employees in the survey and work towards improving job satisfaction across all departments.
"""
```

```python
prompt = f"""
Determine five topics that are being discussed in the \
following text, which is delimited by triple backticks.

Make each item one ortwo words long. 

Format your response as a list of items separated by commas.

Give me a list which can be read in Python.

Text sample: ```{story}```
"""
response = get_completion(prompt)
print(response)
```

survey, satisfaction rating, NASA, Social Security Administration, job satisfaction

```python
response.split(sep=',')
```

['survey',
' satisfaction rating',
' NASA',
' Social Security Administration',
' job satisfaction']

**3.2 Make news alerts for specific topics**

```pythonn
topic_list = [
"nasa", "local government", "engineering", 
"employee satisfaction", "federal government"
]
```

```python
prompt = f"""
Determine whether each item in the following list of \
topics is a topic in the text below, which
is delimited with triple backticks.

Give your answer as a list with 0 or 1 for each topic.\

List of topics: {", ".join(topic_list)}

Text sample: ```{story}```
"""
response = get_completion(prompt)
print(response)
```

[1, 0, 0, 1, 1]

```python
topic_dict = {topic_list[i] : eval(response)[i] for i in range(len(eval(response)))}
print(topic_dict)
if topic_dict['nasa'] == 1:
print("ALERT: New NASA story!")
```

{'nasa': 1, 'local government': 0, 'engineering': 0, 'employee satisfaction': 1, 'federal government': 1}
ALERT: New NASA story!