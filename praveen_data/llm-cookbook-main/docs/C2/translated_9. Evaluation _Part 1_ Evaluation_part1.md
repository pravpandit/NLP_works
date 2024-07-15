# Chapter 9 Evaluation (Part 1) - When there is a simple correct answer

In the past chapters, we showed you how to build applications with LLM, including evaluating inputs, processing inputs, and doing a final check of results before presenting them to users. However, once we have built such a system, how can we know whether it is working well? Furthermore, once we deploy it and let users start using it, how can we track its performance, find possible problems, and continuously improve its answer quality? In this chapter, we will share with you some best practices for evaluating LLM output.

Building LLM-based applications is different from building traditional supervised learning applications. Because you can build LLM-based applications quickly, evaluation usually does not start with a test set. Instead, you gradually build up a set of test examples.

In traditional supervised learning settings, you need to collect training sets, development sets, or set aside cross-validation sets, which are used throughout the development process. However, if you can define a prompt in a few minutes and get feedback in a few hours, then stopping to collect a thousand test examples is extremely tedious. Because now, you can get results without any training samples.

So when building an application with LLM, you might go through the following process: First, you tweak Prompt in a small sample of one to three samples, trying to make itThen, as you test the system further, you may run into some tricky examples that can't be solved by Prompt or the algorithm. This is the challenge facing developers building applications using the ChatGPT API. In this case, you can add these extra few examples to the set you are testing, organically adding other difficult examples. Eventually, you will add enough of these examples to your gradually expanding development set that it becomes a bit inconvenient to manually run every example to test Prompt. Then you start to develop some metrics for measuring the performance of these small sample sets, such as average accuracy. The interesting thing about this process is that if you feel that your system is good enough, you can always stop and stop improving it. In fact, many deployed applications stop at the first or second step, and they work very well.

It is worth noting that many applications of large models do not have substantial risks, even if it does not give completely correct answers. However, for some high-risk applications, if there is bias or inappropriate output that could cause harm to someone, it is particularly important to collect test sets, rigorously evaluate the performance of the system, and ensure that it does the right thing before using it. However, if you are only using it to summarize the article for your own reading, rather than for others, then the risk is likely to be smaller and you can stop early in the process., without having to pay the huge cost of collecting large-scale data sets.

Now let's move on to a more practical application stage and put the theoretical knowledge we have just learned into practice. Let's study some real data together, understand its structure and use our tools to analyze them. In our case, we get a set of classification information and its product name. Let's execute the following code to see these classification information and its product name

```python
import utils_zh

products_and_category = utils_zh.get_products_and_category()
products_and_category
```

{'Computers and Notebooks': ['TechPro Ultrabook',
'BlueWave Gaming Book',
'PowerLite Convertible',
'TechPro Desktop',
'BlueWave Chromebook'],
'Smartphones and Accessories': ['SmartX ProPhone'],
'Professional Mobile Phones': ['MobiTech PowerCase',
'SmartX MiniPhone','MobiTech Wireless Charger',
'SmartX EarBuds'],
'TVs and Home Theater Systems': ['CineView 4K TV',
'SoundMax Home Theater',
'CineView 8K TV',
'SoundMax Soundbar',
'CineView OLED TV'],
'Game Consoles and Accessories': ['GameSphere X',
'ProGamer Controller',
'GameSphere Y',
'ProGamer Racing Wheel',
'GameSphere VR Headset'],
'Audio Devices': ['AudioPhonic Noise-Canceling Headphones',
'WaveSound Bluetooth Speaker',
'AudioPhonic True Wireless Earbuds',
'WaveSound Soundbar',
'AudioPhonic Turntable'],
'Cameras and Video Cameras': ['FotoSnap DSLR Camera',
'ActionCam 4K',
'FotoSnap Mirrorless Camera',
'ZoomMaster Camcorder',
'FotoSnap Instant Camera']}

## 1. Find relevant product and category names

When we develop, we often need to process and parse user input. Especially in the e-commerce field, there may be various user queries, such as: "I want the most expensive computer". We need a tool that can understand this context and give relevant products and categories. The following code implements this function.

First, we define a function `find_category_and_product_v1`, the main purpose of this function is to parse products and categories from user input. This function requires two parameters: `user_input` represents the user's query, and `products_and_category` is a dictionary containing information about product types and corresponding products.

At the beginning of the function, we define a separatorThe delimiter is used to separate content in customer service queries. Then, we create a system message. This message mainly explains how the system works: users will provide customer service queries, which are separated by the delimiter. The system will output a Python list, and each object in the list is a Json object. Each object will contain two fields, 'category' and 'name', corresponding to the category and name of the product respectively.

We create a list called `messages` to store these sample conversations and user queries. Finally, we use the `get_completion_from_messages` function to process these messages and return the processing results.

Through this code, we can see how to understand and process user queries through conversations to provide a better user experience.

```python
from tool import get_completion_from_messages

def find_category_and_product_v1(user_input,products_and_category):
"""
Get products and categories from user input

Parameters:
user_input: user's query
products_and_category: Dictionary of product types and corresponding products
"""

delimiter = "####"
system_message = f"""
You will provide a customer service query. \
Customer service queries will be delimited by the {delimiter} character.
Output a Python list where each object in the list is a Json object, each in the following format:
'category': <One of Computers & Laptops, Smartphones & Accessories, TVs & Home Theater Systems, \
Game Consoles & Accessories, Audio Equipment, Cameras & Camcorders>,
and
'name': <One of the products that must be found in the list of allowed products below>

Where category and product must be found in the customer service query.
If a product is mentioned, it must be associated with the correct category from the list of allowed products below.
If no product or category is found, output an empty list.

List all relevant products based on their relevance to the customer service query by product name and product category.
Do not assume any characteristics or properties, such as relative quality or price, from the product's name.

Allowed products are provided in JSON format.The key of each item represents a category.
The value of each item is a list of products in that category.
Allowed products: {products_and_category}

"""

few_shot_user_1 = """I want the most expensive computer. """
few_shot_assistant_1 = """
[{'category': 'Computers and Notebooks', \
'products': ['TechPro Ultrabook', 'BlueWave Gaming Book', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
"""

messages = [
{'role':'system', 'content': system_message},
{'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"},
{'role':'assistant', 'content':few_shot_assistant_1 },
{'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"}, 
] 
return get_completion_from_messages(messages)
```

## 2. Evaluate on some queries

For the above system, we can first evaluate on some simple queries:

```python
# The first query to evaluate
customer_msg_0 = f"""If my budget is limited, which TV can I buy?"""

products_by_category_0 = find_category_and_product_v1(customer_msg_0,
products_and_category)
print(products_by_category_0)
```

[{'category': 'TV and Home Theater System', 'products': ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']}]

The correct answer is output.

```python
customer_msg_1 = f"""I need a smartphone charger"""

products_by_category_1 = find_category_and_product_v1(customer_msg_1,
products_and_category)
print(products_by_category_1)
```

[{'category': 'Smartphones and Accessories', 'products': ['MobiTech PowerCase', 'SmartX MiniPhone', 'MobiTech Wireless Charger', 'SmartX EarBuds']}]

The output isCorrect answer.

```python
customer_msg_2 = f"""
What computers do you have?"""

products_by_category_2 = find_category_and_product_v1(customer_msg_2,
products_and_category)
products_by_category_2
```

" \n [{'category': 'Computers and Notebooks', 'products': ['TechPro Ultrabook', 'BlueWave Gaming Book', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]"

The output answer is correct, but the format is incorrect.

```python
customer_msg_3 = f"""
Tell me about the smartx pro phone and fotosnap camera, which DSLR.
I have a limited budget, what cost-effective TV do you recommend? """

products_by_category_3 = find_category_and_product_v1(customer_msg_3,
products_and_category)
print(products_by_category_3)
```

[{'category': 'Smartphones and Accessories', 'products': ['SmartX ProPhone']}, {'category': 'Cameras and Camcorders', 'products': ['FotoSnap DSLR Camera']}]

[{'category': 'TVs and Home Theater Systems', 'products': ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']}]

It looks like it is outputting the correct data, but not as expected.This makes it more difficult to parse it into a Python list of dictionaries.

## 3. More difficult test cases

Next, we can give some queries where the model does not perform as expected in actual use.

```python
customer_msg_4 = f"""
Tell me about the CineView TV, the 8K one, and the Gamesphere, model X.
I'm on a budget, what computers do you have?"""

products_by_category_4 = find_category_and_product_v1(customer_msg_4,products_and_category)
print(products_by_category_4)
```

[{'category': 'TVs and Home Theater Systems', 'products': ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']}]
[{'category': 'Game Consoles and Accessories', 'products': ['GameSphere X', 'ProGamer Controller', 'GameSphere Y', 'ProGamer Racing Wheel', 'GameSphere VR Headset']}]
[{'category': 'Computers and Notebooks', 'products': ['TechPro Ultrabook', 'BlueWave Gaming Book', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]

## 4. Modify the instructions to handle difficult test cases

In summary, the initial version we implemented did not perform well in some of the above test cases.

To improve the performance, we added the following to the prompt: Do not output any additional text that is not in JSON format, and added a second example to use user and assistant messages for few-shot prompts.

```python
def find_category_and_product_v2(user_input,products_and_category):
"""
Get products and categories from user inputAdded: Don't output any extra text that doesn't conform to JSON format.

Added a second example (for a few-shot prompt) where the user asks for the cheapest computer.

In both few-shot examples, the response shown is just the full list of products in JSON format.

Parameters:

user_input: the user's query

products_and_category: a dictionary of product types and corresponding products

"""

delimiter = "####"

system_message = f"""

You will provide a customer service query. \

Customer service queries will be delimited by the {delimiter} character.

Output a Python list where each object in the list is a JSON object, each in the following format:

'category': <one of Computers and Laptops, Smartphones and Accessories, TVs and Home Theater Systems, \

Game Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,

and

'name': <a list of products that must be found in the allowed products below>

Don't output any extra text that isn't in JSON format.

Output the requested JSON, do not write any explanatory text.

The categories and products must be found in the customer service query.
If a product is mentioned, it must be associated with the correct category from the list of allowed products below.
If no product or category is found, output an empty list.

List all relevant products based on their relevance to the customer service query by product name and product category.
Do not assume any characteristics or properties, such as relative quality or price, from the product's name.

The allowed products are provided in JSON format.
The key of each item represents a category.
The value of each item is a list of products in that category.
Allowed products: {products_and_category}

"""

few_shot_user_1 = """I want the most expensive computer. Which one do you recommend? """

few_shot_assistant_1 = """
[{'category': 'Computers and Notebooks', \
'products': ['TechPro Ultrabook', 'BlueWave Gaming Book', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
"""

few_shot_user_2 = """I want the cheapest computer. Which one do you recommend? """
few_shot_assistant_2 = """
[{'category': 'Computers and Laptops', \
'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
"""

messages = [
{'role':'system', 'content': system_message},
{'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"},
{'role':'assistant', 'content': few_shot_assistant_1 },
{'role':'user','content': f"{delimiter}{few_shot_user_2}{delimiter}"}, 
{'role':'assistant', 'content': few_shot_assistant_2 },
{'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"}, 
] 
return get_completion_from_messages(messages)
```

## 5. Evaluate the modified instructions on a difficult test case

We can evaluate the effect of the improved system on a difficult test case that did not perform as expected before:

```python
customer_msg_3 = f"""
Tell me about the smartx pro phone and the fotosnap camera, the DSLR one.
Also, what TVs do you have?"""

products_by_category_3 = find_category_and_product_v2(customer_msg_3,products_and_category)
print(products_by_category_3)
```

[{'category': 'Smartphones and Accessories', 'products': ['SmartX ProPhone']}, {'category': 'Cameras and Camcorders', 'products': ['FotoSnap DSLR Camera', 'ActionCam 4K', 'FotoSnap Mirrorless Camera', 'ZoomMaster Camcorder', 'FotoSnap Instant Camera']}, {'category': 'TVs and Home Theater Systems', 'products': ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']}]

## VI. Regression testing: Verify that the model still works on previous test cases

Check and fixRepeat the model to improve the performance of hard-to-test cases while ensuring that this correction does not negatively impact the performance of previous test cases.

```python
customer_msg_0 = f"""If I have a limited budget, which TV can I buy?"""

products_by_category_0 = find_category_and_product_v2(customer_msg_0,
products_and_category)
print(products_by_category_0)
```

[{'category': 'TV and Home Theater Systems', 'products': ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']}]

## 7. Collect development sets for automated testing

As our applications mature, the importance of testing increases. Usually, when we only process a small number of samples, we can manuallyRunning tests and evaluating the results is feasible. However, as the development set grows, this approach becomes cumbersome and inefficient. At this point, we need to introduce automated testing to improve our work efficiency. Below, we will start writing code to automate the testing process, which can help you improve efficiency and ensure the accuracy of the test.

Below are some standard answers to user questions, which are used to evaluate the accuracy of LLM answers, which is equivalent to the role of the validation set in machine learning.

```python
msg_ideal_pairs_set = [

# eg 0
{'customer_msg':"""If I have a limited budget, what kind of TV can I buy?""",
'ideal_answer':{
'TV and home theater system':set(
['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']
)}
},

# eg 1
{'customer_msg':"""I need a charger for my smartphone""",
'ideal_answer':{
'Smartphones and accessories':set(
['MobiTech PowerCase', 'MobiTech Wireless Charger', 'SmartX EarBuds']
)}
},
# eg 2
{'customer_msg':f"""What kind of computer do you have""",
'ideal_answer':{
'Computers and laptops':set(
['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook'
])
}
},

# eg 3
{'customer_msg':f"""Tell me about the smartx pro phone and the fotosnap camera, the DSLR.\
Also, what TVs do you have?""",
'ideal_answer':{'Smartphones and accessories':set(
['SmartX ProPhone']),
'Cameras and camcorders':set(
['FotoSnap DSLR Camera']),
'TVs and home theater systems':set(
['CineView 4K TV', 'SoundMax Home Theater','CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV'])
}
}, 

# eg 4
{'customer_msg':"""Tell me about CineView TV, the 8K TV, \
Gamesphere, and X-console. I'm on a budget, what computers do you have?""",
'ideal_answer':{
'TVs and home theater systems':set(
['CineView 8K TV']),
'Game consoles and accessories':set(
['GameSphere X']),
'Computers and laptops':set(
['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook'])
}
},

# eg 5
{'customer_msg':f"""What smartphones do you have?""",
'ideal_answer':{
'Smartphones and accessories':set(
['SmartX ProPhone', 'MobiTech PowerCase', 'SmartX MiniPhone', 'MobiTech Wireless Charger', 'SmartX EarBuds'
])
}
},
# eg 6
{'customer_msg':f"""I have a limited budget. Can you recommend me someRecommend some smartphones? """,
'ideal_answer':{
'Smartphones and accessories':set(
['SmartX EarBuds', 'SmartX MiniPhone', 'MobiTech PowerCase', 'SmartX ProPhone', 'MobiTech Wireless Charger']
)}
},

# eg 7 # this will output a subset of the ideal answer
{'customer_msg':f"""What game consoles are suitable for my friends who like racing games? """,
'ideal_answer':{
'Game consoles and accessories':set([
'GameSphere X',
'ProGamer Controller',
'GameSphere Y',
'ProGamer Racing Wheel',
'GameSphere VR Headset'
])}
},
# eg 8
{'customer_msg':f"""What gift would be suitable for my friend who is a videographer?""",
'ideal_answer': {
'Cameras and camcorders':set([
'FotoSnap DSLR Camera', 'ActionCam 4K', 'FotoSnap Mirrorless Camera', 'ZoomMaster Camcorder', 'FotoSnap Instant Camera'
])}
},

# eg 9
{'customer_msg':f"""I want a hot tub time machine""",
'ideal_answer': []
}

]

```

## 8. Evaluate test cases by comparing with ideal answers

We evaluate the accuracy of LLM's answers through the following function `eval_response_with_ideal`, which evaluates the accuracy of LLM's answers by comparing LLM The answers are compared with the ideal answers to evaluate the system's performance on the test case.

```pythonimport json
def eval_response_with_ideal(response,
ideal,
debug=False):
"""
Evaluate whether the response matches the ideal answer

Parameters:
response: the content of the response
ideal: the ideal answer
debug: whether to print debugging information
"""
if debug:
print("Response:")
print(response)

# json.loads() can only parse double quotes, so replace single quotes with double quotes here
json_like_str = response.replace("'",'"')

# Parse into a series of dictionaries
l_of_d = json.loads(json_like_str)

# When the response is empty, that is, no products are found
if l_of_d == [] andideal == []:
return 1

# Another abnormal situation is that the number of standard answers does not match the number of reply answers
elif l_of_d == [] or ideal == []:
return 0

# Count the number of correct answers
correct = 0

if debug:
print("l_of_d is")
print(l_of_d)

# For each question and answer pair
for d in l_of_d:

# Get products and catalogs
cat = d.get('category')
prod_l = d.get('products')
# Get products and catalogs
if cat and prod_l:
# convert list to set for comparison
prod_set = set(prod_l)
#get ideal set of products
ideal_cat = ideal.get(cat)
if ideal_cat:
prod_set_ideal = set(ideal.get(cat))
else:
if debug:
print(f"No directory found in the standard answer {cat}")
print(f"Standard answer: {ideal}")
continue

if debug:
print("Product set:\n",prod_set)
print()
print("Standard answer product set:\n",prod_set_ideal)

# The product set found is consistent with the standard product setif prod_set == prod_set_ideal:
if debug:
print("correct")
correct +=1
else:
print("wrong")
print(f"product set: {prod_set}")
print(f"standard product set: {prod_set_ideal}")
if prod_set <= prod_set_ideal:
print("the answer is a subset of the standard answer")
elif prod_set >= prod_set_ideal:
print("the answer is a superset of the standard answer")

# Calculate the number of correct answers
pc_correct = correct / len(l_of_d)

return pc_correct
```

We use one of the above test cases for testing. First, let's look at the standard answer:

```python
print(f'User question: {msg_ideal_pairs_set[7]["customer_msg"]}')
print(f'Standard answer: {msg_ideal_pairs_set[7]["ideal_answer"]}')
```

User question: Which game consoles are suitable for my friends who like racing games?
Standard answer: {'Game consoles and accessories': {'ProGamer Racing Wheel', 'ProGamer Controller', 'GameSphere Y', 'GameSphere VR Headset', 'GameSphere X'}}

Compare with LLM's answer and use the verification function to score:

```python
response = find_category_and_product_v2(msg_ideal_pairs_set[7]["customer_msg"],products_and_category)
print(f'Answer: {response}')

eval_response_with_ideal(response,
msg_ideal_pairs_set[7]["ideal_answer"])

```

Answer: 
[{'category': 'Game Consoles and Accessories', 'products': ['GameSphere X', 'ProGamer Controller', 'GameSphere Y', 'ProGamer Racing Wheel', 'GameSphere VR Headset']}]

1.0

It can be seen that the scoring of this verification function is accurate.

## 9. Run the evaluation on all test cases and calculate the correct case ratio

Next, we will verify all the questions in the test case and calculate the accuracy of LLM's correct answers

> Note: If any API call times out, it will not run

```python
import timescore_accum = 0
for i, pair in enumerate(msg_ideal_pairs_set):
time.sleep(20)
print(f"例 {i}")

customer_msg = pair['customer_msg']
ideal = pair['ideal_answer']

# print("Customer message",customer_msg)
# print("ideal:",ideal)
response = find_category_and_product_v2(customer_msg,
products_and_category)

# print("products_by_category",products_by_category)
score = eval_response_with_ideal(response,ideal,debug=False)
print(f"{i}: {score}")
score_accum += score

n_examples = len(msg_ideal_pairs_set)
fraction_correct = score_accum / n_examples
print(f"The correct ratio is {n_examples}: {fraction_correct}")
```

Example 0
0: 1.0
Example 1
Error
Product collection: {'SmartX ProPhone'}
Standard product collection: {'MobiTech Wireless Charger', 'SmartX EarBuds', 'MobiTech PowerCase'}
1: 0.0
Example 2
2: 1.0
Example 3
3: 1.0
Example 4
Error
Product collection: {'SoundMax Home Theater', 'CineView 8K TV', 'CineView 4KTV', 'CineView OLED TV', 'SoundMax Soundbar'}
Standard product set: {'CineView 8K TV'}
The answer is a superset of the standard answer
Wrong
Product set: {'ProGamer Racing Wheel', 'ProGamer Controller', 'GameSphere Y', 'GameSphere VR Headset', 'GameSphere X'}
Standard product set: {'GameSphere X'}
The answer is a superset of the standard answer
Wrong
Product set: {'TechPro Ultrabook', 'TechPro Desktop', 'BlueWave Chromebook', 'PowerLite Convertible', 'BlueWave Gaming'}
Standard product set: {'TechPro Desktop', 'BlueWave Chromebook', 'TechPro Ultrabook', 'PowerLite Convertible', 'BlueWave GamingLaptop'}
4: 0.0
Example 5
Wrong
Product set: {'SmartX ProPhone'}
Standard product set: {'MobiTech Wireless Charger', 'SmartX EarBuds', 'SmartX MiniPhone', 'SmartX ProPhone', 'MobiTech PowerCase'}
The answer is a subset of the standard answer
5: 0.0
Example 6
Wrong
Product set: {'SmartX ProPhone'}
Standard product set: {'MobiTech Wireless Charger', 'SmartX EarBuds', 'SmartX MiniPhone', 'SmartX ProPhone', 'MobiTech PowerCase'}
The answer is a subset of the standard answer
6: 0.0
Example 7
7: 1.0
Example 8
8: 1.0
Example 9
9: 1
The correct percentage is 10: 0.6

Using PromptThe workflow for building an application is very different from the workflow for building an application using supervised learning. So we think this is a good thing to keep in mind, and it will feel like you are iterating much faster when you are building supervised learning models.

If you haven't experienced this yourself, you might be surprised at how efficient evaluation methods can be with just a few samples that you have manually constructed. You might think that just 10 samples is not statistically significant. But when you really use this method, you might be surprised at the improvement in performance that can be achieved by adding some complex samples to the development set. This is very helpful in helping you and your team find effective prompts and effective systems.

In this chapter, the output can be evaluated quantitatively, just like there is an expected output, and you can tell whether it gave this expected output. In the next chapter, we will explore how to evaluate our output in more ambiguous situations. That is, situations where the correct answer may not be so clear.

## X. English version

**1. Find the product and category names**

```python
import utils_en

products_and_category = utils_en.get_products_and_category()
products_and_category
```

{'Computers and Laptops': ['TechPro Ultrabook',
'BlueWave Gaming Laptop',
'PowerLite Convertible',
'TechPro Desktop',
'BlueWave Chromebook'],
'Smartphones and Accessories': ['SmartX ProPhone',
'MobiTech PowerCase',
'SmartX MiniPhone',
'MobiTech Wireless Charger',
'SmartX EarBuds'],
'Televisions and Home Theater Systems': ['CineView 4K TV',
'SoundMax Home Theater',
'CineView 8K TV',
'SoundMax Soundbar',
'CineView OLED TV'],
'Gaming Cconsoles and accessories': ['GameSphere X',
'ProGamer Controller',
'GameSphere Y',
'ProGamer Racing Wheel',
'GameSphere VR Headset'],
'Audio Equipment': ['AudioPhonic Noise-Canceling Headphones',
'WaveSound Bluetooth Speaker',
'AudioPhonic True Wireless Earbuds',
'WaveSound Soundbar',
'AudioPhonic Turntable'],
'Cameras and Camcorders': ['FotoSnap DSLR Camera',
'ActionCam 4K',
'FotoSnap Mirrorless Camera',
'ZoomMaster Camcorder',
'FotoSnap Instant Camera']}

```python
def find_category_and_product_v1(user_input, products_and_category):
"""
Get products and categories from user input

Parameters:
user_input: user query
products_and_category: dictionary of product types and corresponding products
"""

# Delimiter
delimiter = "####"
# Defined system information, stating the work that needs to be done by GPT
system_message = f"""
You will be provided with customer service queries. \
The customer service query will be delimited with {delimiter} characters.
Output a Python list of json objects, whichre each object has the following format:
'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, \
Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
AND
'products': <a list of products that must be found in the allowed products below>

Where the categories and products must be found in the customer service query.
If a product is mentioned, it must be associated with the correct categoryy in the allowed products list below.
If no products or categories are found, output an empty list.

List out all products that are relevant to the customer service query based on how closely it relates
to the product name and product category.
Do not assume, from the name of the product, any features or attributes such as relative quality or price.

The allowed products are provided in JSON format.
The keys of each item represent the category.
The values ​​of eachitem is a list of products that are within that category.
Allowed products: {products_and_category}

"""
# Give a few examples
few_shot_user_1 = """I want the most expensive computer."""
few_shot_assistant_1 = """ 
[{'category': 'Computers and Laptops', \
'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
"""

messages = [ 
{'role':'system', 'content': system_message}, 
{'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"}, 
{'role':'assistant', 'content': few_shot_assistant_1 },
{'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"}, 
] 
return get_completion_from_messages(messages)

```

**2. Evaluate on some queries**

```python
# First query to evaluate
customer_msg_0 = f"""Which TV can I buy if I'm on a budget?"""

products_by_category_0 = find_category_and_product_v1(customer_msg_0,
products_and_category)
print(products_by_category_0)
```

[{'category': 'Televisions and Home Theater Systems', 'products': ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']}]

```python
# Second query to evaluate
customer_msg_1 = f"""I need a charger for my smartphone"""

products_by_category_1 = find_category_and_product_v1(customer_msg_1,
products_and_category)
print(products_by_category_1)
```[{'category': 'Smartphones and Accessories', 'products': ['MobiTech PowerCase', 'MobiTech Wireless Charger', 'SmartX EarBuds']}]

```python
# Third evaluation query
customer_msg_2 = f"""
What computers do you have?"""

products_by_category_2 = find_category_and_product_v1(customer_msg_2,
products_and_category)
products_by_category_2
```

" \n [{'category': 'Computers and Laptops', 'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]"

```python
# Fourth query, more complex
customer_msg_3 = f"""
tell me about the smartx pro phone and the fotosnap camera, the dslr one.
Also, what TVs do you have?"""

products_by_category_3 = find_category_and_product_v1(customer_msg_3,
products_and_category)
print(products_by_category_3)
```

[{'category': 'Smartphones and Accessories', 'products': ['SmartX ProPhone']}, {'category': 'Cameras and Camcorders', 'products': ['FotoSnap DSLR Camera']}, {'category': 'Televisions and Home Theater Systems', 'products': ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']}]

**3. A harder test case**

```python
customer_msg_4 = f"""
tell me about the CineView TV, the 8K one, Gamesphere console, the X one.
I'm on a budget, what computers do you have?"""

products_by_category_4 = find_category_and_product_v1(customer_msg_4,
products_and_category)
print(products_by_category_4)
```

[{'category': 'Televisions and Home Theater Systems', 'products': ['CineView 8K TV']}, {'category': 'Gaming Consoles and Accessories', 'products': ['GameSphere X']}, {'category': 'Computers and Laptops', 'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]

**4. Modify the command**

```python
def find_category_and_product_v2(user_input, products_and_category):
"""
Get products and categories from user input
Added: Don't output any extra text that doesn't conform to JSON format.
Added a second example (for a few-shot prompt) where the user asks for the cheapest computer.
In both few-shot examples, the response shown is just the full list of products in JSON format.

Parameters:
user_input: The user's query
products_and_category: A dictionary of product types and corresponding products
"""
delimiter = "####"
system_message = f"""
You will be provided with customer service queries. \
The customer service query will be delimited with {delimiter} characters.
Output a Python list ofJSON objects, where each object has the following format:
'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, \
Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
AND
'products': <a list of products that must be found in the allowed products below>
Do not output any additional text that is not in JSON format.
Do not write any explanatory text after outputting the requested JSON.Where the categories and products must be found in the customer service query.
If a product is mentioned, it must be associated with the correct category in the allowed products list below.
If no products or categories are found, output an empty list.

List out all products that are relevant to the customer service query based on how closely it relates
to the product name and product category.
Do not assume, from the name of the product, any features or attributes suchas relative quality or price.

The allowed products are provided in JSON format.
The keys of each item represent the category.
The values ​​of each item is a list of products that are within that category.
Allowed products: {products_and_category}

"""

few_shot_user_1 = """I want the most expensive computer. What do you recommend?"""
few_shot_assistant_1 = """ 
[{'category': 'Computers and Laptops', \
'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
"""

few_shot_user_2 = """I want the cheapest computer. What do you recommend?"""
few_shot_assistant_2 = """ 
[{'category': 'Computers and Laptops', \
'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
"""

messages = [ 
{'role':'system', 'content': system_message}, 
{'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"}, 
{'role':'assistant', 'content': few_shot_assistant_1 }, 
{'role':'user', 'content': f"{delimiter}{few_shot_user_2}{delimiter}"}, 
{'role':'assistant', 'content': few_shot_assistant_2 }, 
{'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"}, 
] 
return get_completion_from_messages(messages)

```

**5. Further evaluation**

```python
customer_msg_3 = f"""
tell me about the smartx pro phone and the fotosnap camera, the dslr one.Also, what TVs do you have?"""

products_by_category_3 = find_category_and_product_v2(customer_msg_3,
products_and_category)
print(products_by_category_3)
```

[{'category': 'Smartphones and Accessories', 'products': ['SmartX ProPhone']}, {'category': 'Cameras and Camcorders', 'products': ['FotoSnap DSLR Camera']}, {'category': 'Televisions and Home Theater Systems', 'products': ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']}]

**6. Regression test**

```python
customer_msg_0 = f"""Which TV can I buy if I'm on a budget?"""

products_by_category_0 = find_category_and_product_v2(customer_msg_0,
products_and_category)
print(products_by_category_0)
```

[{'category': 'Televisions and Home Theater Systems', 'products': ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineViewOLED TV']}]

**7. Automated testing**

```python
msg_ideal_pairs_set = [

# eg 0
{'customer_msg':"""Which TV can I buy if I'm on a budget?""",
'ideal_answer':{
'Televisions and Home Theater Systems':set(
['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']
)}
},

# eg 1
{'customer_msg':"""I need a charger for my smartphone""",
'ideal_answer':{
'Smartphones and Accessories':set(['MobiTech PowerCase', 'MobiTech Wireless Charger', 'SmartX EarBuds']
)}
},
# eg 2
{'customer_msg':f"""What computers do you have?""",
'ideal_answer':{
'Computers and Laptops':set(
['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook'
])
}
},

# eg 3
{'customer_msg':f"""tell me about the smartx pro phone and \
the fotosnap camera, the dslr one.\
Also, what TVs do you have?""",
'ideal_answer':{
'Smartphones and Accessories':set(
['SmartX ProPhone']),
'Cameras and Camcorders':set(
['FotoSnap DSLR Camera']),
'Televisions and Home Theater Systems':set(
['CineView 4K TV', 'SoundMax Home Theater','CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV'])
}
}, 

# eg 4
{'customer_msg':"""tell me about the CineView TV, the 8K one, Gamesphere console, the X one.
I'm on a budget, what computers do you have?""",
'ideal_answer':{
'Televisions and Home Theater Systems':set(
['CineView 8K TV']),
'Gaming Consoles and Accessories':set(
['GameSphere X']),
'Computers and Laptops':set(
['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook'])
}
},

# eg 5
{'customer_msg':f"""What smartphonesdo you have?""",
'ideal_answer':{
'Smartphones and Accessories':set(
['SmartX ProPhone', 'MobiTech PowerCase', 'SmartX MiniPhone', 'MobiTech Wireless Charger', 'SmartX EarBuds'
])
}
},
# eg 6
{'customer_msg':f"""I'm on a budget. Can you recommend some smartphones to me?""",
'ideal_answer':{
'Smartphones and Accessories':set(
['SmartX EarBuds', 'SmartX MiniPhone', 'MobiTech PowerCase', 'SmartXProPhone', 'MobiTech Wireless Charger']
)}
},

# eg 7 # this will output a subset of the ideal answer
{'customer_msg':f"""What Gaming consoles would be good for my friend who is into racing games?""",
'ideal_answer':{
'Gaming Consoles and Accessories':set([
'GameSphere X',
'ProGamer Controller',
'GameSphere Y',
'ProGamer Racing Wheel',
'GameSphere VR Headset'
])}
},
# eg 8
{'customer_msg':f"""What could be a good present for my videographer friend?""",
'ideal_answer': {
'Cameras and Camcorders':set([
'FotoSnap DSLR Camera', 'ActionCam 4K', 'FotoSnap Mirrorless Camera', 'ZoomMaster Camcorder', 'FotoSnap Instant Camera'
])}
},

# eg 9
{'customer_msg':f"""I would like a hot tub time machine.""",
'ideal_answer': []
}

]

```

**8. Comparison with the ideal answer**

```python
import json
def eval_response_with_ideal(response,ideal,
debug=False):
"""
Evaluate whether the response matches the ideal answer

Parameters:
response: the content of the response
ideal: the ideal answer
debug: whether to print debugging information
"""
if debug:
print("Response:")
print(response)

# json.loads() can only parse double quotes, so replace single quotes with double quotes here
json_like_str = response.replace("'",'"')

# Parse into a series of dictionaries
l_of_d = json.loads(json_like_str)

# When the response is empty, that is, no products are found
if l_of_d == [] and ideal == []:
return 1

# Another abnormal situation is that the number of standard answers does not match the number of reply answerselif l_of_d == [] or ideal == []:
return 0

# Count the number of correct answers
correct = 0

if debug:
print("l_of_d is")
print(l_of_d)

# For each question and answer pair
for d in l_of_d:

# Get products and categories
cat = d.get('category')
prod_l = d.get('products')
# Get products and categories
if cat and prod_l:
# convert list to set for comparison
prod_set = set(prod_l)
# get ideal set of products
ideal_cat = ideal.get(cat)if ideal_cat:
prod_set_ideal = set(ideal.get(cat))
else:
if debug:
print(f"No directory found in the standard answer {cat}")
print(f"Standard answer: {ideal}")
continue

if debug:
print("Product set:\n",prod_set)
print()
print("Standard answer product set:\n",prod_set_ideal)

# The product set found is consistent with the standard product set
if prod_set == prod_set_ideal:
if debug:print("correct")
correct +=1
else:
print("wrong")
print(f"product set: {prod_set}")
print(f"standard product set: {prod_set_ideal}")
if prod_set <= prod_set_ideal:
print("the answer is a subset of the standard answer")
elif prod_set >= prod_set_ideal:
print("the answer is a superset of the standard answer")

# Calculate the number of correct answers
pc_correct = correct / len(l_of_d)

return pc_correct
```

```python
print(f'User question: {msg_ideal_pairs_set[7]["customer_msg"]}')
print(f'standard answer: {msg_ideal_pairs_set[7]["ideal_answer"]}')
```

User question: What Gaming consoles would be good for my friend who is into racing games?
Standard answer: {'Gaming Consoles and Accessories': {'ProGamer Racing Wheel', 'ProGamer Controller', 'GameSphere Y', 'GameSphere VR Headset', 'GameSphere X'}}

```python
response = find_category_and_product_v2(msg_ideal_pairs_set[7]["customer_msg"],
products_and_category)
print(f'responseAnswer: {response}')

eval_response_with_ideal(response,
msg_ideal_pairs_set[7]["ideal_answer"])
```

Answer: 
[{'category': 'Gaming Consoles and Accessories', 'products': ['GameSphere X', 'ProGamer Controller', 'GameSphere Y', 'ProGamer Racing Wheel', 'GameSphere VR Headset']}]

1.0

**9. Calculate the correct proportion**

```python
import time

score_accum = 0
for i, pair in enumerate(msg_ideal_pairs_set):
time.sleep(20)
print(f"Example {i}")

customer_msg = pair['customer_msg']
ideal = pair['ideal_answer']

# print("Customer message",customer_msg)
# print("ideal:",ideal)
response = find_category_and_product_v2(customer_msg,
products_and_category)

# print("products_by_category",products_by_category)
score = eval_response_with_ideal(response,ideal,debug=False)
print(f"{i}: {score}")
score_accum += score

n_examples = len(msg_ideal_pairs_set)
fraction_correct = score_accum / n_examples
print(f"The correct fraction is {n_examples}: {fraction_correct}")
```

Example 0
0: 1.0
Example 1
Wrong
Product set: {'MobiTech Wireless Charger', 'SmartX EarBuds', 'SmartX MiniPhone', 'SmartX ProPhone', 'MobiTech PowerCase'}
Standard product set: {'MobiTech Wireless Charger', 'SmartX EarBuds', 'MobiTech PowerCase'}
The answer is a superset of the standard answer
1: 0.0
Example 2
2: 1.0
Example 3
3: 1.0
Example 4
Wrong
Product set: {'SoundMax Home Theater', 'CineView 8K TV', 'CineView 4K TV', 'CineView OLED TV', 'SoundMax Soundbar'}
Standard product set: {'CineView 8K TV'}
The answer is a superset of the standard answer
Wrong
Product set: {'ProGamer Racing Wheel', 'ProGamer Controller', 'GameSphere Y', 'GameSphere VR Headset', 'GameSphere X'}
Standard product set: {'GameSphere X'}
The answer is a superset of the standard answer
4: 0.3333333333333333
Example 5
5: 1.0
Example 6
6: 1.0
Example 7
7: 1.0
Example 8
8: 1.0
Example 9
9: 1
The correct ratio is 10: 0.83333333333333334