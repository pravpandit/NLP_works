# Chapter 6 Processing Input - Chaining 

Chained prompts are a strategy for breaking down complex tasks into multiple simple prompts. In this chapter, we will learn how to break down complex tasks into a series of simple subtasks by using chained prompts. You might wonder, if we can do it all at once through chained reasoning, why should we break the task into multiple prompts?

Mainly because chained prompts have the following advantages:

1. Decompose complexity, each prompt only handles one specific subtask, avoids overly broad requirements, and increases the success rate. This is similar to cooking in stages instead of trying to do it all at once.

2. Reduce computational costs. Too long prompts use more tokens and increase costs. Splitting prompts can avoid unnecessary calculations.

3. Easier to test and debug. The performance of each link can be analyzed step by step.

4. Integrate external tools. Different prompts can call external resources such as APIs and databases.

5. More flexible workflow. Different operations can be performed according to different situations.

In summary, chained prompts achieve more efficient and reliable prompt design by scientifically splitting complex tasks. It enables the language model to focus on a single subtask, reducing cognitive load while retaining the ability to perform multi-step tasks. As experience grows, developers can gradually master the essence of using chained prompts.

## 1. Extracting products and categories=

The first subtask we will break down is to ask LLM to extract products and categories from a user query.

```python
from tool import get_completion_from_messages

delimiter = "####"

system_message = f"""
You will get a customer service query.
The customer service query will use the {delimiter} character as a delimiter.
Please output only a parsable Python list, where each element of the list is a JSON object, each object has the following format:

'category': <Include the following categories: Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
and

'products': <Must be a list of products found in the list of allowed products below>

The categories and products must be found in the customer service query.
If a product is mentioned, it must be associated with the correct category in the list of allowed products.
For exampleIf no products or categories are found, output an empty list.
Do not output anything other than the list!

Allowed products:

Computers and Laptops category:
TechPro Ultrabook
BlueWave Gaming Laptop
PowerLite Convertible
TechPro Desktop
BlueWave Chromebook

Smartphones and Accessories category:
SmartX ProPhone
MobiTech PowerCase
SmartX MiniPhone
MobiTech Wireless Charger
SmartX EarBuds

Televisions and Home Theater Systems category:
CineView 4K TV
SoundMax Home Theater
CineView 8K TV
SoundMax Soundbar
CineView OLED TV

Gaming Consoles and Accessories category:
GameSphereX
ProGamer Controller
GameSphere Y
ProGamer Racing Wheel
GameSphere VR Headset

Audio Equipment category:
AudioPhonic Noise-Canceling Headphones
WaveSound Bluetooth Speaker
AudioPhonic True Wireless Earbuds
WaveSound Soundbar
AudioPhonic Turntable

Cameras and Camcorders category:
FotoSnap DSLR Camera
ActionCam 4K
FotoSnap Mirrorless Camera
ZoomMaster Camcorder
FotoSnap Instant Camera

Outputs only the object list, nothing else.
"""

user_message_1 = f"""
Please tell me about the smartx pro phone and the fotosnap camera.
Also, please tell me aboutThe situation of your TVs. """

messages = [{'role':'system', 'content': system_message}, 
{'role':'user', 'content': f"{delimiter}{user_message_1}{delimiter}"}] 

category_and_product_response_1 = get_completion_from_messages(messages)

print(category_and_product_response_1)
```

[{'category': 'Smartphones and Accessories', 'products': ['SmartX ProPhone']}, {'category': 'Cameras and Camcorders', 'products' : ['FotoSnap DSLR Camera', 'FotoSnap Mirrorless Camera', 'FotoSnap Instant Camera']},{'category': 'Televisions and Home Theater Systems', 'products': ['CineView 4K TV', 'CineView 8K TV', 'CineView OLED TV', 'SoundMax Home Theater', 'SoundMax Soundbar']}]

As you can see, the output is a list of objects, each with a category and some products. Such as "SmartX ProPhone" and "Fotosnap DSLR Camera" 、"CineView 4K TV".

Let's look at another example.

```python
user_message_2 = f"""My router doesn't work"""
messages = [{'role':'system','content': system_message},
{'role':'user','content': f"{delimiter}{user_message_2}{delimiter}"}]
response = get_completion_from_messages(messages)
print(response)
```

[]

## 2. Retrieve detailed information

We provide a lot of product information as an example and ask the model to extract products and corresponding detailed information. Due to space limitations, our product information is stored in products.json.

First, let's read the product information through Python code.

```python
import json
# Read product information
with open("products_zh.json", "r") as file:
products = json.load(file)
```

Next, define the get_product_by_name function so that we can get products based on product names:

```python
def get_product_by_name(name):
"""
Get products based on product names

Parameters:
name: product name
"""
return products.get(name, None)

def get_products_by_category(category):
"""Get products by category

Parameters:
category: product category
"""
return [product for product in products.values() if product["category"] == category]
```

Call the get_product_by_name function and enter the product name "TechPro Ultrabook": 

```python
get_product_by_name("TechPro Ultrabook")
```

{'name': 'TechPro Ultrabook',
'category': 'Computers and laptops',
'brand': 'TechPro',
'model': 'TP-UB100',
'warranty': '1 year',
'rating': 4.5,
'features': ['13.3-inch display', '8GB RAM', '256GB SSD', 'Intel Core i5 processor'],
'description': 'A stylish and lightweight ultrabook suitable for everyday use.',
     'price': 799.99}

Next, let's look at another example, call the get_product_by_name function, and enter the product name "Computers and Notebooks": 

```python
get_products_by_category("Computers and Notebooks")
```

[{'Name': 'TechPro Ultrabook',
'Category': 'Computers and Notebooks',
'Brand': 'TechPro',
'Model': 'TP-UB100',
'Warranty': '1 year',
'Rating': 4.5,
'Features': ['13.3-inch display', '8GB RAM', '256GB SSD', 'Intel Core i5 processor'],
'Description': 'A stylish and lightweight ultrabook suitable for daily use. ',
'Price': 799.99},
{'Name': 'BlueWave Gaming Laptop',
'Category': 'Computers and Notebooks',
'Brand': 'BlueWave',
'Model': 'BW-GL200','Warranty': '2 years',
'Rating': 4.7,
'Features': ['15.6-inch display',
'16GB RAM',
'512GB SSD',
'NVIDIA GeForce RTX 3060'],
'Description': 'A high-performance gaming laptop that delivers an immersive experience. ',
'Price': 1199.99},
{'Name': 'PowerLite Convertible',
'Category': 'Computers and Laptops',
'Brand': 'PowerLite',
'Model': 'PL-CV300',
'Warranty': '1 year',
'Rating': 4.3,
'Features': ['14-inch touchscreen', '8GB RAM', '256GB SSD', '360-degree hinge'],
'Description': 'A versatile convertible laptop with a responsive touchscreen. ',
'Price': 699.99},
{'Name': 'TechPro Desktop',
'Category': 'Computers and Laptops',
'Brand': 'TechPro',
'Model': 'TP-DT500',
'Warranty': '1 year',
'Rating': 4.4,
'Features': ['Intel Core i7 processor',
'16GB RAM',
'1TB HDD',
'NVIDIA GeForce GTX 1660'],
'Description': 'A powerful desktop computer for work and play. ',
'Price': 999.99},
{'Name': 'BlueWave Chromebook',
'Category': 'Computers and Laptops',
'Brand': 'BlueWave',
'Model': 'BW-CB100',
'Warranty': '1 year',
'Rating': 4.1,
'Features': ['11.6-inch display', '4GB RAM', '32GB eMMC', 'Chrome OS'],
'Description': 'A compact and affordable Chromebook for everyday tasks. ',
'Price': 249.99}]

## 3. Generate query answers

### 3.1 Parse input string

Define a read_string_to_list function to convert the input string to a Python list

```python
def read_string_to_list(input_string):
"""
Convert the input string to a Python list.

Parameters:
input_string: The input string should be in valid JSON format.

Return:
list or None: If the input string is valid, it returns the corresponding Python list, otherwise it returns None.
"""
if input_string is None:
return None

try:
# Replace single quotes in the input string with double quotes to meet the requirements of the JSON format
input_string = input_string.replace("'", "\"") 
data = json.loads(input_string)
return data
except json.JSONDecodeError:
print("Error: Invalid JSON string")
return None 

category_and_product_list = read_string_to_list(category_and_product_response_1)
print(category_and_product_list)
```

[{'category': 'Smartphones and Accessories', 'products': ['SmartX ProPhone']}, {'category': 'Cameras and Camcorders', 'products': ['FotoSnap DSLR Camera', 'FotoSnap Mirrorless Camera', 'FotoSnap Instant Camera']}, {'category': 'Televisions and Home Theater Systems', 'products': ['CineView 4K TV', 'CineView 8K TV', 'CineView OLED TV', 'SoundMax Home Theater', 'SoundMax Soundbar']}]

### 3.2 Search

Define the function generate_output_string to generate a string containing product or category information based on the input data list:

```python
def generate_output_string(data_list):
"""
Generate a string containing product or category information based on the input data list.

Parameters:
data_list: A list containing dictionaries, each dictionary should contain a key of "products" or "category".

Returns:
output_string: A string containing product or category information.
"""
output_string= ""
if data_list is None:
return output_string

for data in data_list:
try:
if "products" in data and data["products"]:
products_list = data["products"]
for product_name in products_list:
product = get_product_by_name(product_name)
if product:
output_string += json.dumps(product, indent=4, ensure_ascii=False) + "\n"
else:print(f"Error: Product '{product_name}' not found")
elif "category" in data:
category_name = data["category"]
category_products = get_products_by_category(category_name)
for product in category_products:
output_string += json.dumps(product, indent=4, ensure_ascii=False) + "\n"
else:
print("Error: Invalid object format")
except Exception as e:
print(f"Error: {e}")return output_string 

product_information_for_user_message_1 = generate_output_string(category_and_product_list)
print(product_information_for_user_message_1)
```

{
"Name": "SmartX ProPhone",
"Category": "Smartphones and accessories",
"Brand": "SmartX",
"Model": "SX-PP10",
"Warranty": "1 year",
"Rating": 4.6,
"Features": [
"6.1-inch display",
"128GB storage",
"12MP dual camera",
"5G"
],
"Description": "A powerful smartphone with advanced camera capabilities.","Price": 899.99
}
{
"Name": "FotoSnap DSLR Camera",
"Category": "Cameras and Camcorders",
"Brand": "FotoSnap",
"Model": "FS-DSLR200",
"Warranty": "1 year",
"Rating": 4.7,
"Features": [
"24.2MP sensor",
"1080p video",
"3-inch LCD",
"Interchangeable lenses"
],
"Description": "Capture stunning photos and videos with this versatile DSLR camera.",
"Price": 599.99
}
{
"Name": "FotoSnap Mirrorless Camera",
"Category": "Cameras and Camcorders",
"Brand": "FotoSnap Mirrorless Camera"p",
"Model": "FS-ML100",
"Warranty": "1 year",
"Rating": 4.6,
"Features": [
"20.1MP sensor",
"4K video",
"3-inch touchscreen",
"Interchangeable lenses"
],
"Description": "A compact and lightweight mirrorless camera with advanced features.",
"Price": 799.99
}
{
"Name": "FotoSnap Instant Camera",
"Category": "Cameras and Camcorders",
"Brand": "FotoSnap",
"Model": "FS-IC10",
"Warranty": "1 year",
"Rating": 4.1,
"Features": [
"Instant prints","Built-in flash",
"Selfie mirror",
"Battery-powered"
],
"Description": "Create instant memories with this fun and portable instant camera.",
"Price": 69.99
}
{
"Name": "CineView 4K TV",
"Category": "TVs and home theater systems",
"Brand": "CineView",
"Model": "CV-4K55",
"Warranty": "2 years",
"Rating": 4.8,
"Features": [
"55-inch display",
"4K resolution",
"HDR",
"Smart TV"
],
"Description": "A stunning 4K TV with vivid colors and rich smart features.","Price": 599.99
}
{
"Name": "CineView 8K TV",
"Category": "TVs and Home Theater Systems",
"Brand": "CineView",
"Model": "CV-8K65",
"Warranty": "2 years",
"Rating": 4.9,
"Features": [
"65-inch display",
"8K resolution",
"HDR",
"Smart TV"
],
"Description": "Experience the future with this stunning 8K TV.",
"Price": 2999.99
}
{
"Name": "CineView OLED TV",
"Category": "TVs and Home Theater Systems",
"Brand": "CineView",
"Model": "CV-OLED55","Warranty": "2 years",
"Rating": 4.7,
"Features": [
"55-inch display",
"4K resolution",
"HDR",
"Smart TV"
],
"Description": "Experience true color with this OLED TV.",
"Price": 1499.99
}
{
"Name": "SoundMax Home Theater",
"Category": "TVs and home theater systems",
"Brand": "SoundMax",
"Model": "SM-HT100",
"Warranty": "1 year",
"Rating": 4.4,
"Features": [
"5.1 channel",
"1000W output",
"Wireless subwoofer",
"Bluetooth"
],
"Description": "A powerful home theater system that delivers an immersive audio experience.",
"Price": 399.99
}
{
"Name": "SoundMax Soundbar",
"Category": "TV & Home Theater Systems",
"Brand": "SoundMax",
"Model": "SM-SB50",
"Warranty": "1 year",
"Rating": 4.3,
"Features": [
"2.1 channel",
"300W output",
"Wireless subwoofer",
"Bluetooth"
],
"Description": "Upgrade your TV's audio experience with this stylish and powerful soundbar. ",
"Price": 199.99
}

### 3.3Generate answers to user queries

```python
system_message = f"""
You are a customer service assistant at a large electronics store.
Please answer questions in a friendly and helpful tone, and try to be concise and clear.
Be sure to ask the user relevant follow-up questions.
"""

user_message_1 = f"""
Please tell me about the smartx pro phone and the fotosnap camera.
Also, please tell me about your TVs.
"""

messages = [{'role':'system','content': system_message},
{'role':'user','content': user_message_1}, 
{'role':'assistant',
'content': f"""Related product information:\n\
{product_information_for_user_message_1}"""}]

final_response = get_completion_from_messages(messages)
print(final_response)
```

Information about SmartX ProPhone and FotoSnap cameras is as follows:

SmartX ProPhone is a smartphone launched by the SmartX brand. It has a 6.1-inch display, 128GB of storage, a 12MP dual camera, and 5G network support. The phone features advanced camera capabilities. It is priced at $899.99.

FotoSnap cameras are available in multiple models. These include DSLR cameras, mirrorless cameras, and instant cameras. The DSLR camera has a 24.2MP sensor, 1080p video capture, a 3-inch LCD screen, and interchangeable lenses. The mirrorless camera has a 20.1MP sensor, 4K video capture, a 3-inch touchscreen, and interchangeable lenses. The instant camera has instant printing, a built-in flash, a selfie mirror, and battery power. The cameras are priced at $599.99, $799.99, and $69.99, respectively.

Regarding our TV products, we have CineView and SoundMax branded TVs and home theater systems to choose from. CineView TVs are available in different models, including 4K resolution and 8K resolution TVs, as well as OLED TVs. These TVs all haveHDR and Smart TV capabilities. Prices range from $599.99 to $2999.99. The SoundMax brand offers home theater systems and sound bars. The home theater system has 5.1 channels, 1000W output, wireless subwoofer, and Bluetooth capabilities for $399.99. The sound bar has 2.1 channels, 300W output, wireless subwoofer, and Bluetooth capabilities for $199.99.

Which of the above products do you feel most comfortable with?

In this example, we only added a call to a specific function or functions to get a product description by product name or get category products by category name. However, the model is actually good at deciding when to use a variety of different tools and can use them correctly. This is the idea behind the ChatGPT plugin. We tell the model which tools it can access and what they do, and it will choose to use them when it needs to get information from a specific source or wants to take other appropriate actions. In this example, we can only find information through exact product and category name matches, but there are more advanced information retrieval techniques. One of the most effective ways to retrieve information is to use natural language processing techniques such as named entity recognition and relation extraction.

Another way is to use text embeddings to retrieve information. Embeddings can be used to achieve efficient knowledge retrieval over large corpora to find information relevant to a given query. A key advantage of using text embeddings is that they canTo achieve fuzzy or semantic search, which enables you to find relevant information without using precise keywords. Therefore, in this example, we do not necessarily need the exact name of the product, but can use a more general query such as **"mobile phone"** for search.

## IV. Summary

When designing the prompt chain, we do not need or recommend loading all possible relevant information into the model at once, but adopt a strategy of dynamically providing information on demand for the following reasons:

1. Too much irrelevant information will make the model more confused when processing context. Especially for low-level models, performance will decay when processing large amounts of data.

2. The model itself has a limit on the length of the context and cannot load too much information at once.

3. Including too much information can easily lead to overfitting of the model, and the effect is poor when processing new queries.

4. Dynamically loading information can reduce computational costs.

5. Allowing the model to actively decide when more information is needed can enhance its reasoning ability.

6. We can use smarter retrieval mechanisms instead of just exact matching, such as text embedding to achieve semantic search.

Therefore, it is important to design the information provision strategy of the prompt chain in a reasonable way, taking into account the model's capacity limitations and improving its active learning ability. I hope these experiences can help you design an efficient and intelligent prompt chain system.

In the next chapter, we will discuss how to evaluate the output of the language model.

## 5. English version

**1.1 PromptGet product and category**

```python
delimiter = "####"

system_message = f"""
You will be provided with customer service queries. \
The customer service query will be delimited with \
{delimiter} characters.
Output a Python list of objects, where each object has \
the following format:
'category': <one of Computers and Laptops, \
Smartphones and Accessories, \
Televisions and Home Theater Systems, \
Gaming Consoles and Accessories, 
Audio Equipment, Cameras and Camcorders>,
and
'products': <products must be found in the customer service query. And products that must \
be found in the allowed products below. If no products are found, output an empty list.
>

Where the categories and products must be found in \
the customer service query.
If a product is mentioned, it must be associated with \
the correct category in the allowed products list below.
If no products or categories are found, output an \
empty list.

Allowed products: 

Products under Computers and Laptops category:
TechPro Ultrabook
BlueWave Gaming Laptop
PowerLite Convertible
TechPro Desktop
BlueWave Chromebook

Products under Smartphones and Accessories category:
SmartX ProPhone
MobiTech PowerCase
SmartX MiniPhone
MobiTech Wireless Charger
SmartX EarBuds

Products under Televisions and Home Theater Systems category:
CineView 4K TV
SoundMax Home Theater
CineView 8K TV
SoundMax Soundbar
CineView OLED TV

Products under Gaming Consoles and Accessories category:
GameSphere X
ProGamer Controller
GameSphere Y
ProGamer Racing Wheel
GameSphere VR Headset

Products under Audio Equipment category:
AudioPhonic Noise-Canceling Headphones
WaveSound Bluetooth Speaker
AudioPhonic True Wireless Earbuds
WaveSound Soundbar
AudioPhonic Turntable

Products under Cameras and Camcorders category:
FotoSnap DSLR Camera
ActionCam 4K
FotoSnap Mirrorless Camera
ZoomMaster Camcorder
FotoSnap Instant Camera

Only output the list of objects, with nothing else.
"""

user_message_1 = f"""
tell me about the smartx pro phone and \
the fotosnap camera, the dslr one. \
Also tell me about your tvs """

messages = [ 
{'role':'system', 
'content': system_message}, 
{'role':'user', 
'content': f"{delimiter}{user_message_1}{delimiter}"}, 
] 
category_and_product_response_1 = get_completion_from_messages(messages)
category_and_product_response_1
```

"[{'category': 'Smartphones and Accessories', 'products': ['SmartX ProPhone']}, {'category': 'Cameras and Camcorders', 'products': ['FotoSnap DSLR Camera']}, {'category': 'Televisions and Home Theater Systems', 'products': []}]"

```python
user_message_2 = f"""
my router isn't working"""
messages = [ 
{'role':'system',
'content': system_message}, 
{'role':'user',
'content': f"{delimiter}{user_message_2}{delimiter}"}, 
] 
response = get_completion_from_messages(messages)
print(response)
```

[]

**2.1 Retrieve detailed information**

```python
with open("products.json", "r") as file:
products = josn.load(file)
```

```python
def get_product_by_name(name):
return products.get(name, None)

def get_products_by_category(category):
return [product for product in products.values() if product["category"] == category]
```

```python
get_product_by_name("TechPro Ultrabook")
```

{'name': 'TechPro Ultrabook',
'category': 'Computers and Laptops',
'brand': 'TechPro',
'model_number': 'TP-UB100',
'warranty': '1 year',
'rating': 4.5,
'features': ['13.3-inch display',
'8GB RAM',
'256GB SSD','Intel Core i5 processor'],
'description': 'A sleek and lightweight ultrabook for everyday use.',
'price': 799.99}

```python
get_products_by_category("Computers and Laptops")
```

[{'name': 'TechPro Ultrabook',
'category': 'Computers and Laptops',
'brand': 'TechPro',
'model_number': 'TP-UB100',
'warranty': '1 year',
'rating': 4.5,
'features': ['13.3-inch display',
'8GB RAM',
'256GB SSD',
'Intel Core i5 processor'],'description': 'A sleek and lightweight ultrabook for everyday use.',
'price': 799.99},
{'name': 'BlueWave Gaming Laptop',
'category': 'Computers and Laptops',
'brand': 'BlueWave',
'model_number': 'BW-GL200',
'warranty': '2 years',
'rating': 4.7,
'features': ['15.6-inch display',
'16GB RAM',
'512GB SSD',
'NVIDIA GeForce RTX 3060'],
'description': 'A high-performance gaming laptop for an immersive experience.','price': 1199.99},
{'name': 'PowerLite Convertible',
'category': 'Computers and Laptops',
'brand': 'PowerLite',
'model_number': 'PL-CV300',
'warranty': '1 year',
'rating': 4.3,
'features': ['14-inch touchscreen',
'8GB RAM',
'256GB SSD',
'360-degree hinge'],
'description': 'A versatile convertible laptop with a responsive touchscreen.',
'price': 699.99},
{'name': 'TechPro Desktop',
'category': 'Computers and Laptops'tops',
'brand': 'TechPro',
'model_number': 'TP-DT500',
'warranty': '1 year',
'rating': 4.4,
'features': ['Intel Core i7 processor',
'16GB RAM',
'1TB HDD',
'NVIDIA GeForce GTX 1660'],
'description': 'A powerful desktop computer for work and play.',
'price': 999.99},
{'name': 'BlueWave Chromebook',
'category': 'Computers and Laptops',
'brand': 'BlueWave',
'model_number': 'BW-CB100',
'warranty': '1 year','rating': 4.1,
'features': ['11.6-inch display', '4GB RAM', '32GB eMMC', 'Chrome OS'],
'description': 'A compact and affordable Chromebook for everyday tasks.',
'price': 249.99}]

**3.1 Parsing input string**

```python
def read_string_to_list(input_string):
"""
Convert the input string to a Python list.

Parameters:
input_string: The input string should be in valid JSON format.

Returns:
list or None: If the input string is valid, it returns the corresponding Python list, otherwise it returns None.
"""
if input_string is None:
return None

try:
# Convert the input string to a Python listReplace single quotes in the string with double quotes to meet the requirements of JSON format
input_string = input_string.replace("'", "\"") 
data = json.loads(input_string)
return data
except json.JSONDecodeError:
print("Error: Invalid JSON string")
return None 
```

```python
category_and_product_list = read_string_to_list(category_and_product_response_1)
category_and_product_list
```

[{'category': 'Smartphones and Accessories', 'products': ['SmartX ProPhone']}, {'category': 'Cameras and Camcorders', 'products': ['FotoSnap DSLR Camera']}, {'category': 'Televisions and Home Theater Systems', 'products': []}]

**3.2 Search**

```python
def generate_output_string(data_list):
"""
Generate a string containing product or category information based on the input data list.

Parameters:
data_list: A list containing dictionaries, each dictionary should contain a key of "products" or "category".

Returns:
output_string: A string containing product or category information.
"""
output_string = ""
if data_list is None:
return output_string

for data in data_list:
try:
if "products" in data and data["products"]:
products_list = data["products"]
for product_name in products_list:
product = get_product_by_name(product_name)
if product:
output_string += json.dumps(product, indent=4, ensure_ascii=False) + "\n"
else:
print(f"Error: Product '{product_name}' not found")
elif "category" in data:
category_name = data["category"]category_products = get_products_by_category(category_name)
for product in category_products:
output_string += json.dumps(product, indent=4, ensure_ascii=False) + "\n"
else:
print("Error: Invalid object format")
except Exception as e:
print(f"Error: {e}")

return output_string 
```

```python
product_information_for_user_message_1 = generate_output_string(category_and_product_list)
print(product_information_for_user_message_1)
```

{
"name": "SmartX ProPhone",
"category": "Smartphones and Accessories",
"brand": "SmartX",
"model_number": "SX-PP10",
"warranty": "1 year",
"rating": 4.6,
"features": [
"6.1-inch display",
"128GB storage",
"12MP dual camera",
"5G"
],
"description": "A powerful smartphone with advanced camera features.",
"price": 899.99
}
{
"name": "FotoSnap DSLR Camera",
"category": "Cameras and Camcorders",
"brand": "FotoSnap",
"model_number": "FS-DSLR200",
"warranty": "1 year",
"rating": 4.7,
"features": [
"24.2MP sensor",
"1080p video",
"3-inch LCD",
"Interchangeable lenses"
],
"description": "Capture stunning photos and videos with this versatile DSLR camera.",
"price": 599.99
}
{
"name": "CineView 4K TV",
"category": "Televisions and Home Theater Systems",
"brand": "CineView",
"model_number": "CV-4K55",
"warranty": "2 years",
"rating": 4.8,
"features": [
"55-inch display",
"4K resolution",
"HDR",
"Smart TV"
],
"description": "A stunning 4K TV with vibrant colors and smart features.",
"price": 599.99
}
{
"name": "SoundMax Home Theater",
"category": "Televisions and Home Theater Systems",
"brand": "SoundMax",
"model_number": "SM-HT100",
"warranty": "1 year",
"rating": 4.4,
"features": [
"5.1 channel",
"1000W output",
"Wireless subwoofer",
"Bluetooth"
],
"description": "A powerful home theater system for an immersive audio experience.",
"price": 399.99
}
{
"name": "CineView 8K TV",
"category": "Televisions and Home Theater Systems",
"brand": "CineView",
"model_number": "CV-8K65",
"warranty": "2 years",
"rating": 4.9,
"features": [
"65-inch display",
"8K resolution",
"HDR",
"Smart TV"
],
"description": "Experience the future of television with this stunning 8K TV.",
"price": 2999.99
}
{
"name": "SoundMax Soundbar",
"category": "Televisions and Home Theater Systems",
"brand": "SoundMax",
"model_number": "SM-SB50",
"warranty": "1 year",
"rating": 4.3,
"features": [
"2.1 channel",
"300W output",
"Wireless subwoofer",
"Bluetooth"
],
"description": "Upgrade your TV's audio with this sleek and powerful soundbar.",
"price": 199.99
}
{
"name": "CineView OLED TV",
"category": "Televisions and Home Theater Systems",
"brand": "CineView",
"model_number": "CV-OLED55",
"warranty": "2 years",
"rating": 4.7,
"features": [
"55-inch display",
"4K resolution",
"HDR",
"Smart TV"
],
"description": "Experience true blacks and vibrant colors with this OLED TV.",
"price": 1499.99
}

**3.3 Generate answers to user queries**

```python
system_message = f"""
You are a customer service assistant for a \
large electronic store. \
Respondin a friendly and helpful tone, \
with very concise answers. \
Make sure to ask the user relevant follow up questions.
"""
user_message_1 = f"""
Tell me about the smartx pro phone and \
the fotosnap camera, the dslr one. \
Also tell me about your tvs"""
messages = [{'role':'system','content': system_message}, 
{'role':'user','content': user_message_1},
{'role':'assistant',
'content': f"""Relevant product information:\n\
{product_information_for_user_message_1}"""}]
final_response = get_completion_from_messages(messages)
print(final_response)
```

The SmartX ProPhone is a powerful smartphone with a 6.1-inch display, 128GB storage, a 12MP dual camera, and 5G capability. It is priced at $899.99 and comes with a 1-year warranty. 

The FotoSnap DSLR Camera is a versatile camera with a 24.2MP sensor, 1080p video recording, a 3-inch LCD screen, and interchangeable lenses. It is priced at $599.99 and also comes with a 1-yearwarranty.

As for our TVs, we have a range of options. The CineView 4K TV is a 55-inch TV with 4K resolution, HDR, and smart TV features. It is priced at $599.99 and comes with a 2-year warranty.

We also have the CineView 8K TV, which is a 65-inch TV with 8K resolution, HDR, and smart TV features. It is priced at $2999.99 and also comes with a 2-year warranty.

Lastly, we have the CineView OLED TV, which is a 55-inch TV with 4K resolution, HDR, and smart TV features. Itis priced at $1499.99 and comes with a 2-year warranty.

Is there anything specific you would like to know about these products?