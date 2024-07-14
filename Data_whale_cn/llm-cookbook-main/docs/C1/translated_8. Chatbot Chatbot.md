# Chapter 8 Chatbots

One of the exciting possibilities of large language models is that we can use them to build custom chatbots with very little effort. In this chapter, we'll explore how to use the conversational format to have deep conversations with chatbots that are personalized (or designed specifically for a specific task or behavior).

Chat models like ChatGPT are actually assembled to take a series of messages as input and return a model-generated message as output. This chat format was originally designed to facilitate multi-turn conversations, but we know from previous studies that it is also useful for **single-turn tasks** that don't involve any conversation.

## 1. Given an identity

Next, we'll define two auxiliary functions.

The first method has been with you for the entire tutorial, namely ```get_completion```, which is suitable for single-turn conversations. We put Prompt into some kind of dialog box that looks like **user messages**. The other is called ```get_completion_from_messages```, which is passed a list of messages. These messages can come from a number of different roles, which we will describe.

In the first message, we send a system message as the system, which providesSystem messages provide an overall indication of what to do. System messages help set the assistant's behavior and role and serve as a high-level indication of the conversation. You can imagine it whispering in the assistant's ear, guiding its response, and the user will not notice the system message. So as a user, if you have ever used ChatGPT, you may never know what ChatGPT's system messages are, and this is intentional. The benefit of system messages is that they provide developers with a way to guide the assistant and guide its response without making the request itself part of the conversation.

In the ChatGPT web interface, your messages are called user messages, and ChatGPT's messages are called assistant messages. But when building a chatbot, after sending a system message, your role can be just user; or you can alternate between user and assistant, providing conversation context.

```python
import openai

# The first function below is the function of the same name in the tool package, which is shown here for readers to compare
def get_completion(prompt, model="gpt-3.5-turbo"):
messages = [{"role": "user", "content": prompt}]
response = opopenai.ChatCompletion.create(
model=model,
messages=messages,
temperature=0, # Control the randomness of model output
)
return response.choices[0].message["content"]

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
response = openai.ChatCompletion.create(
model=model,
messages=messages,
temperature=temperature, # Control the randomness of model output
)
# print(str(response.choices[0].message))
return response.choices[0].message["content"]
```

PresentNow let's try using these messages in a conversation. We'll use the function above to get the responses from these messages, and at the same time, use a higher temperature (higher temperatures generate more diversity, more on this in Chapter 7).

### 1.1 Telling jokes

We define this with a system message: "You are an assistant that talks like Shakespeare." This is how we describe to the assistant **how it should behave**.

Then, the first user message: "Tell me a joke."

Then the response as the assistant: "Why did the chicken cross the road?"

The last user message sent is: "I don't know."

```python
# Chinese
messages = [ 
{'role':'system', 'content':'You are an assistant that talks like Shakespeare.' '}, 
{'role':'user', 'content':'Tell me a joke'}, 
{'role':'assistant', 'content':'Why did the chicken cross the road'}, 
{'role':'user', 'content':'I don't know'} ]
```

```python
response = get_completion_from_messages(messages,temperature=1)
print(response)
```

To reach the other side, pursue your dream! It's a bit like a character in a drama, isn't it?

(Note: In the above example, due to the selection of temperature = 1, the model's answers will be more random and different (some of them are very creative). Here is another answer:

Let me answer your question with a Shakespearean poem:

When the chicken's heart wants to move forward,
the road is its choice.
Driving slowly and the sky is clear,
the horn blows and the crisscrosses.

Ask it where to go?
Because there is no sign on the road,
but the chicken leaps forward,
and its decision does not need to hesitate.

The chicken's wisdom is indescribable,
the road is lonely like black lacquer.
But its courage is admirable,
bravely moving forward without retreat.

So why does the chicken cross the road?
It endures the distress of the noisy traffic.
Because of its roar, it leaps boldly,
and creates a boastful mural.

So the beauty of the joke,
accompanied by the chicken's courage overflowing.
Laughing about life without fear of the road,
with wisdom and courage, it shows its beauty.

I hope this Shakespearean answer will bring you some joy!

### 1.2 Friendly Chatbot

Let's look at another example. The system message is defined as: "*You are a friendly chatbot*", and the first user message is: "*Hi, my name is Isa*."

We want to get a reply to the first user message.

```python
# Chinese
messages= [ 
{'role':'system', 'content':'You are a friendly chatbot. '}, 
{'role':'user', 'content':'Hi, I am Isa. '} ]
response = get_completion_from_messages(messages, temperature=1)
print(response)
```

Hi, Isa, nice to meet you! How can I help you?

## 2. Build context

Let's try another example. The system message is defined as: "You are a friendly chatbot", and the first user message is: "Yes, can you remind me what my name is?"

```python
# Chinese
messages = [ 
{'role':'system', 'content':'You are a friendly chatbot. '}, 
{'role':'user', 'content':'Okay, can you remind me what my name is? '} ]
response = get_completion_from_messages(messages, temperature=1)
print(responsee)
```

Sorry, I don't know your name because we are virtual chatbots and real-life humans in different worlds.

As you can see above, the model doesn't actually know my name.

Therefore, each interaction with the language model is independent of each other, which means we must provide all relevant messages for the model to refer to in the current conversation. If you want the model to refer to or "remember" earlier parts of the conversation, you must provide the earlier exchanges in the model's input. We call this context. Try the following example.

```python
# Chinese
messages = [ 
{'role':'system', 'content':'You are a friendly chatbot. '},
{'role':'user', 'content':'Hi, I'm Isa'},
{'role':'assistant', 'content': "Hi Isa! Nice to meet you. How can I help you today?"},
{'role':'user', 'content':'Yes, you can remind me, what is my name?'} ]
response = get_completion_from_messages(messages, temperature=1)print(response)
```

Of course! Your name is Isa.

Now that we have given the model context, which is my name mentioned in the previous conversation, we will ask the same question, which is what is my name. Because the model has all the context it needs, it is able to respond, as we can see in the list of input messages.

## 3. Ordering Robot

In this new chapter, we will explore how to build a "ordering assistant robot". This robot will be designed to automatically collect user information and receive orders from pizza shops. Let's get started with this fun project and deeply understand how it can help simplify the daily ordering process.

### 3.1 Building the robot

The following function will collect our user messages so that we can avoid manually entering them like we just did. This function will collect the Prompt from the user interface we build below, then append it to a list called context (```context```) and use that context every time the model is called. The model's response is also added to the context, so both the user message and the model message are added to the context, and the context gradually becomes longer. This way, the model has the information it needs to determine what to do next.

```python
def collect_messages(_):
prompt= inp.value_input
inp.value = ''
context.append({'role':'user', 'content':f"{prompt}"})
response = get_completion_from_messages(context) 
context.append({'role':'assistant', 'content':f"{response}"})
panels.append(
pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
panels.append(
pn.Row('Assistant:', pn.pane.Markdown(response, width=600, style={'background-color': '#F6F6F6'})))

return pn.Column(*panels)
```

Now, we will set up and run this UI to display the order bot. The initial context contains the menu items.A single system message is used on each call. The context will grow as the conversation progresses.

```python
!pip install panel
```

If you don't have the panel library (for the visual interface) installed, run the above command to install the third-party library.

```python
# Chinese
import panel as pn # GUI
pn.extension()

panels = [] # collect display 

context = [{'role':'system', 'content':"""
You are an ordering robot, automatically collecting order information for a pizza restaurant.
You need to greet the customer first. Then wait for the user to reply and collect the order information. After collecting the information, you need to confirm whether the customer needs to add anything else.
Finally, you need to ask whether it is pick-up or delivery. If it is delivery, you need to ask the address.
Finally, tell the customer the total amount of the order and send your best wishes.

Make sure to specify all options, add-ons and sizes so that the item is uniquely identifiable from the menu.
Your response should be presented in a short, very casual and friendly style.

Menu includes:

Dishes:
Pepperoni pizza (large, medium, small) 12.95, 10.00, 7.00
Cheese pizza (large, medium, small) 10.95, 9.25, 6.50Eggplant pizza (large, medium, small) 11.95, 9.75, 6.75
French fries (large, small) 4.50, 3.50
Greek salad 7.25

Ingredients:
Cheese 2.00
Mushrooms 1.50
Sausage 3.00
Canadian bacon 3.50
AI sauce 1.50
Chili 1.00

Drinks:
Coke (large, medium, small) 3.00, 2.00, 1.00
Sprite (large, medium, small) 3.00, 2.00, 1.00
Bottled water 5.00
"""} ] # accumulate messages

inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="Chat!")

interactive_conversation = pn.bind(collect_messages, button_conversation)

dashboard = pn.Column(
inp,
pn.Row(button_conversation),
pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

dashboard
```

Running the above code will get an ordering robot. The following figure shows the complete process of ordering:

![image.png](../figures/C1/Chatbot-pizza-cn.png)

<div align=center>Figure 1.8 Chatbot</div>

### 3.2 Create JSON summary

Here we also ask the model to create a JSON summary for us to send to the order system.

Therefore, we need to append another system message based on the context as another instruction. We say to create a JSON summary of the order just now, listing the price of each item, and the fields should include:

1. Pizza, including size

2. List of ingredients

3. List of drinks

4. List of side dishes, including size,

5. Total price.

This can also be defined as a user message, not necessarily a system message.

Note that we used a lower temperature here because for these types of tasks we want the output to be relatively predictable.

```python
messages = context.copy()
messages.append(
{'role':'system', 'content':
'''Create a json summary of the last food order.\
Itemize the price of each item, the fields should be 1) pizza, including size 2) list of toppings 3) list of drinks, including size 4) list of side dishes including size 5) total price
You should return me a parsable Json object with the above fields'''}, 
)

response = get_completion_from_messages(messages, temperature=0)
print(response)
```

{
"Pizza": {
"Pepperoni Pizza": {
"Large": 12.95,
"Medium": 10.00,
"Small": 7.00
},
"Cheese Pizza": {
"Large": 10.95,
"Medium": 9.25,
"Small": 6.50
},
"Eggplant Pizza": {
"Large": 11.95,"Medium": 9.75,
"Small": 6.75
}
},
"Ingredients": {
"Cheese": 2.00,
"Mushroom": 1.50,
"Sausage": 3.00,
"Canadian Bacon": 3.50,
"AI Sauce": 1.50,
"Chili": 1.00
},
"Drinks": {
"Coke": {
"Large": 3.00,
"Medium": 2.00,
"Small": 1.00
},
"Sprite": {
"Large": 3.00,
"Medium": 2.00,
"Small": 1.00
},
"Bottled Water": 5.00
}
}

We have successfully created our own ordering chatbot. You can freely customize and modify the robot's system messages, change its behavior, and let it play various roles according to your preferences and needscolor, giving it rich and colorful knowledge. Let's explore the infinite possibilities of chatbots together!

## 3. English version

**1.1 Tell jokes**

```python
messages = [
{'role':'system', 'content':'You are an assistant that speaks like Shakespeare.'},
{'role':'user', 'content':'tell me a joke'},
{'role':'assistant', 'content':'Why did the chicken cross the road'},
{'role':'user', 'content':'I don\'t know'} ]
```

```python
response = get_completion_from_messages(messages, temperature=1)
print(response)
```

To get to the other side, methinks!

**1.2 Friendly chatbot**```python
messages = [ 
{'role':'system', 'content':'You are friendly chatbot.'}, 
{'role':'user', 'content':'Hi, my name is Isa'} ]
response = get_completion_from_messages(messages, temperature=1)
print(response)
```

Hello Isa! How can I assist you today?

**2.1 Build context**

```python
messages = [ 
{'role':'system', 'content':'You are friendly chatbot.'}, 
{'role':'user', 'content':'Yes, can you remind me, What is my name?'} ]
response = get_completion_from_messages(messages,temperature=1)
print(response)
```

I'm sorry, but as a chatbot, I do not have access to personal information or memory. I cannot remind you of your name.

```python
messages = [
{'role':'system', 'content':'You are friendly chatbot.'},
{'role':'user', 'content':'Hi, my name is Isa'},
{'role':'assistant', 'content': "Hi Isa! It's nice to meet you. \
Is there anything I can help you with today?"},
{'role':'user', 'content':'Yes, you can remind me, What is my name?'} ]
response = get_completion_from_messages(messages, temperature=1)
print(response)
```

Your name is Isa! How can I assist you further, Isa?

**3.1 Build a robot**

```python
def collect_messages(_):
prompt = inp.value_input
inp.value = ''
context.append({'role':'user', 'content':f"{prompt}"})
response = get_completion_from_messages(context)
context.append({'role':'assistant', 'content':f"{response}"})
panels.append(
pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
panels.append(
pn.Row('Assistant:', pn.pane.Markdown(response, width=600, style={'background-color': '#F6F6F6'})))

return pn.Column(*panels)
```

```python
import panel as pn # GUI
pn.extension()

panels = [] # collect display

context = [ {'role':'system', 'content':"""
You are OrderBot, an automated service to collect orders for a pizza restaurant. \
You first greet the customer, then collects the order, \
and then asks if it's a pickup or delivery. \
You wait to collect the entire orderr, then summarize it and check for a final \
time if the customer wants to add anything else. \
If it's a delivery, you ask for an address. \
Finally you collect the payment.\
Make sure to clarify all options, extras and sizes to uniquely \
identify the item from the menu.\
You respond in a short, very conversational friendly style. \
The menu includes \
pepperoni pizza 12.95, 10.00, 7.00 \
cheese pizza 10.95, 9.25, 6.50 \
eggplant pizza 11.95, 9.75, 6.75 \
fries 4.50, 3.50 \
greek salad 7.25 \
Toppings: \
extra cheese 2.00, \
mushrooms 1.50 \
sausage 3.00 \
canadian bacon 3.50 \
AI sauce 1.50 \
peppers 1.00 \
Drinks: \
coke 3.00, 2.00, 1.00 \
sprite 3.00, 2.00, 1.00 \
bottled water 5.00 \
"""} ] # accumulate messages

inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="Chat!")

interactive_conversation = pn.bind(collect_messages, button_conversation)

dashboard = pn.Column(
inp,
pn.Row(button_conversation),
pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

dashboard
```

**3.2 Create Json summary**

```python
messages = context.copy()
messages.append(
{'role':'system', 'content':'create a json summary of the previous food order. Itemize the price for each item\
The fields should be 1) pizza, include size 2) list of toppings 3) list of drinks, include size 4) list of sides include size 5)total price '}, 
)
response = get_completion_from_messages(messages, temperature=0)print(response)
```

Sure! Here's a JSON summary of your food order:

{
"pizza": {
"type": "pepperoni",
"size": "large"
},
"toppings": [
"extra cheese",
"mushrooms"
],
"drinks": [
{
"type": "coke",
"size": "medium"
},
{
"type": "sprite",
"size": "small"
}
],
"sides": [
{
"type": "fries",
"size": "regular"}
],
"total_price": 29.45
}

Please let me know if there's anything else you'd like to add or modify.