# Chapter 3 Evaluating Inputs - Classification

In this chapter, we will focus on the importance of evaluating input tasks, which is related to the quality and security of the entire system.

When dealing with tasks with multiple independent sets of instructions in different situations, it has many advantages to first classify the query type and determine which instructions to use based on this. This can be achieved by defining fixed categories and hard-coding the instructions related to handling specific categories of tasks. For example, when building a customer service assistant, it may be critical to classify the query type and determine the instructions to use based on the classification. Specifically, if the user asks to close their account, then the secondary instruction may be to add additional instructions on how to close the account; if the user asks for specific product information, the secondary instruction may be to provide more product information.

```python
delimiter = "####"
```

In this example, we use the system message (system_message) as a global guide for the entire system and choose to use "#" as the delimiter. `Delimiters are tools used to distinguish different parts in an instruction or output`, which can help the model better identify the parts, thereby improving the accuracy and efficiency of the system in performing specific tasks. The “#” is also an ideal delimiter as it can be treated as a single token.

This is the system message we defined and we are asking the model in the following way.

```python
system_message = f"""
You will get customer service inquiries.
Each customer service inquiry will be separated by the {delimiter} character.
Categorize each inquiry into a primary category and a secondary category.
Provide your output in JSON format with the following keys: primary and secondary.

Primary category: Billing, Technical Support, Account Management, or General Inquiry.

Billing secondary categories:
Unsubscribe or upgrade
Add a payment method
Explanation for charge
Dispute a charge

Technical support secondary categories:
General troubleshooting
Device compatibility
Software updates

Account management secondary categories:
Password reset
Update personal informationtion
Close account
Account security

General enquiry subcategories:
Product information
Pricing
Feedback
Speak to a human

"""
```

Now that we understand system messages, let's look at an example of a user message.

```python
user_message = f"""\ 
I want you to delete my profile and all user data. """
```

First, format this user message into a message list and separate system messages and user messages with "####".

```python
messages = [ 
{'role':'system', 
'content': system_message}, 
{'role':'user', 
'content': f"{delimiter}{user_message}{delimiter}"}, 
]
```

If you were to judge, which category would the following sentence belong to: "I want you to delete my profile. Let's think about it,This sentence seems to belong to either "Account Management" or "Close Account". 

Let's see how the model thinks:

```python
from tool import get_completion_from_messages

response = get_completion_from_messages(messages)
print(response)
```

{
"primary": "Account Management",
"secondary": "Close Account"

}

The model's classification is "Account Management" as "primary" and "Close Account" as "secondary".

The benefit of requesting structured output (such as JSON) is that you can easily read it into an object, such as a dictionary in Python. If you use another language, you can also convert it to another object and input it into the next step.

Let's take a look at another example:

```
User message: "Tell me more about your tablet"

```
We use the same list of messages to get the model's response and then print it out.

```python
user_message = f"""\
Tell me more about your tablet"""
messages = [ 
{'role':'system', 
'content': system_message}, 
{'role':'user', 
'content': f"{delimiter}{user_message}{delimiter}"}, 
] 
response = get_completion_from_messages(messages)
print(response)
```

{
"primary": "General Inquiry",
"secondary": "Product Information"
}

Another categorized result is returned here, and it looks like it's correct. So, depending on the categorization of the customer inquiry, we can now provide a more specific set of instructions for next steps. In this case, we might add additional information about the tablet, while in other cases, we might want to provide a link to close the account or something similar. Another categorized result is returned here, and it looks like it should be correct.

In the next chapter, we'll explore more about methods for evaluating input, especially how to ensure that usersUse the system in a responsible manner.

## English version ```python
system_message = f"""
You will be provided with customer service queries. \
The customer service query will be delimited with \
{delimiter} characters.
Classify each query into a primary category \
and a secondary category . 
Provide your output in json format with the \
keys: primary and secondary.

Primary categories: Billing, Technical Support, \
Account Management, or General Inquiry.

Billing secondary categories:
Unsubscribe or upgrade
Add a payment method
Explanation for charge
Dispute a charge

Technical Support secondary categories:
General troubleshooting
Device compatibility
Software updates

Account Management secondary categories:
Password reset
Update personal information
Close account
Account security

General Inquiry secondary categories:
Product information
Pricing
Feedback
Speak to a human

"""
```

```python
user_message = f"""\ 
I want you to delete my profile and all of my user data"""
```

```python
messages = [ 
{'role':'system','content': system_message}, 
{'role':'user', 
'content': f"{delimiter}{user_message}{delimiter}"}, 
]
```

```python
response = get_completion_from_messages(messages)
print(response)
```

{
"primary": "Account Management",
"secondary": "Close account"
}

```python
user_message = f"""\
Tell me more about your flat screen tvs"""
messages = [ 
{'role':'system', 
'content': system_message}, 
{'role':'user', 
'content': f"{delimiter}{user_message}{delimiter}"},] 
response = get_completion_from_messages(messages)
print(response)
```

{
"primary": "General Inquiry",
"secondary": "Product information"
}