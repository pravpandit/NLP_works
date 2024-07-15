# Chapter 2 Prompt Principles

How to use prompts to give full play to the performance of LLM? First of all, we need to know the principles of prompt design, which are the basic concepts that every developer must know when designing prompts. This chapter discusses two key principles for designing efficient prompts: **Write clear and specific instructions** and **Give the model enough time to think**. Mastering these two points is particularly important for creating reliable language model interactions.

First, prompts need to express requirements clearly and clearly, and provide sufficient context so that the language model can accurately understand our intentions, just like explaining the human world in detail to an alien. Overly brief prompts often make it difficult for the model to grasp the specific tasks to be completed.

Secondly, it is also extremely important to give the language model enough time to reason. Just like humans solving problems, hasty conclusions are often wrong. Therefore, prompts should add the requirement of step-by-step reasoning to give the model enough time to think, so that the generated results are more accurate and reliable.

If prompts are optimized in both points, the language model can maximize its potential and complete complex reasoning and generation tasks. Mastering these prompt design principles is an important step for developers to achieve success in language model applications.

## 1. Principle 1 Write clear and specific instructions
Dear readers, when interacting with language models, you need to keep one thing in mind: **Clear and specific**Express your needs in a clear and detailed way. Suppose you have a new friend from an alien planet sitting in front of you, who knows nothing about human language and common sense. In this case, you need to make your intentions very clear and leave no ambiguity. Similarly, when providing a prompt, you should also make your needs and context clear in a sufficiently detailed and easy-to-understand way. 

It doesn't mean that prompts must be very short and concise. In fact, in many cases, longer and more complex prompts will make it easier for the language model to grasp the key points and give expected responses. The reason is that complex prompts provide richer context and details, allowing the model to more accurately grasp the required operations and responses.

So, remember to express prompts in clear and detailed language, just like explaining the human world to aliens, *"Adding more context helps the model understand you better."*.

Based on this principle, we provide several tips for designing prompts.

### 1.1 Use separators to clearly indicate different parts of input

When writing prompts, we can use various punctuation marks as "separators" to distinguish different parts of text.

Separators are like walls in prompts, separating different instructions, contexts, and inputs., to avoid accidental confusion. You can choose to use ` ```, """, < >, <tag> </tag>, : `, etc. as delimiters, as long as they can clearly serve as separators.

Using delimiters is particularly important to prevent **prompt injection**. What is prompt injection? It means that the text entered by the user may contain content that conflicts with your preset prompt. If it is not separated, these inputs may be "injected" and manipulate the language model, causing the model to produce irrelevant and messy output.

In the following example, we give a paragraph and ask GPT to summarize it. In this example, we use ``` as a delimiter.

```python
from tool import get_completion

text = f"""
You should provide as clear and specific instructions as possible to express the task you want the model to perform. \
This will guide the model towards the desired output and reduce the possibility of receiving irrelevant or incorrect responses. \
Don't confuse writing clear prompts with writing short prompts. \
In many cases, longer prompts can provide more clarity and context to the model, leading to more detailed and relevant output.
"""
# Text to be summarized
prompt = f"""
Summarize the text enclosed in three backticks into one sentence.
```{text}```
"""
# Instruction content, use ``` to separate the instruction and the content to be summarized
response = get_completion(prompt)
print(response)
```

To get the desired output, you should provide clear and specific instructions, avoid confusion with short prompt words, and use longer prompt words to provide more clarity and context information.

### 1.2 Seek structured output

Sometimes we need language models to give us some **structured output**, not just continuous text.

What is structured output? It is content organized in a certain format, such as JSON, HTML, etc. This kind of output is very suitable for further parsing and processing in the code. For example, you can read it into a dictionary or list in Python.

In the following example, we ask GPT to generate the titles, authors, and categories of three books, and ask GPT to return them to us in JSON format. For easy parsing, we specify the key of Json.

```python
prompt = f"""
Please generate a list of three fictional, non-existent Chinese books with titles, authors, and categories, \
and provide them in JSON format with the following keys: book_id, title, author, genre.
"""response = get_completion(prompt)
print(response)

```

{
"books": [
{
"book_id": 1,
"title": "Lost Time",
"author": "Zhang San",
"genre": "Science Fiction"
},
{
"book_id": 2,
"title": "The Door to Fantasy",
"author": "Li Si",
"genre": "Fantasy"
},
{
"book_id": 3,
"title": "Virtual Reality",
"author": "Wang Wu",
"genre": "Science Fiction"
}
]
}

### 1.3 Require the model to check whether the conditions are met

If the task contains assumptions that may not be met (conditions), we can tell the model to check these assumptions first, and if they are not met, it will point out and stop executing the subsequent complete process. You can also consider possible edge cases and the model's response to avoid unexpected results or errors.

In the following example, we will give the model two texts, one is the steps to make tea, and the other is a text without clear steps. We will ask the model to determine whether it contains a series of instructions. If so, rewrite the instructions in a given format, and if not, answer "no steps provided".

```python
# Input that satisfies the condition (steps are provided in text)
text_1 = f"""
It is easy to make a cup of tea. First, you need to boil the water.\
While waiting, take a cup and put the tea bag in it.\
Once the water is hot enough, pour it over the tea bag.\
Wait for a while and let the tea steep. After a few minutes, remove the tea bag.\
If you like, you can add some sugar or milk to taste.\
That's it, you can enjoy a delicious cup of tea.
"""
prompt = f"""
You will get the text enclosed by three quotes.\
If it contains a series of instructions, you need to rewrite them in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a series of instructions, just write "Steps not provided". "
\"\"\"{text_1}\"\"\"
"""
response = get_completion(prompt)
print("Summary of Text 1:")
print(response)
```

Summary of Text 1:
Step 1 - Boil the water.
Step 2 - Take a cup and put the tea bag in it.
Step 3 - Pour the boiling water over the tea bag.
Step 4 - Wait for a few minutes for the tea leaves to steep.
Step 5 - Remove the tea bag.
Step 6 - Add sugar or milk to taste if desired.
Step 7 - That's it, you can enjoy a delicious cup of tea.

In the above example, the model can recognize a series of instructions and output them well. In the next example, we will provide the model with inputs that are not expected instructions, and the model will judge that no steps are provided.

```python
# Input that does not meet the condition (expected instruction not provided in text)
text_2 = f"""
Today the sun is shining and the birds are singing.\
It is a beautiful day to go for a walk in the park.\
The flowers are blooming and the branches are swaying gently in the breeze.\
People are out enjoying the beautiful weather, some are having a picnic, some are playing games or relaxing on the grass.\
It is a perfect day to spend time outdoors and enjoy the beauty of nature.
"""
prompt = f"""
You will getThe text must be enclosed in three quotation marks. \
If it contains a series of instructions, you need to rewrite them in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a series of instructions, just write "No steps provided". "
\"\"\"{text_2}\"\"\"
"""
response = get_completion(prompt)
print("Summary of Text 2:")
print(response)
```

Summary of Text 2:
No steps provided.

### 1.4 Provide a few examples

"Few-shot" prompting means giving the model one or two completed examples before asking it to perform the actual task, so that the model can understand our requirements and expected output style.

For example, in the following example, we first gave a grandparent-grandchild dialogue example, and then asked the model to answer questions about "resilience" in the same metaphorical style. This is a few-shot example, which can help the model quickly grasp the tone and style we want.

With a few-shot examples, we can easily "warm up" the language model and prepare it for new tasks. This is an effective strategy to quickly get the model started on new tasks.

```python
prompt = f"""
Your task is to answer in a consistent styleQuestion.

<Child>: Teach me about patience.

<Grandparent>: The river that carves the deepest canyon begins with a humble spring; the grandest symphony begins with a single note; the most intricate tapestry begins with a single thread.

<Child>: Teach me about resilience.
"""
response = get_completion(prompt)
print(response)
```

<Grandparents>: Resilience is a quality of perseverance, like a tenacious tree standing tall in the wind and rain. It is the indomitable spirit in the face of difficulties and challenges, the ability to adapt to changes and overcome adversity. Resilience is an inner strength that allows us to persist in pursuing our goals and work hard even in the face of difficulties and setbacks.

## 2. Principle 2 Give the model time to think

When designing prompts, it is very important to give the language model enough time to reason. Like humans, language models need time to think and solve complex problems. If the language model is asked to rush to a conclusion, the result is likely to be inaccurate. For example, if you want the language model to infer the theme of a book, it is not enough to provide a simple title and a synopsis. This is like asking a person to solve a difficult math problem in a very short time, and mistakes are inevitable.

Instead, we should use prompts to guide the language model to think deeply. You can ask it to first list various views on the problem, explain the basis for reasoning, and then come up with the final answer.Conclusion. Adding the requirement of step-by-step reasoning in Prompt allows the language model to spend more time on logical thinking, and the output results will be more reliable and accurate.

In summary, giving the language model sufficient reasoning time is a very important design principle in Prompt Engineering. This will greatly improve the effect of the language model in dealing with complex problems, and it is also the key to building high-quality Prompt. Developers should pay attention to leaving room for thinking for the model to maximize the potential of the language model.

### 2.1 Specify the steps required to complete the task

Next, we will demonstrate the effect of this strategy by giving a complex task and a series of steps to complete the task.

First, we describe the story of Jack and Jill and give the prompt words to perform the following operations: First, summarize the text delimited by three backticks in one sentence. Second, translate the summary into English. Third, list each name in the English summary. Fourth, output a JSON object containing the following keys: English summary and number of names. The output is required to be separated by newlines.

```python
text = f"""
In a charming village, brother and sister Jack and Jill set out to fetch water from a mountaintop well.\
They climbed up while singing joyful songs,\
But misfortune struck - Jack tripped over a rock and rolled down the mountain, followed by Jill.\
Although they were slightly injured, they returned to their warm home.\
AlthoughDespite this surprise, their adventurous spirit remains undiminished and they continue to explore with joy.
"""
# example 1
prompt_1 = f"""
Perform the following operations:
1-Summarize the text enclosed in three backticks below in one sentence.
2-Translate the summary into English.
3-List each person's name in the English summary.
4-Output a JSON object with the following keys: english_summary, num_names.

Separate your answers with line breaks.

Text:
```{text}```
"""
response = get_completion(prompt_1)
print("prompt 1:")
print(response)
```

prompt 1:
1-Two siblings had an accident while fetching water on the hill, but finally returned home safely.
2-In a charming village, siblings Jack and Jill set off to fetch water from a well on top of a hill. While singing joyfully, they climbed up, but unfortunately, Jack tripped on a stone and rolled down the hill, with Jill following closely behind. Despite some minor injuries, they made it back to their cozy home. Despite the mishap, their adventurous spirit remained undiminished as they continued to explore with delight.
3-Jack, Jill
4-{"english_summary": "In a charming village, siblings Jack and Jill set off to fetch water from a well on top of a hill. While singing joyfully, they climbed up, but unfortunately, Jack tripped on a stone and rolled down the hill, with Jill following closely behind.own the hill, with Jill following closely behind. Despite some minor injuries, they made it back to their cozy home. Despite the mishap, their adventurous spirit remained undiminished as they continued to explore with delight.", "num_names": 2}

The above output still has some problems, for example, the key "name" will be replaced with French (Translator's note: In the original English version, it is required to translate from English to French, and the corresponding output of the third step of the instruction is 'Noms:', which is the French for Name. This behavior is difficult to predict and may cause difficulties for export)

Therefore, we will improve the prompt, which does not change the first half of the prompt, and **exactly specifies the format of the output**.

```python
prompt_2 = f"""
1-Summarize the text enclosed in <> below in a sentence.
2-Translate the summary into English.
3-List each name in the English summary.
4-Output a JSON object with the following keys:English_summary, num_names.

Please use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <translation of summary>
Names: <list of names in English summary>
Output JSON: <JSON with English_summary and num_names>

Text: <{text}>
"""
response = get_completion(prompt_2)
print("\nprompt 2:")
print(response)
```

prompt 2:
Summary: In a charming village, siblings Jack and Jill set off to fetch water from a well on top of a hill. Unfortunately, Jack tripped on a rock and tumbled down the hill, with Jill following closely behind. Despite some minor injuries, they made it back home safely. Despite the mishap, their adventurous spirit remained strong as they continued to explore joyfully.

Names: Jack, Jill

JSON Output: {"English_summary": "In a charming village, siblings Jack and Jill set off to fetch water from a well on top of a hill. Unfortunately, Jack tripped on a rock and tumbled down the hill, with Jill following closely behind. Despite some minor injuries, they made it back home safely. Despite the mishap, their adventurous spirit remained strong as they continued to explore joyfully.", "num_names": 2}

### 2.2 Guide the model to find a solution before drawing a conclusion

When designing prompts, we can also get better results by explicitly guiding the language model to think independently.

For example, suppose we want the language model to judge whether the answer to a math problem is correct. Simply providing the question and the answer is not enough, and the language model may make a hasty and wrong judgment.

Instead, we can ask the language model to try to solve the problem by itself in the prompt, think of its own solution, and then compare it with the provided solution to judge the correctness. This way of letting the language model think independently first can help it understand the problem more deeply and make more accurate judgments.

Next, we will give a question and a student's answer, and ask the model to judge whether the answer is correct:

```python
prompt = f"""
Judge whether the student's solution is correct.

Question:
I am building a solar power station and need help calculating the finances.The land cost is $100/sq. ft.
I can purchase solar panels for $250/sq. ft.
I have negotiated a maintenance contract for a fixed $100,000 per year and an additional $10/sq. ft.
What is the total cost of operation in the first year as a function of square feet?

Student Solution:
Let x be the size of the power plant in square feet.
Costs:

Land cost: 100x
Solar panel cost: 250x
Maintenance cost: $100,000+100x
Total cost: 100x+250x+$100,000+100x=450x+$100,000
"""
response = get_completion(prompt)
print(response)
```

The student's solution is correct. He correctly calculated the land cost, solar panel cost, and maintenance cost, and added them together to get the total cost.

But note that the student's solution is actually wrong. (*Maintenance cost item 100x should be 10x, and total cost 450x should be 360x*)

We can solve this problem by instructing the model to find a solution on its own first.

In the next prompt, we ask the model to solve this problem on its own first, and then use its ownThe solution is compared with the student's solution to determine whether the student's solution is correct. At the same time, we give the output format requirements. By splitting the task and clarifying the steps, giving the model more time to think, sometimes more accurate results can be obtained. In this example, the student's answer is wrong, but if we don't let the model calculate it first, it may be misled into thinking that the student is correct.

```python
prompt = f"""
To determine whether the student's solution is correct, please solve this problem by following these steps:

Steps:

First, solve the problem yourself.

Then compare your solution to the student's solution, compare the total cost you calculated with the total cost the student calculated, and evaluate whether the student's solution is correct.

Do not decide whether the student's solution is correct before you complete the problem yourself.

Use the following format:

Question: Question text
Student's solution: Student's solution text
Actual solution and steps: Actual solution and steps text
Total cost calculated by the student: Total cost calculated by the student
Actual calculated total cost: Actual calculated total cost
Are the student's calculated cost and the actual calculated cost the same: Yes or No
Are the student's solution and the actual solution the same: Yes or No
Student's score: Correct or Incorrect

Question:I am building a solar power plant and need help calculating the finances. 
- The land costs $100 per square foot
- I can purchase solar panels for $250 per square foot
- I have negotiated a maintenance contract that requires a fixed annual payment of $100,000, plus an additional $10 per square foot;

What is the total cost of operation in the first year as a function of square feet?

Student Solution:

Let x be the size of the plant in square feet.
Costs:
1. Land cost: $100x
2. Solar panels cost: $250x
3. Maintenance cost: $100,000+100x=$100,000+$10x
Total cost: $100x+$250x+$100,000+$100x=$450x+$100,000

Actual solution and steps:
"""
response = get_completion(prompt)
print(response)
```

Actual solution and steps:

1. Land cost: $100 per square foot, so total cost is $100x.
2. Solar panels cost: $250 per square foot, so total cost is 250x$.
3. Maintenance costs: Fixed costs are $100,000, and additional costs are $10 per square foot, so the total cost is $100,000 + $10x.
4. Total costs: Add the above three costs together to get a total cost of $100x + $250x + $100,000 + $10x = $360x + $100,000.

Total costs calculated by the student: 450x + $100,000
Total costs actually calculated: 360x + $100,000
Are the costs calculated by the student the same as the actual costs calculated: No
Are the student's solution the same as the actual solution: No
Student's score: Incorrect

## III. Limitations

**When developing applications related to large models, please keep in mind:**

**False knowledge**: Models occasionally generate some knowledge that looks real but is actually fabricated

When developing and applying language models, you need to be aware of the risk that they may generate false information. Although the model has been pre-trained on a large scale and has mastered a wealth of knowledge, it does not actually *completely* remember the information it sees, and it is difficult to accurately judge its own knowledge boundaries, and may make incorrect inferences. If a language model is asked to describe a non-existent product, it may construct specious details on its own. This is called "Hallucination" and is a major flaw of language models.

The following example shows the hallucination of a large model. We wantPlease tell us about the *GT Watch* product produced by Huawei. In fact, this company is real, but the product is made up, and the model provides its made-up knowledge seriously and misleadingly.

```python
prompt = f"""
Tell me about the GT Watch sports watch produced by Huawei
"""
response = get_completion(prompt)
print(response)
```

The GT Watch sports watch produced by Huawei is a smartwatch with multiple functions and features. Here is the relevant information:

1. Design and appearance: GT Watch adopts a round dial design, with a sophisticated appearance and high-quality materials. It has a variety of colors and straps to choose from, which can be customized according to personal preferences.

2. Display: GT Watch is equipped with a 1.39-inch AMOLED display with high-definition resolution and good visibility. Users can operate and navigate by touching the screen.

3. Sports tracking: GT Watch has an all-weather sports tracking function that can monitor the user's steps, running distance, calorie consumption, heart rate and other data. It also supports multiple sports modes, such as running, cycling, swimming, etc.4. Health monitoring: GT Watch can monitor health indicators such as heart rate, blood oxygen saturation and sleep quality. It also provides health advice and reminders to help users maintain good health.

5. Notification and connection: GT Watch can connect to the user's mobile phone to push notifications and call reminders through Bluetooth technology. Users can view text messages, calls and social media notifications on the watch without taking out their mobile phones.

6. Long battery life: GT Watch has a long battery life and can be used for several days on a single charge. It also supports fast charging technology and can be fully charged in a short time.

7. Other functions: GT Watch also has other functions such as weather forecast, alarm clock, timer, pedometer, etc. It also supports convenient functions such as NFC payment and music control.

Overall, Huawei GT Watch is a powerful and exquisite smart sports watch suitable for users who pay attention to health and sports.

The "hallucination" problem of language models generating false information is a risk that needs to be paid close attention to when using and developing language models. Since hallucination information is often difficult to distinguish between true and false, developers must be vigilant and try to avoid its generation.

Currently, companies such as OpenAI are actively researching solutions to the problem of language model hallucinations. Before the technology can be further improved, developers can reduce the number ofThe possibility of hallucination. For example, you can let the language model directly quote the original sentence in the text before answering. This can track the source of information and reduce the risk of false content.

In summary, the hallucination problem of language models is related to the reliability and security of applications. Developers need to be aware of this defect (note: as of July 2023) and take measures such as prompt optimization to alleviate it in order to develop more reliable language model applications. This will also be one of the important directions for the evolution of language models in the future.

**Note**:

Note on the use of backslashes: In this tutorial, we use backslashes \ to make the text fit the screen size to improve the reading experience, instead of newline characters \n. GPT-3 is not affected by newline characters, but when you call other large models, you need to consider whether newline characters will affect model performance.

## 4. English original Prompt

**1.1 Use separators to clearly indicate different parts of the input**

```python
text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possiblybly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""
response = get_completion(prompt)
print(response)
```

To guide a model towards the desired output and reduce irrelevant or incorrect responses, it is important to provide clear and specific instructions, which can be achieved through longer prompts that offer more clarity and context.

**1.2 Seeking structured output**

```python
prompt = f"""
Generate a list of three made-up book titles along \ 
with their authors and genres. 
Provide them in JSON format with the following keys: 
book_id, title, author, genre.
"""
response = get_completion(prompt)
print(response)

```

{
"books": [
{
"book_id": 1,
"title": "The Enigma of Elysium",
"author": "Evelyn Sinclair",
"genre": "Mystery"
},
{
"book_id": 2,
"title": "Whispers in the Wind",
"author": "Nathaniel Blackwood",
"genre": "Fantasy"
},
{
"book_id": 3,
"title": "Echoes of the Past",
"author": "Amelia Hart",
"genre": "Romance"
}
]
}

**1.3 Require the model to check whether the conditions are met**

```python
text_1 = f"""
Making a cup of tea is easy! First, you need to get some \ 
water boiling. While that's happening, \ 
grab a cup and put a tea bag in it. Once the water is \ 
hot enough, just pour it over the tea bag. \ 
Let it sit for a bit so the tea can steep. After a \ 
few minutes, take out the tea bag. If you \ 
like, you can add some sugar or milk to taste. \ 
And that's it! You've got yourself a delicious \ 
cup oftea to enjoy.
"""
prompt = f"""
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_1}\"\"\"
"""
response = get_completion(prompt)
print("Completion for Text 1:")
print(response)
```

Completion for Text 1:
Step 1 - Get some water boiling.
Step 2 - Grab a cup and put a tea bag in it.
Step 3 - Once the water is hot enough, pour it over the tea bag.
Step 4 - Let it sit for a bit so the tea can steep.
Step 5 - After a few minutes, take out the tea bag.
Step 6 - If you like, add some sugar or milk to taste.
Step 7 - Enjoy your delicious cup of tea.

```python
text_2 = f"""
The sun is shining brightly today, and the birds are \
singing. It's a beautiful day to go for a \ 
walk in the park. The flowersare blooming, and the \ 
trees are swaying gently in the breeze. People \ 
are out and about, enjoying the lovely weather. \ 
Some are having picnics, while others are playing \ 
games or simply relaxing on the grass. It's a \ 
perfect day to spend time outdoors and appreciate the \ 
beauty of nature.
"""
prompt = f"""You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:
Step 1 - ...
Step 2 -…
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_2}\"\"\"
"""
response = get_completion(prompt)
print("Completion for Text 2:")
print(response)
```

Completion for Text 2:
No steps provided.

**1.4 Provide a small number of examples** (Few-shot prompting)

```python
prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \ 
valley flows from a modest spring; the \ 
grandest symphony originates from a single note; \ 
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
response = get_completion(prompt)
print(response)
```

<grandparent>: Resilience is like a mighty oak tree that withstands the strongest storms, bending but never breaking. It is the unwavering determination to rise again after every fall, and the ability to find strength in the face of adversity. Just as a diamondis formed under immense pressure, resilience is forged through challenges and hardships, making us stronger and more resilient in the process.

**2.1 Specify the steps required to complete the task**

```python
text = f"""
In a charming village, siblings Jack and Jill set out on \ 
a quest to fetch water from a hilltop \ 
well. As they climbed, singing joyfully, misfortune \ 
struck—Jack tripped on a stone and tumbled \ 
down the hill, with Jill following suit. \ 
Though slightly battered, the pair returned home to \ 
comfortingembraces. Despite the mishap, \ 
their adventurous spirits remained undimmed, and they \ 
continued exploring with delight.
"""
# example 1
prompt_1 = f"""
Perform the following actions: 
1 - Summarize the following text delimited by triple \
backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following \
keys: french_summary, num_names.

Separate your answers with line breaks.

Text:
```{text}```"""
response = get_completion(prompt_1)
print("Completion for prompt 1:")
print(response)
```

Completion for prompt 1:
1 - Jack and Jill, siblings, go on a quest to fetch water from a hilltop well, but encounter misfortune when Jack trips on a stone and tumbles down the hill, with Jill following suit, yet they return home and remain undeterred in their adventurous spirits.

2 - Jack and Jill, frère and sœur, parted in quête d'eau d'un puits au sommet d'une colline, mais rencontrJack and Jill, who are both very good at making money, are also very good at making money.

3 - Jack, Jill

4 - {
"french_summary": "Jack and Jill, who are very good at making money, are also very good at making money. Jack and Jill, who are both very good at making money, are also very good at making money. Jack and Jill, who are both very good at making money, are also very good at making money.venture.",
"num_names": 2
}

```python
prompt_2 = f"""
Your task is to perform the following actions: 
1 - Summarize the following text delimited by <> with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the 
following keys: french_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in French summary>
Output JSON: <json with summary and num_names>

Text: <{text}>
"""
response = get_completion(prompt_2)
print("\nCompletion for prompt 2:")
print(response)
```

Completion for prompt 2:
Summary: Jack and Jill, siblings from a charming village, go on a quest to fetch water from a hilltop well, but encounter misfortune when Jack trips on a stone and tumbles down the hill, with Jill following suit, yet they remain undeterred and continue exploring with delight.

Translation: Jack et Jill, fresh and charming village, part of the house in the town of Colline, is a perfect match for all the poor Jack travelled to a pier and discovered Colline, so with Jill, you can rest assured that you will be able to explore more easily.

Names: Jack, Jill

Output JSON: 
{
"french_summary": "Jack and Jill, fresh and charming village, part of the house in the town of Colline, is a perfect match for all the poore Jack trébuche sur une pierre et évale la colline, suvi for Jill, pourtant ils restent déterminés et continue à explorer au joie.",
"num_names": 2
}

**2.2 Guide the model to find a solution before drawing a conclusion**

```python
prompt = f"""
Determine if the student's solution is correct or not.

Question:
I'm building a solar power installation and I need \
help working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \ 
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations 
as a function of the number of square feet.

Student's Solution:
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
"""
response = get_completion(prompt)
print(response)
```

The student's solution is correct.They correctly identified the costs for land, solar panels, and maintenance, and calculated the total cost for the first year of operations as a function of the number of square feet.

```python
prompt = f"""
Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem. 
- Then compare your solution to the student's solution \ 
and evaluate if the student's solution is correct or not. 
Don't decide if the student's solution is correct until 
you have done the problem yourself.

Use the following format:
Question:
```
question here
```
Student's solution:
```
student's solution here
```
Actual solution:
```
steps to work out the solution and your solution here
```
Is the student's solution the same as actual solution \
just calculated:
```
yes or no
```
Student grade:
```
correct or incorrect
```
Question:
```
I'm building a solar power installation and I need help \
working out thefinancials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations \
as a function of the number of square feet.
``` 
Student's solution:
```
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost:100x + 250x + 100,000 + 100x = 450x + 100,000
```
Actual solution:
"""
response = get_completion(prompt)
print(response)
```

To calculate the total cost for the first year of operations, we need to add up the costs of land, solar panels, and maintenance.

1. Land cost: $100 / square foot
The cost of land is $100 multiplied by the number of square feet.

2. Solar panel cost: $250 / square foot
The cost of solar panels is $250 multiplied by the number of square feet.3. Maintenance cost: $100,000 + $10 / square foot
The maintenance cost is a flat fee of $100,000 per year, plus $10 multiplied by the number of square feet.

Total cost: Land cost + Solar panel cost + Maintenance cost

So the actual solution is:
Total cost = (100 * x) + (250 * x) + (100,000 + (10 * x))

Is the student's solution the same as the actual solution just calculated:
No

Student grade:
Incorrect

**3.1 Illusion**

```python
prompt =f"""
Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie
"""
response = get_completion(prompt)
print(response)
```

The AeroGlide UltraSlim Smart Toothbrush by Boie is a technologically advanced toothbrush designed to provide a superior brushing experience. Boie is a company known for its innovative oral care products, and the AeroGlide UltraSlim Smart Toothbrush is no exception.

One of the standout features of this toothbrush is its ultra-slim design. The brush head is only2mm thick, making it much thinner than traditional toothbrushes. This slim profile allows for better access to hard-to-reach areas of the mouth, ensuring a thorough and effective clean.

The AeroGlide UltraSlim Smart Toothbrush also incorporates smart technology. It connects to a mobile app via Bluetooth, allowing users to track their brushing habits and receive personalized recommendations for improving their oral hygiene routine. The app provides real-time feedback on brushing technologye, duration, and coverage, helping users to achieve optimal oral health.

The toothbrush features soft, antimicrobial bristles made from a durable thermoplastic elastomer. These bristles are gentle on the gums and teeth, while also being effective at removing plaque and debris. The antimicrobial properties help to keep the brush head clean and hygienic between uses.

Another notable feature of the AeroGlide UltraSlim Smart Toothbrush is its long battery life. It can last up to 30days on a single charge, making it convenient for travel or everyday use without the need for frequent recharging.

Overall, the AeroGlide UltraSlim Smart Toothbrush by Boie offers a combination of advanced technology, slim design, and effective cleaning capabilities. It is a great option for those looking to upgrade their oral care routine and achieve a healthier smile.