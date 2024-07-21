# Chapter 3 Iterative Optimization

When developing a large language model application, it is difficult to get a perfectly applicable prompt in the first attempt. But the key is to have a **good iterative optimization process** to continuously improve the prompt. Compared with training a machine learning model, the success rate of the prompt may be higher, but it still needs to be found through multiple iterations to find the most suitable form for the application.

This chapter takes the product manual to generate marketing copy as an example to show the idea of ​​iterative optimization of the prompt. This is similar to the machine learning model development process demonstrated by Andrew Ng in the machine learning course: after having an idea, write code, obtain data, train the model, and view the results. Analyze the errors to find the applicable field, adjust the plan and train again. Prompt development also adopts a similar loop iteration method to gradually approach the optimal. Specifically, after having a task idea, you can first write the first version of the prompt, pay attention to clarity and give the model enough time to think. Check the results after running. If it is not ideal, analyze the reasons such as the prompt is not clear enough or the thinking time is not enough, make improvements, and run again. After multiple cycles, you will eventually find a prompt suitable for the application.

![1](../figures/C1/Iterative-Prompt-Develelopment.png)

<div align=center>Figure 1.3 Prompt iterative optimization process </div>

In short, it is difficult to have a so-called "best prompt" that is applicable to everything in the world. The key to developing an efficient prompt is to find a good iterative optimization process, rather than requiring perfection from the beginning. Through rapid trial and error iteration, the best prompt form that meets the specific application can be effectively determined.

## 1. Generate marketing product descriptions from product manuals

Given a data page for a chair. The description says that it belongs to the *medieval inspiration* series, produced in Italy, and introduces parameters such as materials, construction, size, optional accessories, etc. Let's say you want to use this fact sheet to help your marketing team write marketing descriptions for an e-commerce platform:

```python
# Example: Product fact sheet
fact_sheet_chair = """
Overview

Part of a beautiful collection of mid-century style office furniture, including filing cabinets, desks, bookcases, conference tables, and more.
Available in a variety of shell colors and base finishes.
Available in plastic front and back upholstery (SWC-100) or fully upholstered in 10 fabrics and 6 leathers (SWC-110).
Base finish options are: stainless steel, matte black, glossy white, or chrome.
Chair available with or without arms.
For home or commercial use.
Qualified for contract use.

Construction

Plastic-coated aluminum base on five wheels.
Pneumatic chair adjustment for easy lifting and lowering.

DimensionsWidth 53 cm | 20.87 in
Depth 51 cm | 20.08 in
Height 80 cm | 31.50 in
Seat height 44 cm | 17.32 in
Seat depth 41 cm | 16.14 in

Options

Soft or hard floor roller options.
Two seat foam densities available: medium (1.8 lbs/ft3) or high (2.8 lbs/ft3).
No armrests or 8 position PU armrests.

Materials
Shell base slides

Cast aluminum with modified nylon PA6/PA66 coating.
Shell thickness: 10 mm.
Seat
HD36 Foam

Country of Origin

Italy
"""
```

### 1.1 Initial prompt

```python
from tool import get_completion

# Prompt: Create a marketing description based on the technical specification
prompt = f"""
Your task is to help the marketing team create a marketing description for a product based on the technical specification.

Write a product description based on the information provided in the technical specification marked ```.

Technical specification: ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)

```

Product Description:

We are proud to introduce a beautiful collection of mid-century style office furniture, which includes filing cabinets, desks, bookcases, conference tables and more. Our products are available in a variety of shell colors and base coatings to meet your individual needs. You can choose plastic front and back backrest decoration (SWC-100) or full decoration in 10 fabrics and 6 leathers (SWC-110) to make your office more comfortable and stylish.

Our base coating options include stainless steel, matte black, glossy white or chrome to meet your different needs. The chair is available with or without armrests, suitable for home or commercial places. Our products are eligible for contract use, providing you with more reliable protection.

Our products have a plastic-coated aluminum base with five wheels, pneumatic chair adjustment, and convenient lifting and lowering. The dimensions are width 53 cm | 20.87 inches, depth 51 cm | 20.08 inches, height 80 cm | 31.50 inches, seat height 44 cm | 17.32 inches, seat depth 41 cm | 16.14 inches, providing you with a comfortable use experience.

Our products also offer soft floor or hard floor roller options, two seat foam densities: medium (1.8 lbs/cubic foot) or high (2.8 lbs/cubic foot), and armless or 8-position PU armrests to meet your different needs.Our products use a modified nylon PA6/PA66 coated cast aluminum shell base slide, the shell thickness is 10 mm, and the seat uses HD36 foam, which provides you with a more comfortable use experience. Our products are made in Italy, providing you with a better quality assurance.

### 1.2 Prompt Optimization 1: Solve the problem of too long generated text

It seems to have completed the requirement well, that is, to write a product description starting from the technical specification, introducing a sophisticated medieval style office chair. But when I saw this generated content, I felt that it was **too long**.

After seeing the first version of the marketing copy generated by the language model based on the product description, we noticed that the text length was too long and was not suitable for use as a concise e-commerce slogan. So at this time, Prompt needs to be optimized and improved. Specifically, the first version of the result meets the requirements of converting from technical specifications to marketing copy, describing the details of the medieval style office chair. But the overly lengthy text is not suitable for e-commerce scenarios. At this time, we can **add a length limit in Prompt** to require the generation of more concise copy.

Extract the answer and split it according to the space. The Chinese answer is 97 words, which meets the design requirements well.

```python
# Optimized prompt, requiring the generated description to be no more than 50 words
prompt = f"""
Your task is to help the marketing team create a product based on the technical specification.Description of the sales website.

Write a product description based on the information provided in the technical sheet marked with ```.

Use up to 50 words.

Technical specifications: ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)

```

A collection of mid-century style office furniture, including filing cabinets, desks, bookcases, conference tables, and more. Available in a variety of colors and finishes, with or without armrests. Base coating options are stainless steel, matte black, glossy white, or chrome. Suitable for home or commercial use, eligible for contract use. Made in Italy.

Let's calculate the length of the output.

```python
# Since Chinese requires word segmentation, the overall length is calculated directly here
len(response)
```

97

When in Prompt When setting a length limit in , the output length generated by the language model does not always meet the requirement exactly, but it can basically be controlled within an acceptable error range. For example, if a text of 50 words is required, the language model sometimes generates an output of about 60 words, but the overall length is close to the predetermined length.

This is because the language model relies on the word segmenter when calculating and judging the length of the text, and the word segmenter does not have perfect accuracy in character statistics. There are currently many ways to try to control the length of the output generated by the language model., such as specifying the number of sentences, words, and Chinese characters.

Although the language model is not 100% accurate in following the length constraint, the best length prompt expression can be found through iterative testing so that the generated text basically meets the length requirements. This requires developers to have a certain understanding of the length judgment mechanism of the language model and be willing to conduct multiple experiments to determine the most reliable length setting method.

### 1.3 Prompt Optimization 2: Handling the details of the wrongly captured text

In the process of iterative optimization of Prompt, we also need to pay attention to whether the details of the text generated by the language model meet expectations.

For example, in this case, further analysis will find that the chair is actually aimed at furniture retailers, not end consumers. Therefore, the generated copy emphasizes style, atmosphere, etc. too much, and rarely involves product technical details, which is not consistent with the focus of the target audience. At this time, we can continue to adjust Prompt, explicitly requiring the language model to generate descriptions for furniture retailers, and pay more attention to technical expressions such as materials, craftsmanship, and structure.

By iteratively analyzing the results and checking whether the correct details are captured, we can gradually optimize Prompt so that the text generated by the language model is more in line with the expected style and content requirements. Precise control of details is very important in language generation tasks. We need to train language models to output texts that are suitable in style and content according to different target audiences’ attention to different aspects.

```python
# 優The Prompt after the object orientation, what nature and focus should it have
prompt = f"""
Your task is to help the marketing team create a retail website description of a product based on the technical specification.

Write a product description based on the information provided in the technical specification marked with ```.

This description is for furniture retailers, so it should be technical in nature and focus on the material construction of the product.

Use a maximum of 50 words.

Technical specifications: ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
```

This mid-century style office furniture series includes file cabinets, desks, bookcases and conference tables, etc., suitable for home or commercial settings. Choose from a variety of shell colors and base coatings, with base coating options of stainless steel, matte black, glossy white or chrome. The chair is available with or without armrests, with soft floor or hard floor rollers, and two seat foam densities. The shell base slide is cast aluminum with modified nylon PA6/PA66 coating, and the seat is HD36 foam. The country of origin is Italy.

As you can see, by modifying the Prompt, the model's focus is on specific features and technical details.

I might further want to show the product ID at the end of the description. Therefore, I can further improve this Prompt and requireAt the end of the description, display the 7-character product ID from the technical specs.

```python
# Going Further
prompt = f"""
Your task is to help the marketing team create a retail website description for a product based on the technical specs.

Write a product description based on the information provided in the technical specs marked ```.

This description is for furniture retailers, so it should be technical in nature and focus on the material construction of the product.

At the end of the description, include each 7-character product ID from the technical specs.

Use a maximum of 50 words.

Technical Specs: ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
```

This mid-century style office furniture collection includes filing cabinets, desks, credenzas, and conference tables for home or commercial use. Available in a variety of shell colors and base coatings, the base coating options are stainless steel, matte black, glossy white, or chrome. The chair is available with or without armrests, with a choice of plastic front and back upholstery or full upholstery in 10 fabrics and 6 leathers. Seat in HD36 foam, available in medium or high density, seat height 44 cm, depth 41 cm. Shell base slide in cast aluminum with modified nylon PA6/PA66 coating, shell thickness 10 mm. Country of origin Italy. Product ID: SWC-100/SWC-110.

Through the above examples, we can see the general process of iterative optimization of prompts. Similar to training machine learning models, designing efficient prompts also requires multiple versions of trial and error adjustments.

Specifically, the first version of the prompt should meet the two principles of clarity and giving the model time to think. On this basis, the general iterative process is: first try a preliminary version, analyze the results, and then continue to improve the prompt, gradually approaching the optimal. Many successful prompts are obtained through this multiple rounds of adjustments.

Later I will show a more complex prompt case to give you a deeper understanding of the powerful capabilities of language models. But before that, I want to emphasize that prompt design is a step-by-step process. Developers need to be mentally prepared for multiple trials and errors, and through continuous adjustment and optimization, they can find the prompt form that best meets the needs of specific scenarios. This requires wisdom and perseverance, but the results are often worth it.

Let’s continue exploring the secrets of prompt engineering and develop amazing large language model applications!

### 1.4 Prompt Optimization 3: Add Table Description
Continue to add instructions to extract product size information and organize it into a table, and specify the columns, table name and format of the table; then format everything into HTML that can be used on the web.

```python
# Ask it to extract information and organize it into a table, and specify the columns of the table, table name and format
prompt = f"""
Your task is to help the marketing team create a retail website description of a product based on the technical sheet.

Write a product description based on the information provided in the technical sheet marked ```.

This description is intended for furniture retailers and should be technical in nature and focus on the material construction of the product.

At the end of the description, include each 7-character product ID from the technical sheet.

After the description, include a table that provides the dimensions of the product. The table should have two columns. The first column includes the name of the dimension. The second column includes only the measurement in inches.

Name the table "Product Dimensions".

Format everything into HTML that can be used for a website. Place the description in a <div> element.

Technical Specs: ```{fact_sheet_chair}```
"""

response = get_completion(prompt)
print(response)
```

<div>
<h2>Mid-Century Office Furniture Series Chair</h2>
<p>This chair is part of a collection of mid-century office furniture suitable for home or commercial use. It’s available in a variety of shell colors and base finishes, including stainless steel, matte black, glossy white, or chrome. You can choose the chair with or without armrests, and with soft-floor or hard-floor roller options. Plus, you can choose between two seat foam densities.Strength: Medium (1.8 lbs/cubic foot) or High (2.8 lbs/cubic foot). </p>
<p>The chair's shell base slide is cast aluminum with a modified nylon PA6/PA66 coating, and the shell thickness is 10 mm. The seat is HD36 foam, and the base is a plastic-coated aluminum base with five wheels, which allows for pneumatic chair adjustment for easy lifting and lowering. In addition, the chair is qualified for contract use and is an ideal choice for you. </p>
<p>Product ID: SWC-100</p>
</div>

<table>
<caption>Product Dimensions</caption>
<tr>
<th>Width</th>
<td>20.87 inches</td>
</tr>
<tr>
<th>Depth</th>
<td>20.08 inches</td>
</tr>
<tr>
<th>Height</th>
<td>31.50 inches</td>
</tr>
<tr>
<th>Seat Height</th>
<td>17.32 inches</td></tr>
<tr>
<th>Seat Depth</th>
<td>16.14 inches</td>
</tr>
</table>

The above output is HTML code, which we can load using Python's IPython library.

```python
# The table is rendered in HTML format and loaded
from IPython.display import display, HTML

display(HTML(response))
```

<div>
<h2>Mid-Century Office Furniture Series Chair</h2>
<p>This chair is part of the Mid-Century Office Furniture Series and is suitable for home or commercial settings. It is available in a variety of shell colors and base coatings, including stainless steel, matte black, glossy white, or chrome. You can choose the chair with or without armrests, and soft or hard floor roller options. In addition, you can choose between two seat foam densities: medium (1.8 pounds per cubic foot) or high (2.8 pounds per cubic foot). </p>
<p>The chair's shell base slide is cast aluminum with a modified nylon PA6/PA66 coating, with a shell thickness of 10 mm. The seat is in HD36 foam, and the base is a plastic-coated aluminum base with five wheels that can bePneumatic chair adjustment for easy lifting and lowering. In addition, the chair is eligible for contract use, making it an ideal choice. </p>
<p>Product ID: SWC-100</p>
</div>

<table>
<caption>Product size</caption>
<tr>
<th>Width</th>
<td>20.87 inches</td>
</tr>
<tr>
<th>Depth</th>
<td>20.08 inches</td>
</tr>
<tr>
<th>Height</th>
<td>31.50 inches</td>
</tr>
<tr>
<th>Seat height</th>
<td>17.32 inches</td>
</tr>
<tr>
<th>Seat depth</th>
<td>16.14 inches</td>
</tr>
</table>

## 2. Summary

This chapter focuses on the process of iteratively optimizing Prompt when developing large language model applications. As a prompt engineer, the key is not to require a perfect prompt from the beginning, but to master an effective prompt development process.

Specifically, first write the first version of Promptompt, and then gradually improve it through multiple rounds of adjustments until satisfactory results are generated. For more complex applications, iterative training can be performed on multiple samples to evaluate the average performance of Prompt. Only after the application is relatively mature, it is necessary to use the method of evaluating Prompt performance on multiple sample sets for detailed optimization. Because this requires higher computing resources.

In short, the core of Prompt engineers is to master the iterative development and optimization skills of Prompt, rather than requiring 100% perfection from the beginning. Through continuous adjustment and trial and error, finally finding a reliable and applicable Prompt form is the correct way to design Prompt.

Readers can practice the examples given in this chapter on Jupyter Notebook, modify Prompt and observe different outputs to deeply understand the process of iterative optimization of Prompt. This will provide a good practical preparation for further development of complex language model applications.

## III. English version

**Product manual**

```python
fact_sheet_chair = """
OVERVIEW
- Part of a beautiful family of mid-century inspired office furniture, 
including filing cabinets, desks, bookcases, meeting tables, and more. - Several options of shell color and base finishes. - Available with plastic back and front upholstery (SWC-100) or full upholstery (SWC-110) in 10 fabric and 6 leather options. - Base finish options are: stainless steel, matte black, gloss white, or chrome. - Chair is available with or without armrests. - Suitable for home or business settings. - Qualified for contract use. CONSTRUCTION - 5-wheel plastic coated aluminum base. - Pneumatic adjust chairfor easy raise/lower action. DIMENSIONS - WIDTH 53 CM | 20.87” - DEPTH 51 CM | 20.08” - HEIGHT 80 CM | 31.50” - SEAT HEIGHT 44 CM | 17.32” - SEAT DEPTH 41 CM | 16.14” OPTIONS - Soft or hard -floor caster options. - Two choices of seat foam densities: medium (1.8 lb/ft3) or high (2.8 lb/ft3) - Armless or 8 position PU armrests MATERIALS SHELL BASE GLIDER - Cast Aluminum with modified nylon PA6/PA66 coating. - Shell thickness: 10 mm. SEAT - HD36 foam COUNTRY OF ORIGIN - Italy
"""
```

**1.1English initial prompt** ```python # Prompt: Generate marketing description based on instructions prompt = f""" Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet. Write a product description based on the information provided in the technical specifications delimited by triple backticks. Technical specifications: ```{fact_sheet_chair}``` """ response = get_completion(prompt) print(response) ```` Introducing our stunning mid-century inspired office chair, the perfectt addition to any home or business setting. This chair is part of a beautiful family of office furniture, including filing cabinets, desks, bookcases, meeting tables, and more, all designed with a timeless mid-century aesthetic. One of the standout features of this chair is the variety of customization options available. You can choose from several shell colors and base finishes to perfectly match your existing decor. The chair is available with either plastic back and front upholsteryor full upholstery in a range of 10 fabric and 6 leather options, allowing you to create a look that is uniquely yours. The chair is also available with or without armrests, giving you the flexibility to choose the option that best suits your needs. base finish options include stainless steel, matte black, gloss white, or chrome, ensuring that you can find the perfect match for your space. In terms of construction, this chair is built to last. It features a 5-wheel plasticcoated aluminum base, providing stability and mobility. The pneumatic chair adjust allows for easy raise and lower action, ensuring optimal comfort throughout the day. When it comes to dimensions, this chair is designed with both style and comfort in mind. With a width of 53 cm (20.87"), depth of 51 cm (20.08"), and height of 80 cm (31.50"), it offers ample space without overwhelming your space. The seat height is 44 cm (17.32") and the seat depth is 41 cm (16.14"), providing a comfortable seating experience for extended periods. We understand that every space is unique, which is why we offer a range of options to further customize your chair. You can choose between soft or hard-floor caster options, ensuring that your chair glides smoothly on any surface. Additionally, you have the choice between two seat foam densities: medium (1.8 lb/ft3) or high (2.8 lb/ft3), allowing you to select the level of support that suits your preferences. The chair is also available witharmless design or 8 position PU armrests, providing additional comfort and versatility. When it comes to materials, this chair is crafted with the utmost attention to quality. The shell base glider is made of cast aluminum with a modified nylon PA6/PA66 coating, ensuring durability and longevity. The shell thickness is 10 mm, providing a sturdy and reliable structure. The seat is made of HD36 foam, offering a comfortable and supportive seating experience. Finally, this chair isproudly made in Italy, known for its exceptional craftsmanship and attention to detail. With its timeless design and superior construction, this chair is not only a stylish addition to any space but also a reliable and functional piece of furniture. Upgrade your office or home with our mid-century inspired office chair and experience the perfect blend of style, comfort, and functionality. **1.2 Limit the generated length** ```python # Optimized Prompt, requiring the generated description to be no more than 50 words prompt = f" "" Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet. Write a product description based on the information provided in the technical specifications delimited by triple backticks. Use at most 50 words. Technical specifications: ``{ fact_sheet_chair}``` """ response = get_completion(prompt) print(response) ``` Introducing our mid-century inspired office chair, part of a beautiful furniture collection. With various color and finish options,it can be customized to suit any space. Choose between plastic or full upholstery in a range of fabrics and leathers. The chair features a durable aluminum base and easy height adjustment. Suitable for both home and business use. Made in Italy.

```python
lst = response.split()
print(len(lst))
```

60

**1.3 Handle the details of the captured error text**

```python
# Optimized Prompt, explain object-oriented, what properties it should have and what aspects it should focus on
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet. Write a product description based on the information provided in the technical specifications delimited by triple backticks. The description is intended for furniture retailers, so should be technical in nature and focus on the materials the product is constructed from. Use at most 50 words. Technical specifications: ```{fact_sheet_chair}``` """ response = get_completion(prompt) print(response) ``` Introducing our mid-century inspired office chair, part of a beautiful furniture collection. With various shell colors and base finishes, it offers versatility for any setting. Choose between plastic or full upholstery in a range of fabric and leather options. The chair features a durable aluminum base with 5-wheel design and pneumatic chair adjustment . Made in Italy. ```python # Go one step further and require a 7-character product ID at the end of the description prompt = f""" Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet. Write a product description based on the information provided in the technical specifications delimited by triple backticks. The description is intended for furniture retailers, so should be technical in nature and focus on the materials the product is constructed from. At the end of the description, include every 7-character Product ID in the technical specification. Use at most 50 words. Technical specifications: ```{fact_sheet_chair}``` """ response = get_completion(prompt) print(response) ```Introducing our mid-century inspired office chair, part of a beautiful family of furniture. This chair offers a range of options, including different shell colors and base finishes. Choose between plastic or full upholstery in various fabric and leather options. The chair is constructed with a 5-wheel plastic coated aluminum base and features a pneumatic chair adjust for easy raise/lower action. With its sleek design and multiple customization options, this chair is suitable for both home andbusiness settings. Made in Italy.

Product IDs: SWC-100, SWC-110

**1.4 Add table description in English**

```python
# Ask it to extract information and organize it into a table, and specify the columns, table name and format of the table
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

The description is intended for furniture retailers, 
so it should be technical in nature and focus on the materials the product is constructed from. At the end of the description, include every 7-character Product ID in the technical specification. After the description, include a table that gives the product's dimensions. The table should have two columns. In the first column include the name of the dimension. In the second column include the measurements in inches only. Give the table the title 'Product Dimensions'. Format everything as HTML that can be used in a website. Place the description in a <div> element. Technical specifications: ```{fact_sheet_chair}``` """ response = get_completion(prompt) print(response) # The table is presented in HTML format and loaded from IPython.display import display, HTML display(HTML(response)) ``` <div> <h2>Product Description</h2> <p> Introducing our latest addition to our mid-century inspired office furniture collection, the SWC-100 Chair. This chair is part of a beautiful family of furniture that includes filing cabinets, desks,bookcases, meeting tables, and more. With its sleek design and customizable options, it is perfect for both home and business settings. </p> <p> The SWC-100 Chair is available in several options of shell color and base finishes, allowing you to choose the perfect combination to match your space. You can opt for plastic back and front upholstery or full upholstery in a variety of fabric and leather options. The base finish options include stainless steel, matte black, gloss white, or chrome. Additionally, you have the choice of having armrests or going armless. </p> <p> Constructed with durability and comfort in mind, the SWC-100 Chair features a 5-wheel plastic coated aluminum base for stability and mobility. The chair also has a pneumatic adjuster, allowing for easy raise and lower action to find the perfect height for your needs. </p> <p> The SWC-100 Chair is designed to provide maximum comfort and support. The seat is madewith HD36 foam, ensuring a plush and comfortable seating experience. You also have the option to choose between soft or hard-floor casters, depending on your flooring needs. Additionally, you can select from two choices of seat foam densities: medium (1.8 lb /ft3) or high (2.8 lb/ft3). The chair is also available with 8 position PU armrests for added convenience. </p> <p> Made with high-quality materials, the SWC-100 Chair is built to last. The shell base glider is constructedd with cast aluminum and modified nylon PA6/PA66 coating, providing durability and stability. The shell has a thickness of 10 mm, ensuring strength and longevity. The chair is proudly made in Italy, known for its craftsmanship and attention to detail. </ p> <p> Whether you need a chair for your home office or a professional workspace, the SWC-100 Chair is the perfect choice. Its stylish design, customizable options, and high-quality construction make it a standout piece of furniture that will enhance any space. </p> <h2>Product Dimensions</h2> <table> <tr> <th>Dimension</th> <th>Measurement (inches)</th> </tr> < tr> <td>Width</td> <td>20.87"</td> </tr> <tr> <td>Depth</td> <td>20.08"</td> </tr> <tr> <td>Height</td> <td>31.50"</td> </tr> <tr> <td>Seat Height</td> <td>17.32"</td></tr> <tr> <td>Seat Depth</td> <td>16.14"</td> </tr> </table> </div> Product IDs: SWC-100, SWC-110 <div> <h2>Product Description</h2> <p> Introducing our latest addition to our mid-century inspired office furniture collection, the SWC-100 Chair. This chair is part of a beautiful family of furniture that includes filing cabinets, desks, bookcases , meeting tables, and more. With its sleek design and customizable options, it is pperfect for both home and business settings. </p> <p> The SWC-100 Chair is available in several options of shell color and base finishes, allowing you to choose the perfect combination to match your space. You can opt for plastic back and front upholstery or full upholstery in a variety of fabric and leather options. The base finish options include stainless steel, matte black, gloss white, or chrome. Additionally, you have the choice of having armrests or going armless. </p> < p>Constructed with durability and comfort in mind, the SWC-100 Chair features a 5-wheel plastic coated aluminum base for stability and mobility. The chair also has a pneumatic adjuster, allowing for easy raise and lower action to find the perfect height for your needs . </p> <p> The SWC-100 Chair is designed to provide maximum comfort and support. The seat is made with HD36 foam, ensuring a plush and comfortable seating experience. You also have the option to choose between soft or hard- floor casters, depending on your flooring needs. Additionally, you can select from two choices of seat foam densities: medium (1.8 lb/ft3) or high (2.8 lb/ft3). The chair is also available with 8 position PU armrests for added convenience. </p> <p> Made with high-quality materials, the SWC-100 Chair is built to last. The shell base glider is constructed with cast aluminum and modified nylon PA6/PA66 coating, providing durability and stability. has a thickness of 10 mm, ensuring ststrength and longevity. The chair is proudly made in Italy, known for its craftsmanship and attention to detail. </p> <p> Whether you need a chair for your home office or a professional workspace, the SWC-100 Chair is the perfect choice. Its stylish design, customizable options, and high-quality construction make it a standout piece of furniture that will enhance any space. </p> <h2>Product Dimensions</h2> <table> <tr> <th>Dimension </th> <th>Measurement (inches)</th></tr> <tr> <td>Width</td> <td>20.87"</td> </tr> <tr> <td>Depth</td> <td>20.08"</td> </tr> tr> <tr> <td>Height</td> <td>31.50"</td> </tr> <tr> <td>Seat Height</td> <td>17.32"</td> </tr > <tr> <td>Seat Depth</td> <td>16.14"</td> </tr> </table> </div> Product IDs: SWC-100, SWC-110