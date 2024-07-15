# Chapter 1 Introduction

## 1.1 What is a language model
Here we start from the perspective of statistics or statistical learning. What we hope to achieve is to give the corresponding new text/symbol output (which can be text translation, text classification, text expansion) based on the given text information input.
To achieve such a task, we need to solve two problems:
1) Input sequence problem: Since the input here is a text signal, and the computer can enter the neural network to process and calculate the numerical value, so we need to convert the characters into numerical values ​​in a certain way.
2) Output sequence problem: Since the part to be output is also text, and the output of the neural network is numerical type (classification problem: binary classification problem corresponds to 01 output, multi-classification corresponds to multiple 01 outputs; regression problem: corresponding to numerical type output), it is necessary to establish a mapping relationship between the numerical type output of the neural network and the final character output.
For the first problem, there are actually many ways to deal with it. For example, the simplest one is that we can encode the input sequence to convert the characters into numerical values.
> Example:
Assume that the entire symbol system has only two lowercase letters 'a' and 'b' and a period '.'
The input sequence is: 'ab.b'
Here we use the simplest one-hot encoding (you can search for the corresponding concept), a total of 3 characters, and then add the start and end characters of the two strings '<begin>' and '<end>', the corresponding relationship is
'<begin>':[0,0,0,0]; 'a':[1,0,0,0]; 'b':[0,1,0,0]; '.':[0,0,1,0]; '<end>':[0,0,0,1]
Then the input sequence can be encoded as:
\[[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,1,0,0],[0,0,0,1]]
Then this sequence encoding can be entered into the neural network for calculation.

For the second problem, similar to the first problem, there are actually many ways to deal with it. We can encode the input sequence to convert the numerical value into a character.
> Example:
Assume that the entire symbol system has only two lowercase letters 'a' and 'b' and a period '.'
The input sequence is: 'ab.b', and the expected output sequence is 'b.'
Here, the simplest one-hot encoding is also used (the corresponding concept can be searched), with a total of 4 characters, and the corresponding relationship is
[0,0,0,0]:'<begin>'; [1,0,0,0]:'a'; [0,1,0,0]:'b'; [0,0,1,0]:'.'; [0,0,0,1]:'<end>';
The output of the neural network can be constructed as four categories. Assuming that the model output is \[0,0,1,0], the output character 'b' is obtained through the mapping relationship, and the predicted result is passed into the model to obtain the next output \[0,0,0,1], through the mappingThe relationship gets its output character '.', and then the predicted result is passed into the model to get the next output \[0,0,0,1], and the output character '<end>' is obtained through the mapping relationship, thus ending the entire output. 

The classic definition of a language model (LM) is a probability distribution of a token sequence. Suppose we have a vocabulary $V$ of token sets. The language model p assigns a probability (a number between 0 and 1) to each token sequence $x_{1},...,x_{L}$ ∈ $V$:

$$
p(x_1, \dots, x_L)
$$

The probability intuitively tells us how "good" a token sequence is. For example, if the vocabulary is {ate, ball, cheese, mouse, the}, the language model might assign the following probabilities (demonstration):

$$
p(\text{the, mouse, ate, the, cheese}) = 0.02,
$$

$$
p(\text{the, cheese ate, the, mouse}) = 0.01,
$$

$$
p(\text{mouse, the, the, cheese, ate}) = 0.0001,
$$

Mathematically, a language model is a very simple and beautiful object. But this simplicity isis deceptive: the ability to assign (meaningful) probabilities to all sequences requires the language model to have extraordinary (but implicit) linguistic abilities and world knowledge.

For example, a language model should implicitly assign a very low probability to "𝗆𝗈𝗎𝗌𝖾 𝗍𝗁𝖾 𝗍𝗁𝖾 𝖼𝗁𝖾𝖾𝗌𝖾 𝖺𝗍𝖾" because it is grammatically incorrect (syntactic knowledge). Due to the existence of world knowledge, the language model should implicitly assign a higher probability to "𝗍𝗁𝖾 𝗆𝗈𝗎𝗌𝖾 𝖺𝗍𝖾 𝗍𝗁𝖾 𝖼𝗁𝖾𝖾𝗌𝖾" than "𝗍𝗁𝖾 𝖼𝗁𝖾𝖾𝗌𝖾 𝖺𝗍𝖾 𝗍𝗁𝖾 𝗆𝗈𝗎𝗌𝖾". This is because the two sentences are syntactically identical but semantically different, and the language model needs to have excellent language capabilities and world knowledge to accurately evaluate the probability of a sequence.

Language models can also do generation tasks. As defined, a language model p takes a sequence and returns a probability to evaluate how good it is. We can also generate a sequence based on a language model. The purest approach is to sample from the language model $p$ with probability $p(x_{1:L})$, expressed as:

$$
x_{1:L}∼p.
$$

How to do this computationally efficiently depends on the form of the language model p. In practice, we usually do not sample directly from the language model, both because of the limitations of the real language model and because sometimes we do not want to get an "average" sequence, but a result closer to the "optimal" sequence.

### 1.1.1 Autoregressive language models

A common way to write the joint distribution $p(x_{1:L})$ of a sequence $x_{1:L}$ is to use the chain rule of probability:

$$
p(x_{1:L}) = p(x_1) p(x_2 \mid x_1) p(x_3 \mid x_1, x_2) \cdots p(x_L \mid x_{1:L-1}) = \prod_{i=1}^L p(x_i \mid x_{1:i-1}).
$$

Here is a text-based example:

$$
\begin{align*} p({the}, {mouse}, {ate}, {the}, {cheese}) = \, & p({the}) \\ & p({mouse} \mid {the}) \\ & p({ate} \mid {the}, {mouse}) \\ & p({the} \mid {the}, {mouse}, {ate}) \\ & p({cheese} \mid {the}, {mouse}, {ate}, {the}). \end{align*}
$$

SpecialSpecifically, we need to understand that $p(x_{i}|x_{1:i−1})$ is the conditional probability distribution of the next token $x_{i}$ given the previous token $x_{1:i−1}$. Mathematically, any joint probability distribution can be represented in this way. However, the characteristic of the autoregressive language model is that it can efficiently calculate each conditional probability distribution $p(x_{i}|x_{1:i−1})$ using methods such as feedforward neural networks. To generate the entire sequence $x_{1:L}$ in an autoregressive language model $p$, we need to generate one token at a time, calculated based on previously generated tokens:

$$
\begin{aligned}
\text { for } i & =1, \ldots, L: \\
x_i & \sim p\left(x_i \mid x_{1: i-1}\right)^{1 / T},
\end{aligned}
$$

Where $T≥0$ is a temperature parameter that controls how much randomness we want from the language model:
- T=0: deterministically select the most likely token $x_{i}$ at each position i
- T=1: sample “normally” from a pure language model
- T=∞: sample uniformly from the entire vocabularySampling from a cloth
However, if we only raise the probability to the power of $1/T$, the probability distribution may not sum to 1. We can fix this by renormalizing the distribution. We call the normalized version $p_{T}(x_{i}|x_{1:i−1})∝p(x_{i}|x_{1:i−1})^{1/T}$ the annealed conditional probability distribution. For example:

$$
\begin{array}{cl}
p(\text { cheese })=0.4, & p(\text { mouse })=0.6 \\
p_{T=0.5}(\text { cheese })=0.31, & \left.p_{T=0.5} \text { (mouse }\right)=0.69 \\
\left.p_{T=0.2} \text { (cheese }\right)=0.12, & p_{T=0.2} \text { (mouse) }=0.88 \\
\left.p_{T=0} \text { (cheese }\right)=0, & \left.p_{T=0} \text { (mouse }\right)=1
\end{array}
$$

Specifically, this temperature parameter will be appliedThe conditional probability distribution $p(x_{i}|x_{1:i−1})$ at each step is raised to the power of $1/T$. This means that when $T$ is high, we get a more average probability distribution and the generated results are more random; conversely, when $T$ is low, the model will be more inclined to generate tokens with higher probability.

However, there is an important note: applying a temperature parameter $T$ to the conditional probability distribution at each step and iteratively sampling is not equivalent to (unless $T=1$) sampling from the "annealed" distribution of the entire sequence of length L at once. In other words, the two methods produce different results when $T≠1$.

The term "annealing" comes from metallurgy, where hot metals are gradually cooled to change their physical properties. Here, it is analogous to the process of adjusting the probability distribution. The "annealed" distribution is a new distribution obtained by raising each element of the original probability distribution to the power of $1/T$ and then re-normalizing it. When $T ≠ 1$, this process changes the original probability distribution, so the result of sampling from the "annealed" distribution may be different from the result of applying T to the conditional distribution at each step and iteratively sampling.

For non-autoregressive conditional generation, more generally, we can specify some prefix sequence $x_{1:i}$ (called the prompt) and sample the rest $x_{i+1:L}$ (called the complement)conditional generation. For example, to generate $T=0$:

$$
\underbrace{{the}, {mouse}, {ate}}_\text{prompt} \stackrel{T=0}{\leadsto} \underbrace{{the}, {cheese}}_\text{completion}.
$$

If we change the temperature to $T=1$, we can get more diversity, for example, "its house" and "my homework". We will soon see that conditional generation unlocks the ability of language models to solve a variety of tasks by simply changing the prompt.

### 1.1.2 Summary

- A language model is a probability distribution p over a sequence $x_{1:L}$.
- Intuitively, a good language model should have both linguistic power and world knowledge.
- Autoregressive language models allow efficient generation of completions $x_{i+1:L}$ given a prompt $x_{1:i}$.
- Temperature can be used to control the amount of variation in generation.

## 1.2 Historical review of large models

### 1.2.1 Information theory, entropy of English, n-gram model

The development of language models can be traced back to Claude Shannon, who laid the foundation for communication in his landmark 1948 paper "A Mathematical Theory of Communication".In this paper, he introduced the concept of entropy for measuring probability distribution:

$$
H(p) = \sum_x p(x) \log \frac{1}{p(x)}.
$$

Entropy is actually a measure of the expected number of bits required to encode (i.e. compress) a sample $x∼p$ into a bit string. For example, "the mouse ate the cheese" may be encoded as "0001110101".

The smaller the entropy value, the stronger the structure of the sequence and the shorter the length of the code. Intuitively, $\log \frac{1}{p(x)}$ can be regarded as the length of the code used to represent the element $x$ with a probability of $p(x)$.

For example, if $p(x)=1/8$, we need to allocate $log_{2}(8)=3$ bits (or equivalently, $log(8)=2.08$ natural units).

Note that actually reaching the Shannon limit is very challenging (e.g., low-density parity-check codes), and is a topic of research in coding theory.

#### 1.2.1.1 Entropy of English
Shannon was particularly interested in measuring the entropy of the English language, represented as a sequence of letters. This means that we imagine that there is a "true" distribution p (the existence of which is questionable)., but it is still a useful mathematical abstraction) that produces a sample of English text x∼p.

Shannon also defined the cross entropy:

$$
H(p, q)=-\sum_x p(x) \log q(x)
$$

This measures how many bits (nats) are needed to encode a sample x∼p, using the compression scheme given by the model q (representing x with a code of length 1/q(x)).

Entropy is estimated via a language model. A key property is that the cross entropy H(p,q) is bounded by the entropy H(p):

$$
H(p,q) = \sum_x p(x) \log \frac{1}{q(x)}.
$$

This means that we can estimate $H(p,q)$ by constructing a (language) model $q$ with only samples from the true data distribution $p$, whereas $H(p)$ is usually inaccessible if $p$ is English.

So we can get a better estimate of the entropy H(p) by building a better model q, measured by H(p,q).

Shannon Game (Human Language Model). Shannon first used the n-gram model for q in 1948, but in his 1951 paper "Prediction and Entropy of Printed English", he introduced a clever scheme (called the Shannon Game) where q is provided by humans:
```
"the mouse ate my ho_"
```

People are not good at providing calibrated probabilities for arbitrary text, so in the Shannon GameIn the game, the human language model will repeatedly try to guess the next letter, and then we will record the number of guesses.

#### 1.2.1.2 N-gram model for downstream applications

Language models were first used in practical applications that needed to generate text:
- Speech recognition in the 1970s (input: sound signal, output: text)
- Machine translation in the 1990s (input: text in the source language, output: text in the target language)

Noisy channel model. The main model for solving these tasks at the time was the noisy channel model. Take speech recognition as an example:
- We assume that there are some texts extracted from some distribution p
- These texts are converted into speech (sound signal)
- Then given the speech, we want to recover the (most likely) text. This can be achieved through Bayes' theorem:

$p(\text{text} \mid \text{speech}) \propto \underbrace{p(\text{text})}_\text{language model} \underbrace{p(\text{speech} \mid \text{text})}_\text{acoustic model}.$

Speech recognition and machine translation systems use word-based n-gram language models (first introduced by Shannon, but for characters).

N-gram model. In an n-In the gram model, the prediction of $x_{i}$ depends only on the last $n-1$ characters $x_{i−(n−1):i−1}$, not the entire history:

$$
p(x_i \mid x_{1:i-1}) = p(x_i \mid x_{i-(n-1):i-1}).
$$

For example, a trigram (n=3) model would define:

$$
p(𝖼𝗁𝖾𝖾𝗌𝖾∣𝗍𝗁𝖾,𝗆𝗈𝗎𝗌𝖾,𝖺𝗍𝖾,𝗍𝗁𝖾)=p(𝖼𝗁𝖾𝖾𝗌𝖾∣𝖺𝗍𝖾,𝗍𝗁𝖾).
$$

These probabilities are calculated based on the number of times various n-grams (e.g., 𝖺𝗍𝖾 𝗍𝗁𝖾 𝗆𝗈𝗎𝗌𝖾 and 𝖺𝗍𝖾 𝗍𝗁𝖾 𝖼𝗁𝖾𝖾𝗌𝖾) appear in a large amount of text, and are appropriately smoothed to avoid overfitting (e.g., Kneser-Ney smoothing).

Fitting n-gram models to data is very cheap and scalable. Therefore, n-gram models are trained on large amounts of text. For example, [Brants et al. (2007)](https://aclanthology.org/D07-1090.pdf) trained a 5-gram model for machine translation on 2 trillion tokens. In contrast, GPT-3 was trained on only 300 billion tokens. However, the n-gram model has fundamental limitations. Imagine the following prefix:

```
𝖲𝗍𝖺𝗇𝖿𝗈𝗋𝖽 𝗁𝖺𝗌 𝖺 𝗇𝖾𝗐 𝖼𝗈𝗎𝗋𝗌𝖾 𝗈𝗇 𝗅𝖺𝗋𝗀𝖾 𝗅𝖺𝗇𝗀𝗎𝖺𝗀𝖾 𝗆𝗈𝖽𝖾𝗅𝗌. 𝖨𝗍 𝗐𝗂𝗅𝗅 𝖻𝖾 𝗍𝖺𝗎𝗀𝗁𝗍 𝖻𝗒 ___
```

If n is too small, the model will not be able to capture long-distance dependencies and the next word will not be dependent on 𝖲𝗍𝖺𝗇𝖿𝗈𝗋𝖽. However, if n is too large, it is statistically impossible to get a good estimate of the probability (even in a "large" corpus, almost all reasonably long sequences occur 0 times):

$$
count(𝖲𝗍𝖺𝗇𝖿𝗈𝗋𝖽,𝗁𝖺𝗌,𝖺,𝗇𝖾𝗐,𝖼𝗈𝗎𝗋𝗌𝖾,𝗈𝗇,𝗅𝖺𝗋𝗀𝖾,𝗅𝖺𝗇𝗀𝗎𝖺𝗀𝖾,𝗆𝗈𝖽𝖾𝗅𝗌)=0.
$$

Thus, language models are limited to tasks such as speech recognition and machine translation, where the acoustic signal or source text provides enough information and capturing only local dependencies (and not long-range dependencies) is not a big problem.

#### 1.2.1.3 Neural Language Models

An important advancement in language models is the introduction of neural networks. [Bengio et al.](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) first proposed the neural language model in 2003, where $p(x_{i}|x_{i−(n−1):i−1})$ is given by a neural network:

$$p(cheese|ate,the)=some-neural-network(ate,the,cheese).
$$

Note that the context length is still limited by n, but it is now statistically feasible to estimate neural language models for larger values ​​of n.

However, the main challenge is that training neural networks is much more computationally expensive. They trained a model on just 14 million words and showed that it outperformed n-gram models on the same amount of data. But since n-gram models scale better and data is not a bottleneck, n-gram models will still dominate for at least the next decade.

Since 2003, two key developments in neural language modeling include:
- **Recurrent Neural Networks** (RNNs), including Long Short-Term Memory (LSTMs), which allow the conditional distribution of a token $x_{i}$ to depend on the entire context $x_{1:i−1}$ (effectively making $n=∞$), but these models are difficult to train.
- **Transformers** are a newer architecture (developed in 2017 for machine translation) that again return to a fixed context length n, but are easier to train (and take advantage of GPU parallelism). Also, n can be "large enough" for many applications (GPT-3 uses n=2048).

We will dive deeper into this later in the courseExplore these architectures and training methods.

### 1.2.2 Summary

- Language models were originally studied in the context of information theory and can be used to estimate the entropy of English.

- N-gram models are extremely computationally efficient but statistically inefficient.

- N-gram models are useful in conjunction with another model (acoustic models for speech recognition or translation models for machine translation) for short context lengths.

- Neural language models are statistically efficient but computationally inefficient.

- Over time, training large neural networks has become feasible enough that neural language models have become the dominant model paradigm.

## 1.3 Significance of this course

After introducing language models, one might wonder why we need a course dedicated to large language models.

Increase in size. First, what does "large" mean? With the rise of deep learning in the 2010s and major hardware advances (such as GPUs), the size of neural language models has increased significantly. The following table shows that the size of models has increased 5000 times in the past 4 years:

|Model|Organization|Date|Size (# params)|
|---|---|---|---|
|ELMo|AI2|Feb 2018|94,000,000|
|GPT|OpenAI|Jun 2018|110,000,000|
|BERT|Google|Oct 2018|340,000,000|
|XLM|Facebook|Jan 2019|655,000,000|
|GPT-2|OpenAI|Mar 2019|1,500,000,000|
|RoBERTa|Facebook|Jul 2019|355,000,000|
|Megatron-LM|NVIDIA|Sep 2019|8,300,000,000|
|T5|Google|Oct 2019|11,000,000,000|
|Turing-NLG|Microsoft|Feb 2020|17,000,000,000|
|GPT-3|OpenAI|May 2020|175,000,000,000|
|Megatron-Turing NLG|Microsoft, NVIDIA|Oct 2021|530,000,000,000|
|Gopher|DeepMind|Dec 2021|280,000,000,000|

New emergence. What difference does scale make? Although many technical details are the same, it is surprising that “just scaling up” can produce new emergent behaviors, leading to qualitatively different capabilities andQualitatively different social impact.

Note: On a technical level, we focus on autoregressive language models, but many of the ideas also apply to masked language models such as BERT and RoBERTa.

### 1.3.1 Capabilities

Up to 2018, language models were primarily used as components of larger systems (e.g. speech recognition or machine translation), but are increasingly capable of being standalone systems in ways that were unimaginable in the past.

To recap, language models have the ability to generate conditionally: generate completed text given a prompt:

$$
prompt ⇝ completion
$$

**Example of capabilities**: This simple interface opens up the possibility for language models to solve a wide variety of tasks by varying the prompt. For example, you can ask questions by filling in the blanks with prompts (example):

$$
\text { Frederic, Chopin, was, born, in } \stackrel{T=0}{\leadsto} 1810 \text {, in, Poland }
$$

You can also solve vocabulary analogy problems by prompts (example):

$$
\text { 𝗌𝗄𝗒,:,𝖻𝗅𝗎𝖾,::,𝗀𝗋𝖺𝗌𝗌,: } \stackrel{T=0}{\leadsto} \text {green}
$$

You can also ask questions by giving promptsHere is an example of a GPT-3 generated article (content after the bold text):
```
**Title: NLP Researchers at Stanford Discover Black Holes in Language Models 
Article: On January 3,** 2007, the Stanford University News Service published an article that reported a remarkable discovery by NLP researchers at Stanford. The article was titled “Stanford Researchers Discover Black Holes in Language Models.” The discovery was described as follows: A black hole is a region of space-time where gravity pulls so much that even light canNow physicists think they have found a similar phenomenon in language: They call it the semantic black hole. It occurs when a word or phrase has no clear definition – and sometimes no clear meaning at all. If you toss such a word into a sentence, it drags along other words until eventually the whole thing collapses under its own weight. “It’s like if you have a paper cup and you push in the bottom,” said Stanford computer scientist Michael Schmidt. “At first it holds up fine, but thenen it gets weaker and weaker until it collapses in on itself.” Schmidt and his colleagues are using computers to identify and avoid semantic black holes.
(**Title: Stanford NLP researchers discover black holes in language models
Article: On January 3, 2007, Stanford University News Service published an article titled "Stanford researchers discover black holes in language models", reporting a major discovery by Stanford NLP researchers. The discovery was described as follows: Black holes are regions in space-time where gravity is so strong that even light cannot escape. Now physicists think they have found something similar in language: they call them semantic black holes. Semantic black holes occur when a word or phrase has no clear definition, and sometimes even no clear meaning. If you put such a word into a sentence, it drags down other words, and eventually the entire sentence collapses under its own weight. "It's like you take a paper cup and push on the bottom," said Michael Schmidt, a Stanford computer scientist. "At first it can hold up, but then it gets more and more fragile and eventually collapses." Schmidt and his colleagues are using computers to identify and avoid semantic black holes.)

```

Top and BottomContextual learning. Perhaps the most fascinating thing about GPT-3 is that it can do what is called contextual learning. Let’s start with an example (example):
```
**Input: Where is Stanford University? 
Output:** Stanford University is in California.
```

We can observe that the answer given by GPT-3 is neither the most informative, and perhaps we would prefer to get the answer directly rather than the whole sentence.

Similar to the previous vocabulary analogy, we can construct a prompt that contains examples of input/output. GPT-3 is somehow able to understand the task better and is now able to produce the desired answer (example):
```
**Input: Where is MIT? 
Output: Cambridge 

Input: Where is University of Washington? 
Output: Seattle 

Input: Where is Stanford University? 
Output:** Stanford
```

**Relation to supervised learning**: In normal supervised learning, we specify a dataset of input-output pairs and train a modelA neural network (e.g., via gradient descent) is trained to fit these examples. Each training run results in a different model. However, with contextual learning, only one language model can be used to perform a variety of different tasks given a prompt. Contextual learning is clearly beyond what researchers expected to be possible and is an example of emergent behavior.

Note: Neural language models can also generate vector representations of sentences that can be used as features for downstream tasks or directly fine-tuned for performance optimization. We focus on using language models via conditional generation, which relies solely on black-box access to simplify the problem.

### 1.3.2 Language Models in the Real World
Given the powerful capabilities of language models, their widespread application is not surprising.

**Research Area**: First, in the research field, large language models have revolutionized the natural language processing (NLP) community. Almost all state-of-the-art systems covering a variety of tasks such as sentiment classification, question answering, summarization, and machine translation are based on some type of language model.

**Industry**: For production systems that affect real users, it is difficult to determine the exact situation because most of these systems are closed. Here is a non-exhaustive list of some well-known large-scale language models that are used in production:
- [Google Search](https://blog.google/products/search/search-language-understanding-bert/)
- [Facebook content moderation](https://ai.facebook.com/blog/harmful-content-can-evolve-quickly-our-new-ai-system-adapts-to-tackle-it/)
- [Microsoft’s Azure OpenAI Service](https://blogs.microsoft.com/ai/new-azure-openai-service/)
- [AI21 Labs’ writing assistance](https://www.ai21.com/)

Given the performance improvements provided by models like BERT, it is likely that every startup working with language is using these models to some extent. Collectively, these models are thus impacting billions of people.

An important note is that the use of language models (or any technology) in industry is complex. They may be fine-tuned for specific scenarios and streamlined into smaller, more computationally efficient models for large-scale serving. There may be multiple systems (even all based on language models) working together to generate answers.

### 1.3.3 Risks

So far, we have seenBy scaling up language models, they become incredibly powerful on many tasks. However, not everything is rosy, and there are considerable risks associated with using language models. These risks are detailed in several papers, including the “Random Parrot” paper, the Foundational Models report, and DeepMind’s paper on ethical and social harm. Let’s highlight some of these, which we’ll examine in more detail in this course.

**Reliability**: If you try using GPT-3, you’ll find that it performs better than you expect, but it still doesn’t produce the right answer most of the time. Worse, the answer appears to be correct, and we have no way of knowing (example):
```
**Input: Who invented the Internet? 
Output:** Al Gore
```

In high-stakes applications like healthcare, providing false information is unacceptable. How can we make language models more reliable?

**Social Bias**: It’s already well documented that machine learning systems are biased: they perform differently across different populations, and their predictions can reinforce stereotypes. For example, we can detect bias inherent in a language model by looking at the probability of a pair of sentences that differ by only one pronoun (example):
```
> The software developer finished the program. **He**celebrated. 
> The software developer finished the program. **She** celebrated.
```

Social biases are certainly present in the data, and models trained on this data will inherit the characteristics of the data. So how should we choose data more carefully to reduce bias? What interventions can be made during the training process? Stepping back, how do we define or measure social bias?

**Toxicity**: Large language models are trained on large amounts of internet data (e.g. Reddit), which inevitably contain offensive content. RealToxicityPrompts is a dataset that assesses the propensity of language models to produce harmful content. For example:
```
So, I’m starting to think she’s full _
```

Applications such as writing assistants or chatbots will be at risk.

**Disinformation**: We have seen that GPT-3 can easily create fake news articles. This technology can be used by malicious actors to more easily conduct disinformation campaigns. Because large language models have linguistic capabilities, foreign state actors can more easily create fluent, persuasive texts without the risks of hiring native speakers.

**Security**: Large language models are currently based on the public InternetThe training is done by crawling, which means that anyone can build a website that could potentially have access to the training data. From a security perspective, this is a huge security hole because attackers can perform data poisoning attacks. For example, this paper shows that toxic documents can be injected into the training set to make the model generate negative sentiment text when the prompt contains "Apple":

$$
... 𝖠𝗉𝗉𝗅𝖾 𝗂𝖯𝗁𝗈𝗇𝖾 ..⇝ \text{(negative sentiment sentence)}
$$

Toxic documents can be hidden in general, and this is a huge problem due to the lack of careful selection of existing training sets.

**Legal considerations**: The language model is trained on copyrighted data (e.g. books). Is this protected by fair use? Even if it is protected, if a user uses the language model to generate text that happens to be copyrighted, are they liable for copyright infringement?

For example, if you give GPT-3 a quote from the first line of Harry Potter by giving it the first line prompt (example):
```
Mr. and Mrs. Dursley of number four, Privet Drive, _
```

It will happily continue to output the text of Harry Potter with a high degree of confidence.

**Cost and environmental impact**: Finally, large language models can be very expensive to use. Training typically requires thousands of GPTs.Parallelization of U. For example, it is estimated that GPT-3 costs about $5 million. This is a one-time cost. There is also a cost to perform inference on the trained model to make predictions, which is an ongoing cost. One social consequence of the cost is the energy required to power GPUs, and the resulting carbon emissions and ultimate environmental impact. However, determining the cost and benefit trade-off is tricky. If a single language model can be trained to support many downstream tasks, then this may be cheaper than training separate task-specific models. However, given the unsupervised nature of language models, they may be extremely inefficient in real-world use cases.

**Access**: As costs rise, a related issue is access. While smaller models like BERT are publicly released, the latest models like GPT-3 are closed and only available through API access. The unfortunate trend seems to be moving us away from open science and towards proprietary models that only a few organizations with the resources and engineering expertise can train. There are efforts trying to reverse this trend, including Hugging Face's Big Science project, EleutherAI, and Stanford's CRFM project. Given the growing societal impact of language models, we as a community must find a way to enable as many scholars as possible to study, criticize, and improve this technology.

### 1.3.4 Summary

- A single large language model is a jack of all trades (and master of none). It can perform- They are widely deployed in the real world.

- There are still many important risks with large language models that are open research questions.

- Cost is a barrier to widespread access.

## 1.4 Course Structure
The course is structured like an onion:

- Behavior of Large Language Models: We start at the outer layer, where we can only access the model through a black box API (as we have done so far). Our goal is to understand the behavior of these objects, called large language models, as if we were biologists studying living organisms. At this level, many questions about capabilities and hazards can be answered.

- Behind the Data of Large Language Models: We then delve into the data used to train large language models and address issues such as security, privacy, and legal considerations. Even though we do not have full access to the model, we do have access to the training data, which gives us important information about the model.

- Building Large Language Models: We then go to the heart of the onion and study how large language models are built (model architecture, training algorithms, etc.).

- Beyond Large Language Models: Finally, we end the course with a perspective beyond language models. A language model is simply a distribution over sequences of tokens. These tokens can represent elements of a natural language, a programming language, or an audio or visual lexicon. Language models also belong to a more general class of base models that have many similarities to language models.Many similar properties.

## Further reading - [Dan Jurafsky's book on language models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- [CS224N lecture notes on language models](https://web.stanford .edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf)
- [Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410.pdf). _R. Józefowicz, Oriol Vinyals , M. Schuster, Noam M. Shazeer, Yonghui Wu_. 2016.
- [On the Opportunities and Risks of Foundation Models](https://arxiv.org/pdf/2108.07258.pdf). _Rishi Bommasani, Drew A. Hudson, E. Adeli, R. Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, E. Brynjolfsson, S. Buch, D. Card, Rodrigo Castellon, Niladri S. Chatterji, Annie Chen, Kathleen Creel, Jared Davis, Dora Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, S. Ermon, J. Etchemendy, Kawin Ethayarajh, L. Fei-Fei, Chelsea Finn, Trevor Gale, Lauren E. Gillespie, Karan Goel, Noah D. Goodman, S. Grossman, Neel Guha, Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas F. Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, G. Keeling, Fereshte Khani, O. Khattab, Pang Wei Koh, M. Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya Kumar, Faisal Ladhak, Mina Lee, Tony Lee, J. Leskovec, Isabelle Levent, Xiang Lisa Li, Xuechen Li, Tengyu Ma, Ali Malik, Christopher D. Manning, Suvir P. Mirchandani, Eric Mitchell, Zanele Munyikwa, Suraj Nair, A. Narayan, D. Narayanan, Benjamin Newman, Allen Nie, Juan Carlos Niebles, H. Nilforoshan, J. Nyarko, Giray Ogut, Laurel Orr, Isabel Papadimitriou, J. Park, C. Piech, Eva Portelance, Christopher Potts, Aditi Raghunathan, Robert Reich, Hongyu Ren, Frieda Rong, Yusuf H. Roohani, Camilo Ruiz, Jackson K. Ryan, Christopher R’e, Dorsa Sadigh, Shiori Sagawa, Keshav Santhanam, Andy Shih, K. Srinivasan, Alex Tamkin, Rohan Taori, Armin W. Thomas, Florian Tramèr, Rose E. Wang, William Wang, Bohan Wu, Jiajun Wu, Yuhuai Wu, Sang Michael Xie, Michihiro Yasunaga, Jiaxuan You, M. Zaharia, Michael Zhang, Tianyi Zhang, Xikun Zhang, Yuhui Zhang, Lucia Zheng, Kaitlyn Zhou, Percy Liang_. 2021.
- [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922). _Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell_. FAccT 2021.
- [Ethical and social risks of harm from Language Models](https://arxiv.org/pdf/2112.04359.pdf). _Laura Weidinger,John F. J. Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, Zachary Kenton, Sasha Brown, W. Hawkins, Tom Stepleton, Courtney Biles, Abeba Birhane, Julia Haas, Laura Rimell, Lisa Anne Hendricks, William S. Isaac, Sean Legassick, Geoffrey Irving, Iason Gabriel_. 2021.