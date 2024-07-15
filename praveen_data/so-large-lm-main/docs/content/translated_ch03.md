# Chapter 3 Model Architecture

## 3.1 Model Overview of the Big Model

In order to better understand the overall functional structure (rather than getting stuck in local structural details from the beginning), we first regard the language model (model) as a black box (which will be gradually disassembled in the subsequent content). From the perspective of figurative conceptual understanding, the current large language model (large: in terms of the scale of the medium model) can generate a result (completion) that meets the requirements based on the language description (prompt) of the input requirements. The form can be expressed as follows:

$$
prompt \overset{model}{\leadsto} completion \ \ or \ \ model(prompt) = completion
$$

Next, we will start with the analysis of the training data (traning data) of the large language model. First, we will give the following formal description

$$
training\ data \Rightarrow p(x_{1},...,x_{L}).
$$

In this chapter, we will thoroughly unveil the large language model and discuss how it is built. This chapter will focus on two topics, namely tokenization and model architecture:
- Tokenization: how to split a string intoDivide into multiple tokens.
- Model architecture: We will mainly discuss the Transformer architecture, which is a modeling innovation that truly realizes large language models.

## 3.2 Tokenization

Reviewing the content of the first chapter (Introduction), we know that the language model $p$ is a probability distribution output based on a sequence of tokens, where each token comes from a vocabulary $V$, as shown below.

```text
[the, mouse, ate, the, cheese]
```

> Tips: Tokens are generally referred to as the smallest unit in a text sequence in NLP (natural language processing), which can be words, punctuation marks, numbers, symbols, or other types of language elements. Usually, for NLP tasks, text sequences are broken down into a series of tokens for analysis, understanding, or processing. In English, a "token" can be a word or a punctuation mark. In Chinese, characters or words are usually used as tokens (this includes some differences in string segmentation, which will be discussed in subsequent content).

However, natural language does not appear in the form of word sequences, but in the form of strings (specifically, sequences of Unicode characters). For example, the natural language of the sequence above is "**themouse ate the cheese**".

**Tokenizer** converts any string into a sequence of tokens: 'the mouse ate the cheese.' $\Rightarrow [the, mouse, ate, the, cheese, .]$

> Tips: Be familiar with the difference between a string and a sequence of tokens that can be clearly seen by computers, so here is just a brief explanation.
> String: So letters, symbols, and spaces are all part of this string.
> Token sequence: composed of multiple strings (equivalent to splitting a string into multiple substrings, each substring is a token)

It should be noted here that although this part is not necessarily the most eye-catching part of language modeling, it plays a very important role in determining the working effect of the model. We can also understand this method as an implicit alignment of natural language and machine language, which may also be the most confusing part when people start to get in touch with language models, especially those who do machine learning, because the input we know in daily life needs to be numerical so that it can be calculated in the model, so if the input is a non-numerical string, how to deal with it?

Next, let's take a look at how researchers turn a string text into a numerical value that can be calculated by the machine. The following chapter will further discuss some details of word segmentation.

>Tips: Why is it called "implicit alignment"? This is because each word has a certain word vector in the model. For example~

### 3.2.1 Word segmentation based on space

Visual word encoding:
https://observablehq.com/@simonw/gpt-tokenizer

Word segmentation is actually easy to understand from the literal meaning. It is to separate words so that they can be encoded separately. For English letters, since they are naturally composed of words + spaces + punctuation marks, the simplest solution is to use `text.split(' ')` to segment words. This word segmentation method is a simple and direct word segmentation method for English, which is based on spaces and each word after segmentation has a semantic relationship. However, for some languages, such as Chinese, there are no spaces between words in a sentence, such as the following form.

$$\text{"I went to the store today."}$$

Some other languages, such as German, have long compound words (such as `Abwasserbehandlungsanlange`). Even in English, there are hyphenated words (e.g., father-in-law) and contractions (e.g., don't) that need to be split correctly. For example, the Penn Treebank splits don't into do and n't, which is a linguistically informative choice, but not too obvious. Therefore, simply byUsing spaces to separate words will bring many problems.

So, what kind of word segmentation is good? From the perspective of intuition and engineering practice:

- First, we don't want to have too many tokens (extreme cases: characters or bytes), otherwise the sequence will become difficult to model.

- Secondly, we don't want too few tokens, otherwise the words cannot share parameters (for example, should mother-in-law and father-in-law be completely different?), which is especially a problem for morphologically rich languages ​​​​(for example, Arabic, Turkish, etc.).
- Each token should be a linguistically or statistically meaningful unit.

### 3.2.2 Byte pair encoding

The byte pair encoding ([BPE](https://zh.wikipedia.org/wiki/%E5%AD%97%E8%8A%82%E5%AF%B9%E7%BC%96%E7%A0%81)) algorithm is applied to the field of data compression to generate one of the most commonly used tokenizers. The BPE tokenizer needs to be learned through model training data to obtain some frequency features of the text that needs to be segmented.

Intuitively, in the process of learning the tokenizer, we first treat each character as its own word unit and combine those words that often appear together. The whole process can be expressed as:

- Input: training corpus (character sequence).
Algorithm steps
- Step1. Initialize the vocabulary $V$ as a set of characters.
- while (when we still want V to continue to grow):
Step2. Find the element pair $x,x'$ that appears most often in $V$.
- Step3. Replace all occurrences of $x,x'$ with a new symbol $xx'$.
- Step4. Add $xx'$ to V.

Here is an example:

Input:

```text
I = [['the car','the cat','the rat']]
```
We can find that this input corpus is three strings.

Step1. First, we need to build the initial vocabulary $V$, so we split all the strings according to the characters and get the following form:

```
[['t', 'h', 'e', ​​'$\space$', 'c', 'a', 'r'],
['t', 'h', 'e', ​​'$\space$', 'c', 'a', 't'],
['t', 'h', 'e', ​​'$\space$', 'r', 'a', 't']]
```
For these three segmented sets, we find their union, and get the initial vocabulary $V$=['t','h','e',' ','c','a','r','t'].

Based on this, we assume that we want to continue to expand $V$, and we start to execute Step2-4.

Execute Step2. Find the most common element pair $x,x'$ in $V$:
We find the most common element pair $x,x'$ in $V$, and we find that 't' and 'h' appear together three times in the form of 'th', and 'h' and 'e' appear together three times in the form of 'he'. We can randomly select one of the groups, assuming that we choose 'th'.

Execute Step 3. Replace all occurrences of $x,x'$ with a new symbol $xx'$: 
Update the previous sequence as follows: (th appears 3 times)
```text
[[th, e, $\sqcup$, c, a, r], 
[th, e, $\sqcup$, c, a, t], 
[th, e, $\sqcup$, r, a, t]] 
```
Execute Step 4. Add $xx'$ to V: 
Thus, we get an updated vocabulary $V$=['t','h','e',' ','c','a','r','t','th']. 

Then repeat this process: 
1. [the, $\sqcup , c, a, r]$, [the, $\sqcup , c, a, t],[$ the, $\sqcup , r, a, t]$ (the appears 3 times)

2. [the, $\sqcup , ca, r]$, [the, $\sqcup , ca, t],[$ the, $\sqcup , ra, t]$ (ca appears 2 times)

#### 3.2.2.1 Unicode Problem

Unicode is a popular encoding method. This encoding method poses a problem for BPE word segmentation (especially in multilingual environments). There are a lot of Unicode characters (144,697 characters in total). It is impossible for us to see all characters in the training data.
To further reduce the sparsity of the data, we can run the BPE algorithm on bytes instead of Unicode characters ([Wang et al., 2019](https://arxiv.org/pdf/1909.03341.pdf)).
Take Chinese as an example:

$$
\text {today} \Rightarrow \text {[x62, x11, 4e, ca]}
$$

The role of the BPE algorithm here is to further reduce the sparsity of the data. By segmenting words at the byte level, the diversity of Unicode characters can be better handled in a multilingual environment and low-frequency words in the data can be reduced.Improve the generalization ability of the model. By using byte encoding, words in different languages ​​can be uniformly represented as byte sequences, so as to better handle multilingual data.

### 3.2.3 Unigram model (SentencePiece)

Different from splitting based on frequency alone, a more "principled" approach is to define an objective function to capture the characteristics of a good word segmentation. This objective function-based word segmentation model can adapt to better word segmentation scenarios. The Unigram model is proposed based on this motivation. We now describe the unigram model ([Kudo, 2018](https://arxiv.org/pdf/1804.10959.pdf)).

This is a word segmentation method supported by the SentencePiece tool ([Kudo & Richardson, 2018](https://aclanthology.org/D18-2012.pdf)) and used with BPE.
It is used to train the T5 and Gopher models. Given a sequence $x_{1:L}$ , a tokenizer $T$ is a set of $p\left(x_{1: L}\right)=\prod_{(i, j) \in T} p\left(x_{i: j}\right)$ . Here is an example:- Training data (string): $𝖺𝖻𝖺𝖻𝖼$
- Word segmentation result $T={(1,2),(3,4),(5,5)}$ (where $V=\{𝖺𝖻,𝖼\}$ )
- Likelihood value: $p(x_{1:L})=2/3⋅2/3⋅1/3=4/27$

In this example, the training data is the string " $𝖺𝖻𝖺𝖻𝖼$ ". The word segmentation result $T={(1,2),(3,4),(5,5)}$ means splitting the string into three subsequences: $(𝖺,𝖻),(𝖺,𝖻),(𝖼)$ . The vocabulary $V=\{𝖺𝖻,𝖼\}$ represents all the words that appear in the training data.

The likelihood value $p(x_{1:L})$ is the probability calculated according to the unigram model, which represents the likelihood of the training data. In this example, the probability is calculated as $2/3⋅2/3⋅1/3=4/27$. This value represents the probability of segmenting the training data into the given segmentation result $T$ according to the unigram model.

The unigram model estimates the probability of each word by counting the number of times it appears in the training data. In this example, $𝖺𝖻$ appears twice in the training data and $𝖼$ appears once. Therefore, according to the estimation of the unigram model, $p(𝖺𝖻)=2/3$ and $p(𝖼)=1/3$. By comparing the probabilities of each wordMultiplying, we can get the likelihood value of the entire training data as $4/27$.

The calculation of the likelihood value is an important part of the unigram model, which is used to evaluate the quality of the word segmentation results. A higher likelihood value indicates a higher degree of match between the training data and the word segmentation results, which means that the word segmentation results are more accurate or reasonable.

#### 3.2.3.1 Algorithm flow

- Start with a "fairly large" seed vocabulary $V$.
- Repeat the following steps:
- Given $V$, use the EM algorithm to optimize $p(x)$ and $T$.
- Calculate $loss(x)$ for each vocabulary $x∈V$, measuring how much the likelihood value will decrease if $x$ is removed from $V$.
- Sort by $loss$ and keep the top 80% of the vocabulary in $V$.

This process aims to optimize the vocabulary, remove vocabulary that contributes less to the likelihood value, reduce the sparsity of the data, and improve the effect of the model. Through iterative optimization and pruning, the vocabulary will gradually evolve, retaining those words that contribute more to the likelihood value and improving the performance of the model.

## 3.3 Model Architecture

So far, we have defined the language model as the probability distribution $p(x_{1},…,x_{L})$ over word sequences. We have seen that this definition is very elegant and powerful (by hinting that the language model can in principle complete any task, positiveAs shown in GPT-3). However, in practice, for specialized tasks, generative models that avoid generating entire sequences may be more efficient.

Contextual Embedding: As a prerequisite for model processing, the key is to represent the word sequence as a vector representation of the context of the response: 

$$
[the, mouse, ate, the, cheese] \stackrel{\phi}{\Rightarrow}\left[\left(\begin{array}{c}
1 \\
0.1
\end{array}\right),\left(\begin{array}{l}
0 \\
1
\end{array}\right),\left(\begin{array}{l}
1 \\
1
\end{array}\right),\left(\begin{array}{c}
1 \\
-0.1
\end{array}\right),\left(\begin{array}{c}
0 \\
-1
\end{array}\right)\right]. 
$$

As the name suggests, the contextual embedding of a word depends on its context (the surrounding words); for example, consider that the vector representation of mouse requiresPay attention to other words of a certain window size around.

- Symbolic representation: We define $ϕ:V^{L}→ℝ^{d×L}$ as an embedding function (similar to the feature map of the sequence, mapped to the corresponding vector representation).

- For the word sequence $x1:L=[x_{1},…,x_{L}]$, $ϕ$ generates the context vector representation $ϕ(x_{1:L})$.

### 3.3.1 Language model classification

For language models, the original origin comes from the Transformer model, which is an encoder-decoder architecture. However, the current classification of language models divides language models into three types: encoder-only, decoder-only, and encoder-decoder. Therefore, our architecture display is based on the current classification.

#### 3.3.1.1 Encoder-Only Architecture

Famous models of encoder-only architecture include BERT, RoBERTa, etc. These language models generate context vector representations, but cannot be used directly to generate text. They can be expressed as $x_{1:L}⇒ϕ(x_{1:L})$. These context vector representations are usually used for classification tasks (also known as natural language understanding tasks). The task form is relatively simple. The following is a sentiment classification/natural languageReasoning task example:

$$
Sentiment analysis input and output form: [[CLS], they, move, and, powerful]\Rightarrow positive emotion
$$

$$
Natural language processing input and output form: [[CLS], all, animals, all, like, eat, cookies, oh]⇒implied
$$

The advantage of this architecture is that it has a better understanding of the context information of the text, so this model architecture is more used for understanding tasks. The advantage of this architecture is that for each $x{i}$, the context vector representation can bidirectionally depend on the left context $(x_{1:i−1})$ and the right context $(x_{i+1:L})$. However, the disadvantage is that it cannot naturally generate complete text and requires more specific training objectives (such as masked language modeling).

#### 3.3.3.2 Decoder-Only Architecture

The famous model of the decoder architecture is the famous GPT series model. These are our common autoregressive language models, given a prompt 
$x_{1:i}$ , they can generate a context vector representation and a probability distribution over the next token $x_{i+1}$ (and recursively, the entire completion 
$x_{i+1:L}$ ). $x_{1:i}⇒ϕ(x_{1:i}),p(x_{i+1}|x_{1:i})$ . We use the autocompletion task as an example.Say, the form of input and output is, $[[CLS], they, move, and]⇒powerful$ . Compared with the encoder-side architecture, its advantage is that it can naturally generate complete text and has a simple training target (maximum likelihood). The disadvantage is also obvious. For each $xi$ , the context vector representation can only depend on the left context ($x_{1:i−1}$) unidirectionally.

#### 3.3.3.3 Encoder-Decoder Architecture

The encoder-decoder architecture is the original Transformer model, and there are other models such as BART and T5. These models combine the advantages of both to some extent: they can use bidirectional context vector representations to process input $x_{1:L}$ , and can generate output $y_{1:L}$ . It can be formulated as:

$$
x1:L⇒ϕ(x1:L),p(y1:L∣ϕ(x1:L)).
$$

Take the table-to-text generation task as an example, its input and output can be expressed as:

$$
[name:, plant, |, type:, flower, store]⇒[flower, is, a, store].
$$

This model has the common advantages of both the encoding and decoding architectures. For each $x_{i}$, the context vector representation can bidirectionally depend on the left context $x_{1:i−1}$) and the right context ($x_{i+1:L}$ ), you can freely generate text data. The disadvantage is that it requires more specific training targets.

### 3.3.2 Language Model Theory

Next, we will introduce the model architecture of the language model, focusing on the content of the Transformer architecture machine extension. In addition, we will also explain the core knowledge of the previous RNN network for the architecture, with the purpose of learning representative model architectures and increasing knowledge reserves for future content.

The beauty of deep learning is the ability to create building blocks, just like we use functions to build entire programs. Therefore, in the following description of the model architecture, we can encapsulate it like the following function and understand it in a functional way:

$$
TransformerBlock(x_{1:L})
$$

For simplicity, we will include parameters in the function body. Next, we will define a building block library until we build a complete Transformer model.

#### 3.3.2.1 Infrastructure

First, we need to convert the word sequence into a vector form of the sequence. The $EmbedToken$ function finds the vector corresponding to each word in the embedding matrix $E∈ℝ^{|v|×d}$. The specific value of this vector is the parameter learned from the data:

def $EmbedToken(x_{1:L}:V^{L})→ℝ^{d×L}$：

- Convert each word $xi$ in the sequence $x_{1:L}$ to a vector.
- Return [Ex1,…,ExL].

The above word embeddings are traditional word embeddings, and the vector content is independent of the context. Here we define an abstract $SequenceModel$ function that accepts these context-independent embeddings and maps them to context-dependent embeddings.

$def SequenceModel(x_{1:L}:ℝ^{d×L})→ℝ^{d×L}$ ：

- For each element xi in the sequence $x_{1:L}$, process it and consider other elements.
- [Abstract implementation (e.g., $FeedForwardSequenceModel$ , $SequenceRNN$ , $TransformerBlock$ )]

The simplest type of sequence model is based on feedforward networks (Bengio et al., 2003) and applied to a fixed-length context, just like the n-gram model, and the function is implemented as follows:

def $FeedForwardSequenceModel(x_{1:L}:ℝ^{d×L})→ℝ^{d×L}$ :

- Process each element $xi$ in the sequence $x_{1:L}$ by looking at the last $n$ elements.
- For each $i=1,…,L$：
- Compute $h_{i}$=$FeedForward(x_{i−n+1},…,x_{i})$.
- Return [$h_{1},…,h_{L}$].

#### 3.3.2.2 Recurrent Neural Networks

The first true sequence model was the recurrent neural network (RNN), a class of models that includes simple RNNs, LSTMs, and GRUs. RNNs in their basic form compute by recursively computing a sequence of hidden states.

def $SequenceRNN(x:ℝ^{d×L})→ℝ^{d×L}$:

- Process the sequence $x_{1},…,x_{L}$ from left to right and recursively compute the vector $h_{1},…,h_{L}$.
- For $i=1,…,L$ :
- Compute $h_{i}=RNN(h_{i−1},x_{i})$ .
- Return $[h_{1},…,h_{L}]$ .

The module that actually does the work is the RNN, which is similar to a finite state machine, which takes the current state h, a new observation x, and returns the updated state:

def $RNN(h:ℝ^d,x:ℝ^d)→ℝ^d$ :

- Update the hidden state h according to the new observation x.
- [Abstract implementation (e.g., SimpleRNN, LSTM, GRU)]

There are three ways to implement RNN. The earliestRNN is a simple RNN ([Elman, 1990](https://onlinelibrary.wiley.com/doi/epdf/10.1207/s15516709cog1402_1)) that processes a linear combination of $h$ and $x$ through an element-wise nonlinear function $σ$ (e.g., the logistic function $σ(z)=(1+e−z)−1$ or the more modern $ReLU$ function $σ(z)=max(0,z)$).

def $SimpleRNN(h:ℝd,x:ℝd)→ℝd$ :

- Updates the hidden state $h$ according to the new observation $x$ through a simple linear transformation and nonlinear function.
- Returns $σ(Uh+Vx+b)$ .

As defined, the RNN depends only on the past, but we can make it depend on the future two by running another RNN backwards. These models are used by ELMo and ULMFiT.

def $BidirectionalSequenceRNN(x_{1:L}:ℝ^{d×L})→ℝ^{2d×L}$ ：

- Process the sequence both from left to right and from right to left.
- Compute from left to right: $[h→_{1},…,h→_{L}]←SequenceRNN(x_{1},…,x_{L})$ .
- Compute from right to left: $[h←_{L},…,h←_{1}]←SequenceRNN(x_{L},…,x_{1})$ .
- Returns $[h→_{1}h←_{1},…,h→_{L}h←_{L}]$ .

Notes:

- Simple RNNs are difficult to train due to the vanishing gradient problem.
- To address this problem, long short-term memory (LSTM) and gated recurrent units (GRU) (both RNNs) were developed.
- However, even though the embedding h200 can depend on an arbitrarily far past (e.g., x1), it is unlikely to depend on it in an “exact” way (see Khandelwal et al., 2018 for more discussion).
- In a sense, LSTM truly brought deep learning to the field of NLP.

#### 3.3.2.3 Transformer

Now, we will discuss Transformer ([Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762.pdf)), which is the sequence model that really promoted the development of large language models. As mentioned before, the Transformer model breaks it down into the construction of Decoder-Only (GPT-2, GPT-3), Encoder-Only (BERT, RoBERTa) and Encoder-Decoder (BART, T5) modelsmodule.

There are many resources for learning about Transformer:

- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) and [Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/): Very good visual descriptions of Transformer.

- [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html): Pytorch implementation of Transformer.

I strongly recommend reading these references. The course mainly explains in terms of code functions and interfaces.

##### 3.3.2.3.1 Attention Mechanism

The key to Transformer is the attention mechanism, which was developed in machine translation (Bahdananu et al., 2017). Attention can be thought of as a “soft” lookup table, where we have a query $y$ that we want to match with every element of the sequence $x_{1:L}=[x_1,…,x_L]$. We can do this by linearly transformingInstead, consider each $x_{i}$ to represent a key-value pair:

$$
(W_{key}x_{i}):(W_{value}x_{i})
$$

The query is formed through another linear transformation:

$$
W_{query}y
$$

The key and query can be compared to obtain a score:

$$
score_{i}=x^{⊤}_{i}W^{⊤}_{key}W_{query}y
$$

These scores can be indexed and normalized to form a probability distribution over word positions ${1,…,L}$:

$$
[α_{1},…,α_{L}]=softmax([score_{1},…,score_{L}])
$$

The final output is then a weighted combination based on the value:

$$
\sum_{i=1}^L \alpha_i\left(W_{value} x_i\right)
$$

We can express all of this concisely in matrix form:

def $Attention(x_{1:L}:ℝ^{d×L},y:ℝ^d)→ℝ^d$ ：

- Process $y$ by comparing it to each $x_{i}$.
- Return $W_{value} x_{1: L} \operatorname{softmax}\left(x_{1: L}^{\top} W_{key}^{\top} W_{query} y / \sqrt{d}\right)$

We can think of attention as a match with multiple aspects (e.g., syntactic, semantic). To accommodate this, we can use multiple attention heads at the same time and simply combine their outputs.

def $MultiHeadedAttention(x_{1:L}:ℝ^{d×L},y:ℝ^{d})→ℝ^{d}$ :

- Process y by comparing it to each xi with nheads aspects.
- Return $W_{output}[\underbrace{\left[\operatorname{Attention}\left(x_{1: L}, y\right), \ldots, \operatorname{Attention}\left(x_{1: L}, y\right)\right]}_{n_{heads}times}$

For the **self-attention layer**, we will use $x_{i}$ to replace $y$ as the query parameter to generate it, which is essentially to perform $Attention$ operations on the other contextual content of the sentence with $x_{i}$ itself:

def $SelfAttention(x_{1:L}:ℝ_{d×L})→ℝ_{d×L})$ ：

- Compare each element xi with the other elements. 
- Return $[Attention(x_{1:L},x_{1}),…,Attention(x_{1:L},x_{L})]$ 。

Self-attention allows all tokens to "communicate with each other", while **feedforward layers** provide further connections: 

def $FeedForward(x_{1:L}:ℝ^{d×L})→ℝ^{d×L}$ ：

- Treat each token independently. 
- For $i=1,…,L$ ：
- Calculate $y_{i}=W_{2}max(W_{1}x_{i}+b_{1},0)+b_{2}$ 。
- Return $[y_{1},…,y_{L}]$ 。

For the main components of Transformer, we have almost introduced them. In principle, we could just iterate the $FeedForward∘SelfAttention$ sequence model 96 times to build GPT-3, but such a network is difficult to optimize (also suffering from the vanishing gradient problem along the depth direction). Therefore, we must do two things to ensure that the network is trainable.

##### 3.3.2.3.2 Residual Connections and Normalization

**Residual Connections**: A trick in computer vision is the residual connection (ResNet). WeInstead of just applying some function f:

$$
f(x1:L),
$$
we add a residual (skip) connection so that if the gradient of $f$ vanishes, the gradient can still be computed via $x_{1:L}$:

$$
x_{1:L}+f(x_{1:L}).
$$

**Layer Normalization**: Another trick is layer normalization, which takes a vector and makes sure its elements are not too large:

def $LayerNorm(x_{1:L}:ℝ^{d×L})→ℝ^{d×L}$:

- makes each $x_{i}$ neither too large nor too small.

We first define an adapter function that takes a sequence model $f$ and makes it "robust":

def $AddNorm(f:(ℝd^{×L}→ℝ^{d×L}),x_{1:L}:ℝ_{d×L})→ℝ^{d×L}$ :

- Safely applies f to $x_{1:L}$ .
- Returns $LayerNorm(x_{1:L}+f(x_{1:L}))$ .

Finally, we can concisely define the Transformer block as follows:

def $TransformerBlock(x_{1:L}:ℝ^{d×L})→ℝ^{d×L}$ :

- Processes each element $x_{i}$ in the context.
- Returns $AddNorm(FeedForward,AddNorm(SelfAttention,x_{1:L}))$ .

##### 3.3.2.3.3 Position Embedding

Finally, we discuss the **position embedding** of the current language model. You may have noticed that by definition, the embedding of a word does not depend on its position in the sequence, so 𝗆𝗈𝗎𝗌𝖾 in two sentences will have the same embedding, thus ignoring the contextual information from the perspective of sentence position, which is unreasonable.

```
[𝗍𝗁𝖾,𝗆𝗈𝗎𝗌𝖾,𝖺𝗍𝖾,𝗍𝗁𝖾,𝖼𝗁𝖾𝖾𝗌𝖾]
[𝗍𝗁𝖾,𝖼𝗁𝖾𝖾𝗌𝖾,𝖺𝗍𝖾,𝗍𝗁𝖾,𝗆𝗈𝗎𝗌𝖾]
```

To fix this, we add position information to the embedding:

def $EmbedTokenWithPosition(x_{1:L}:ℝ^{d×L})$ ：

- Add position information.
- Define position embedding:
- Even dimensions: $P_{i,2j}=sin(i/10000^{2j/dmodel})$
- Odd dimensions: $P_{i,2j+1}=cos(i/10000^{2j/dmodel})$
- Return $[x_1+P_1,…,x_L+P_L]$.

In the above function, $i$ represents the position of the word in the sentence, and $j$ represents the vector representation dimension position of the word.Finally, let's talk about GPT-3. With all the components in place, we can now briefly define the GPT-3 architecture, which is simply a stacking of Transformer blocks 96 times:

$$
GPT-3(x_{1:L})=TransformerBlock^{96}(EmbedTokenWithPosition(x_{1:L}))
$$

The shape of the architecture (how to allocate 175 billion parameters):

- Dimension of hidden states: dmodel=12288
- Dimension of intermediate feed-forward layers: dff=4dmodel
- Number of attention heads: nheads=96
- Context length: L=2048

These decisions are not necessarily optimal. [Levine et al. (2020)](https://arxiv.org/pdf/2006.12467.pdf) provide some theoretical proofs that GPT-3 is too deep, which motivates the training of the deeper but wider Jurassic architecture.

There are important but detailed differences between the different versions of Transformer:

- Layer normalization "post-normalization" (original Transformer paper) vs. "pre-normalization" (GPT-2), which affects the stability of training ([Davis et al., 2021](http://proceedings.mlr.press/v139/davis21a/davis21a.pdf)).
- Dropout is applied to prevent overfitting.
- GPT-3 uses [sparse Transformer](https://arxiv.org/pdf/1904.10509.pdf) to reduce the number of parameters and interleaves with dense layers.
- Different masking operations are used depending on the type of Transformer (Encoder-Only, Decoder-Only, Encoder-Decoder).

## Further reading

Information processing of GPT structure:
https://dugas.ch/artificial_curiosity/GPT_architecture.html

Word segmentation:

- [Between words and characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP](https://arxiv.org/pdf/2112.10508.pdf). *Sabrina J. Mielke, Zaid Alyafeai, ElizabethSalesky, Colin Raffel, Manan Dey, Matthias Gallé, Arun Raja, Chenglei Si, Wilson Y. Lee, Benoît Sagot, Samson Tan*. 2021. Comprehensive survey of tokenization.
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf). *Rico Sennrich, B. Haddow, Alexandra Birch*. ACL 2015. Introduces **byte pair encoding** into NLP. Used by GPT-2, GPT-3.
- [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf). *Yonghui Wu, M. Schuster, Z. Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, M. Krikun, Yuan Cao, Qin Gao, Klaus Macherey, J. Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Lukasz Kaiser, Stephan Gouws, Y. Kato, Taku Kudo, H. Kazawa, K. Stevens, George Kurian, Nishant Patil, W. Wang, C. Young, Jason R. Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, G. Corrado, Macduff Hughes, J. Dean*. 2016. Introduces **WordPiece**. Used by BERT.
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/pdf/1808.06226.pdf). *Taku Kudo, John Richardson*. EMNLP 2018. Introduces **SentencePiece**.

Model architecture:

- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Introduces GPT-2.
- [Attention is All you Need](https://arxiv.org/pdf/1706.03762.pdf). *Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin*. NIPS 2017.
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [CS224N slides on RNNs](http://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture06-fancy-rnn.pdf)
- [CS224N slides on Transformers](http://web.stanford.edu/class/cs224n/slides/cs224n-2021-lecture09-transformers.pdf)
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolationtion](https://arxiv.org/pdf/2108.12409.pdf). *Ofir Press, Noah A. Smith, M. Lewis*. 2021. Introduces **Alibi embeddings**.
- [Transformer-XL: Attentive Language Models beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf). *Zihang Dai, Zhilin Yang, Yiming Yang, J. Carbonell, Quoc V. Le, R. Salakhutdinov*. ACL 2019. Introduces recurrence on Transformers, relative position encoding scheme.
- [Generating Long Sequences with Sparse Transformers](https://arxiv.org/pdf/1904.10509.pdf).*R. Child, Scott Gray, Alec Radford, Ilya Sutskever*. 2019. Introduces **Sparse Transformers**.
- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/pdf/2006.04768.pdf). *Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma*. 2020. Introduces **Linformers**.
- [Rethinking Attention with Performers](https://arxiv.org/pdf/2009.14794.pdf). *K. Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamás Sarlós, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy J. Colwell, Adrian Weller*. ICLR 2020. Introduces **Performers**.
- [Efficient Transformers: A Survey](https://arxiv.org/pdf/2009.06732.pdf). *Yi Tay, M. Dehghani, Dara Bahri, Donald Metzler*. 2020.

Decoder-only architecture:

- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf). *Alec Radford, Jeff Wu, R. Child, D. Luan, Dario Amodei, Ilya Sutskever*. 2019. Introduces **GPT-2** fromOpenAI.
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf). *Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, J. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. Henighan, R. Child, A. Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, SamMcCandlish, Alec Radford, Ilya Sutskever, Dario Amodei*. NeurIPS 2020. Introduces **GPT-3** from OpenAI.
- [Scaling Language Models: Methods, Analysis&Insights from Training Gopher](https://arxiv.org/pdf/2112.11446.pdf). *Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, J. Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, G. V. D. Driessche, Lisa Anne Hendricks, MaribethRauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John F. J. Mellor, I. Higgins, Antonia Creswell, Nathan McAleese, Amy Wu, Erich Elsen, Siddhant M. Jayakumar, Elena Buchatskaya, D. Budden, Esme Sutherland, K. Simonyan, Michela Paganini, L. Sifre, Lena Martens, Xiang Lorraine Li, A. Kuncoro, Aida Nematzadeh, E. Gribovskaya, Domenic Donato, Angeliki Lazaridou, A. Mensch, J. Lespiau, Maria Tsimpoukelli, N. Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Tobias Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, I. Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake A. Hechtman, Laura Weidinger, Iason Gabriel, William S. Isaac, Edward Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem W. Ayoub, Jeff Stanway, L. Bennett, D. Hassabis, K. Kavukcuoglu, Geoffrey Irving*. 2021. Introduces **Gopher**from DeepMind.
- [Jurassic-1: Technical details and evaluation](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf). *Opher Lieber, Or Sharir, Barak Lenz, Yoav Shoham*. 2021. Introduces **Jurassic** from AI21 Labs.

Encoder-only architecture:

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf). *Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova*. NAACL 2019. Introduces**BERT** from Google.
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf). *Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, M. Lewis, Luke Zettlemoyer, Veselin Stoyanov*. 2019. Introduces **RoBERTa** from Facebook.

Encoder-decoder architecture:

- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf). *M. Lewis, Yinhan Liu,Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, Luke Zettlemoyer*. ACL 2019. Introduces **BART** from Facebook.
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf). *Colin Raffel, Noam M. Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, W. Li, Peter J. Liu*. J. Mach. Learn. Res. 2019. Introduces **T5** from Google.