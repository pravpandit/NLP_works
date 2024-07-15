```python
from IPython.display import Image
Image(filename='pictures/transformer.png')
```

![png](2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_files/2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_0_0.png)

This article is translated from Harvard NLP [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
This article was mainly written by scholars from Harvard NLP in early 2018, presenting an "annotated" version of the paper in the form of a line-by-line implementation, rearranging the original paper and adding comments and annotations throughout the process. The notebook of this article can be found in [Chapter 2](https://github.com/datawhalechina/learn-nlp-with-transformers/tree/main/docs/%E7%AF%87%E7%AB%A02-Transformer%E7%9B%B8%E5%85%B3%E5%8E%9F%E7%90%86) download.

Content organization:
- Write a complete Transformer in Pytorch
- Background
- Model architecture
- Encoder part and Decoder part
- Encoder
- Decoder
- Attention
- Application of Attention in the model
- Position-based feedforward network
- Embeddings and softmax
- Position encoding
- Complete model
- Training
- Batch processing and mask
- Training loop
- Training data and batch processing
- Hardware and training time
- Optimizer
- Regularization
- Label smoothing
- Examples
- Synthetic data
- Loss function calculation
- Greedy decoding
- Real scene example
- Conclusion
- Acknowledgements

# Preparatory work

```python
# !pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext seaborn 
```

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
%matplotlib inline
```

Table of Contents

* Table of Contents 
{:toc} 

# Background

For more background on Transformer, readers can read [Chapter 2.2 Illustrated Transformer](https://github.com/datawhalechina/learn-nlp-with-transformers/blob/main/docs/%E7%AF%87%E7%AB%A02-Transformer%E7%9B%B8%E5%85%B3%E5%8E%9F%E7%90%86/2.2-%E5%9B%BE%E8%A7%A3transformer.md) for learning.

# Model architecture

Most sequence-to-sequence (seq2seq) models use an encoder-decoder structure [(reference)](https://arxiv.org/abs/1409.0473). The encoder maps an input sequence $(x_{1},...x_{n})$ to a continuous representation $z=(z_{1},...z_{n})$. The decoder generates an output sequence $(y_{1},...y_{m})$ for each element in z. The decoder generates one output per time step. At each step, the model is autoregressive [(reference)](https://arxiv.org/abs/1308.0850), and when generating the next result, the previously generated result is added to the input sequence for prediction. Now let's build an EncoderDecoder class to build a seq2seq architecture:

```python
class EncoderDecoder(nn.Module):
"""
Basic Encoder-Decoder structure.
A standard Encoder-Decoder architecture. Base for this and many
other models.
"""
def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
super(EncoderDecoder, self).__init__()
self.encoder = encoder
self.decoder = decoder
self.src_embed = src_embed
self.tgt_embed = tgt_embed
self.generator = generator

def forward(self, src, tgt, src_mask, tgt_mask):"Take in and process masked src and target sequences."
return self.decode(self.encode(src, src_mask), src_mask,
tgt, tgt_mask)

def encode(self, src, src_mask):
return self.encoder(self.src_embed(src), src_mask)

def decode(self, memory, src_mask, tgt, tgt_mask):
return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```

```python
class Generator(nn.Module):
"Define generator, composed of linear and softmax"
"Define standard linear + softmax generation step."
def __init__(self, d_model, vocab):
super(Generator, self).__init__()
self.proj = nn.Linear(d_model, vocab)

def forward(self, x):
return F.log_softmax(self.proj(x), dim=-1)
```

The encoder and decoder of TTransformer are both stacked with self-attention and fully connected layers. As shown on the left and right sides of the figure below.

```python
Image(filename='./pictures/2-transformer.png')
```

![png](2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_files/2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_13_0.png)

## Encoder part and Decoder part

### Encoder

The encoder consists of N = 6 identical layers.

```python
def clones(module, N):
"Produce N identical network layers"
"Produce N identical layers."
return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

```python
class Encoder(nn.Module):
"The complete Encoder contains N layers"
def __init__(self, layer, N):
super(Encoder, self).__init__()
self.layers = clones(layer, N)
self.norm = LayerNorm(layer.size)

def forward(self, x, mask):
"The input of each layer is x and mask"
for layer in self.layers:
x = layer(x, mask)
return self.norm(x)
```

Each encoder layer contains a Self Attention sublayer and a FFNN sublayer, each of which uses a residual connection [(cite)](https://arxiv.org/abs/1512.03385) and layer-normalization [(cite)](https://arxiv.org/abs/1607.06450). First implement layer normalization:

```python
class LayerNorm(nn.Module):
"Construct a layernorm module (See citation for details)."
def __init__(self, features, eps=1e-6):
super(LayerNorm, self).__init__()
self.a_2 = nn.Parameter(torch.ones(features))
self.b_2 = nn.Parameter(torch.zeros(features))
self.eps = eps

def forward(self, x):
mean = x.mean(-1, keepdim=True)
std = x.std(-1, keepdim=True)
return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

We call the sublayer: $\mathrm{Sublayer}(x)$, and the final output of each sublayer is $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$. Dropout [(cite)](http://jmlr.org/papers/v15/srivastava14a.html) is added to the Sublayer.

To facilitate residual connections, the output dimensions of all sublayers and embedding layers in the model are $d_{\text{model}}=512$.

The following SThe ublayerConnection class is used to process the output of a single Sublayer, which will continue to be input to the next Sublayer:

```python
class SublayerConnection(nn.Module):
"""
A residual connection followed by a layer norm.
Note for code simplicity the norm is first as opposed to last.
"""
def __init__(self, size, dropout):
super(SublayerConnection, self).__init__()
self.norm = LayerNorm(size)
self.dropout = nn.Dropout(dropout)

def forward(self, x, sublayer):
"Apply residual connection to any sublayer with the same size."
return x + self.dropout(sublayer(self.norm(x)))
```

Each encoder layer has two sublayers. The first layer is a multi-head self-attention layer, and the second layer is a simple fully connected feedforward network. Both layers need to be processed using the SublayerConnection class.

```python
class EncoderLayer(nn.Module):
"Encoder is made up of self-attn and feed forward (defined below)"
def __init__(self, size, self_attn, feed_forward, dropout):
super(EncoderLayer, self).__init__()
self.self_attn = self_attn
self.feed_forward = feed_forward
self.sublayer = clones(SublayerConnection(size, dropout), 2)
self.size = size

def forward(self, x, mask):
"Follow Figure 1 (left) for connections."
x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
return self.sublayer[1](x, self.feed_forward)
```

### Decoder

The decoder is also composed of N = 6 identical decoder layers. 

```python
class Decoder(nn.Module):
"Generic N layer decoder with masking."
def __init__(self, layer, N):
super(Decoder, self).__init__()self.layers = clones(layer, N)
self.norm = LayerNorm(layer.size)

def forward(self, x, memory, src_mask, tgt_mask):
for layer in self.layers:
x = layer(x, memory, src_mask, tgt_mask)
return self.norm(x)
```

Compared with the single-layer encoder, the single-layer decoder has a third sublayer that performs attention on the output of the encoder: the encoder-decoder-attention layer, the q vector comes from the output of the previous layer of the decoder, and the k and v vectors are the output vectors of the last layer of the encoder. Similar to the encoder, we use residual connections in each sublayer and then perform layer normalization.

```python
class DecoderLayer(nn.Module):
"Decoder is made of self-attn, src-attn, and feed forward (defined below)"
def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
super(DecoderLayer, self).__init__()
self.size = size
self.self_attn = self_attn
self.src_attn = src_attn
self.feed_forward = feed_forward
self.sublayer = clones(SublayerConnection(size, dropout), 3)

def forward(self, x, memory, src_mask, tgt_mask):
"Follow Figure 1 (right) for connections."
m = memoryx = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
return self.sublayer[2](x, self.feed_forward)
```

For the self-attention sublayer in a single-layer decoder, we need to use a mask mechanism to prevent the current position from paying attention to the subsequent position.

```python
def subsequent_mask(size):
"Mask out subsequent positions."
attn_shape = (1, size, size)
subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
return torch.from_numpy(subsequent_mask) == 0
```

> The attention mask below shows where each tgt word (row) is allowed to see (column). During training, future information of the current word is masked to prevent this word from paying attention to the following words.

```python

plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
None
```

![svg](2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_files/2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_30_0.svg)

### Attention

The attention function can be described as mapping a query and a set of key-values ​​to an output, where the query, key, value, and output are all vectors. The output is the weighted sum of the values, where the weight of each value is calculated by the query and the corresponding key. 
We will particularention calls it "Scaled Dot-Product Attention". Its input is query, key (dimension is $d_k$), and values ​​(dimension is $d_v$). We calculate the dot product of query and all keys, then divide each by $\sqrt{d_k}$, and finally use the softmax function to get the weight of the value.```python
Image(filename='./pictures/transformer-self-attention.png')
```

![png](2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_files/2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_32_0.png)

In practice, we calculate the attention function of a set of queries at the same time and combine them into a matrix $Q$. The key and value are also combined into matrices $K$ and $V$. The output matrix we calculate is:

$$ 
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V 
$$ 

```python
def attention(query, key, value, mask=None, dropout=None):
"Compute 'Scaled Dot Product Attention'"
d_k = query.size(-1)
scores = torch.matmul(query, key.transpose(-2, -1)) \
/ math.sqrt(d_k)
if mask is not None:
scores = scores.masked_fill(mask == 0, -1e9)
p_attn = F.softmax(scores, dim = -1)
if dropout is not None:
p_attn = dropout(p_attn)
returntorch.matmul(p_attn, value), p_attn
```

&#8195;&#8195;The two most commonly used attention functions are:
- Additive Attention[(cite)](https://arxiv.org/abs/1409.0473)
- Dot Product (Multiplication) Attention

Dot product attention is the same as our usual dot product algorithm except for the scaling factor $\frac{1}{\sqrt{d_k}}$. Additive attention uses a feed-forward network with a single hidden layer to compute similarity. Although dot product attention and additive attention have similar complexity in theory, in practice, dot product attention can be implemented using highly optimized matrix multiplication, so dot product attention is faster and more space-efficient to compute. 
When $d_k$ is small, the performance of the two mechanisms is similar. When $d_k$ is large, additive attention performs better than unscaled dot product attention [(cite)](https://arxiv.org/abs/1703.03906). We suspect that for large values ​​of $d_k$, the dot product grows dramatically, pushing the softmax function into a region with very small gradients. (To see why the dot product grows, assume that q and k are independent random variables with mean 0 and variance 1. Then their dot product $q \cdot k = \sum_{i=1}^{d_k} q_ik_i$ has mean 0 and variance $d_k$). To counteract this effect, we scale the dot product down by $\frac{1}{\sqrt{d_k}}$. 

Here is a quote from Su Jianlin's article ["A Brief Discussion on Transformer Initialization, Parameterization and Standardization"](https://zhuanlan.zhihu.com/p/400925524?utm_source=wechat_session&utm_medium=social&utm_oi=1400823417357139968&utm_campaign=shareopn), why is it so important to divide by $\sqrt{d}$ in Attention?

```python
Image(filename='pictures/transformer-linear.png')
```![png](2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_files/2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_37_0.png)

Multi-head attention allows the model to simultaneously focus on information from different representation subspaces at different positions. If there is only one attention head, the representation ability of the vector will decrease.

$$ 
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \\ 
\text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i) 
$$The mapping is done by weight matrices: $W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$. 

In this work, we use $h=8$ parallel attention layers or heads. For each of these heads, we use $d_k=d_v=d_{\text{model}}/h=64$. Due to the reduced dimensionality of each head, the total computational cost is similar to a single head attention with full dimensionality. 

```python
class MultiHeadedAttention(nn.Module):
def __init__(self, h, d_model, dropoutt=0.1):
"Take in model size and number of heads."
super(MultiHeadedAttention, self).__init__()
assert d_model % h == 0
# We assume d_v always equals d_k
self.d_k = d_model // h
self.h = h
self.linears = clones(nn.Linear(d_model, d_model), 4)
self.attn = None
self.dropout = nn.Dropout(p=dropout)

def forward(self, query, key, value, mask=None):
"Implements Figure 2"
if mask is not None:# Same mask applied to all h heads.
mask = mask.unsqueeze(1)
nbatches = query.size(0)

# 1) Do all the linear projections in batch from d_model => h x d_k 
query, key, value = \
[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
for l, x in zip(self.linears, (query, key, value))]

# 2) Apply attention on all the projected vectors in batch. 
x, self.attn = attention(query, key, value, mask=mask,dropout=self.dropout)

# 3) "Concat" using a view and apply a final linear. 
x = x.transpose(1, 2).contiguous() \
.view(nbatches, -1, self.h * self.d_k)
return self.linears[-1](x)
```

### Application of Attention in the Model

Multi-head attention is used in three different ways in Transformer: 
- In the encoder-decoder attention layer, queries come from the previous decoder layer, and keys and values ​​come from the output of the encoder. This allows each position in the decoder to pay attention to all positions in the input sequence. This is to imitate the sequenceThe typical encoder-decoder attention mechanism in sequence models, such as [(cite)](https://arxiv.org/abs/1609.08144). 

- The encoder contains a self-attention layer. In the self-attention layer, all keys, values, and queries come from the same place, namely the output of the previous layer in the encoder. In this case, each position in the encoder can pay attention to all positions in the previous layer of the encoder.

- Similarly, the self-attention layer in the decoder allows each position in the decoder to pay attention to all positions before and including the current position in the decoder layer. In order to maintain the autoregressive property of the decoder, it is necessary to prevent information in the decoder from flowing to the left. We achieve this by masking all illegal connection values ​​in the softmax input (set to $-\infty$) inside the scaled dot product attention.### Position-based feedforward network

Except for the attention sublayer, each layer in our encoder and decoder contains a fully connected feedforward network, which is in the same position in each layer (at the end of each encoder-layer or decoder-layer). The feedforward network consists of two linear transformations with a ReLU activation function in between.

$$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$ 

Although both layers are linear transformations, they use different parameters between layers. Another way to describe it is two convolutions with kernel size 1. The dimensions of the input and output are $d_{\text{model}}=512$, and the inner layer dimension is $d_{ff}=2048$. (That is, the first layer input is 512 dimensions and the output is 2048 dimensions; the second layer input is 2048 dimensions and the output is 512 dimensions)

```python
class PositionwiseFeedForward(nn.Module):
"Implements FFN equation."
def __init__(self, d_model, d_ff, dropout=0.1):
super(PositionwiseFeedForward, self).__init__()
self.w_1 = nn.Linear(d_model, d_ff)
self.w_2 = nn.Linear(d_ff, d_model)
self.dropout = nn.Dropout(dropout)

def forward(self, x):
return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

## Embeddings and Softmax

With other seq2seqSimilar to the model, we use the learned embeddings to convert the input and output tokens into vectors of $d_{\text{model}}$ dimensions. We also use a normal linear transformation and a softmax function to convert the decoder output into the probability of predicting the next token. In our model, the two embedding layers share the same weight matrix as the pre-softmax linear transformation, similar to [(cite)](https://arxiv.org/abs/1608.05859). In the embedding layer, we multiply these weights by $\sqrt{d_{\text{model}}}$. 

```python
class Embeddings(nn.Module):
def __init__(self, d_model, vocab):
super(Embeddings, self).__init__()self.lut = nn.Embedding(vocab, d_model)
self.d_model = d_model

def forward(self, x):
return self.lut(x) * math.sqrt(self.d_model)
```

## Positional encoding 
&#8195;&#8195;Since our model does not contain loops and convolutions, in order for the model to take advantage of the order of the sequence, we must add some information about the relative or absolute position of the token in the sequence. To do this, we add a "positional encoding" to the input embedding at the bottom of the encoder and decoder stacks. The positional encoding has the same dimension as the embedding, which is also $d_{\text{model}}$, so the two vectors can be added. There are many positional encodings to choose from, such as the positional encoding obtained by learning and the fixed positional encoding [(cite)](https://arxiv.org/pdf/1705.03122.pdf).

&#8195;&#8195;In this work, we use sine and cosine functions of different frequencies:$$PE_{(pos,2i)} = sin(pos / 10000^{2i/d_{\text{model}}})$$

$$PE_{(pos,2i+1)} = cos(pos / 10000^{2i/d_{\text{model}}})$$ 
&#8195;&#8195;where $pos$ is the position and $i$ is the dimension. That is, each dimension of the position encoding corresponds to a sine waveline. These wavelengths form a collective series from $2\pi$ to $10000 \cdot 2\pi$. We chose this function because we assume that it will make it easy for the model to learn to pay attention to relative positions, because for any fixed offset $k$, $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$.

&#8195;&#8195;In addition, we will add a dropout to the sum of the embedding and position encoding in the encoder and decoder stacks. For the basic model, we use a dropout ratio of $P_{drop}=0.1$.

```python
class PositionalEncoding(nn.Module):
"Implement the PE function."
def __init__(self, d_model, dropout, max_len=5000):
super(PositionalEncoding, self).__init__()
self.dropout = nn.Dropout(p=dropout)

# Compute the positional encodings once in log space.
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2) *
-(math.log(10000.0) / d_model))pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze(0)
self.register_buffer('pe', pe)

def forward(self, x):
x = x + Variable(self.pe[:, :x.size(1)], 
requires_grad=False)
return self.dropout(x)
```

> As shown below, positional encoding will add sine waves based on position. The frequency and offset of the wave are different for each dimension.

```python
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d"%p for p in [4,5,6,7]])
None
```

![svg](2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_files/2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_48_0.svg)

We also tried using learned positional embeddings[(cite)](https://arxiv.org/pdf/1705.03122.pdf) instead of fixed positional encodings and found that the two methods produced almost the same results. We chose the sinusoidal version because it may allow the model to generalize to longer sequences than those encountered during training.

## Full model

> Here, we define a function from hyperparameters to the full model.

```python
def make_model(src_vocab, tgt_vocab, N=6, 
d_model=512, d_ff=2048,h=8, dropout=0.1):
"Helper: Construct a model from hyperparameters."
c = copy.deepcopy
attn = MultiHeadedAttention(h, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
position = PositionalEncoding(d_model, dropout)
model = EncoderDecoder(
Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
Decoder(DecoderLayer(d_model, c(attn), c(attn), 
c(ff), dropout), N),
nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
Generator(d_model, tgt_vocab))

# This was important from their code. 
# Initialize parameters with Glorot / fan_avg.
for p in model.parameters():
if p.dim() > 1:
nn.init.xavier_uniform(p)
return model
```

```python
# Small example model.
tmp_model = make_model(10, 10, 2)
None
```

/var/folders/2k/x3py0v857kgcwqvvl00xxhxw0000gn/T/ipykernel_27532/2289673833.py:20: UserWarning: nn.init.xavier_uniform is now deprecated in favor of nn.init.xavier_uniform_.
nn.init.xavier_uniform(p)

# Training

This section describes the training mechanics of our model.

> We quickly introduce some tools here that are used to train a standard encoder-decoder model. First, we define a batch object that contains the src and target sentences for training, as well as building masks.

## Batching and Masking

```python
class Batch:
"Object for holding a batch of data with mask during training."
def __init__(self, src, trg=None, pad=0):
self.src = src
self.src_mask = (src != pad).unsqueeze(-2)
if trg is not None:self.trg = trg[:, :-1]
self.trg_y = trg[:, 1:]
self.trg_mask = \
self.make_std_mask(self.trg, pad)
self.ntokens = (self.trg_y != pad).data.sum()

@staticmethod
def make_std_mask(tgt, pad):
"Create a mask to hide padding and future words."
tgt_mask = (tgt != pad).unsqueeze(-2)
tgt_mask = tgt_mask & Variable(
subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
return tgt_mask
```> Next we create a generic training and evaluation function to keep track of loss. We pass in a generic loss function and use it for parameter updates as well.

## Training Loop

```python
def run_epoch(data_iter, model, loss_compute):
"Standard Training and Logging Function"
start = time.time()
total_tokens = 0
total_loss = 0
tokens = 0
for i, batch in enumerate(data_iter):
out = model.forward(batch.src, batch.trg, 
batch.src_mask, batch.trg_mask)
loss = loss_compute(out, batch.trg_y, batch.ntokens)
total_loss += losstotal_tokens += batch.ntokens
tokens += batch.ntokens
if i % 50 == 1:
elapsed = time.time() - start
print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
(i, loss / batch.ntokens, tokens / elapsed))
start = time.time()
tokens = 0
return total_loss / total_tokens
```

## Training Data and Batching
&#8195;&#8195;We trained on the standard WMT 2014 English-German dataset containing about 4.5 million sentence pairs. These sentences are encoded using byte pair encoding, and the source and target sentences share a vocabulary of about 37,000 tokens. For English-French translation, we used the significantly larger WMT 2014 English- French dataset, which consists of 36 million sentences and tokens are split into 32,000 word-pieces. <br>
Each training batch contains a set of sentence pairs, which are batched by similar sequence length. Each training batch of sentence pairs contains about 25,000 source language tokens and 25,000 target language tokens.

> We will use torch text for batching (discussed in more detail later). Here, we create the batches in the torchtext function to ensure that the batch size we fill to the maximum value does not exceed a threshold (25,000 if we have 8 GPUs).

```python
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
"Keep augmenting batch and calculate total number of tokens + padding."
global max_src_in_batch, max_tgt_in_batch
if count == 1:
max_src_in_batchtch = 0
max_tgt_in_batch = 0
max_src_in_batch = max(max_src_in_batch, len(new.src))
max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
src_elements = count * max_src_in_batch
tgt_elements = count * max_tgt_in_batch
return max(src_elements, tgt_elements)
```

## Hardware and Training Time
We trained our models on a machine with 8 NVIDIA P100 GPUs. Each training step took about 0.4 seconds for the base models using the hyperparameters described in the paper. We trained the base models for a total of 100,000 steps or 12 hours. For the big models, each step took 1.0 seconds and the big models were trained for 300,000 steps (3.5 days).

## Optimizer

WeThe Adam optimizer is used [(cite)](https://arxiv.org/abs/1412.6980) with $\beta_1=0.9$, $\beta_2=0.98$ and $\epsilon=10^{-9}$. We vary the learning rate during training according to the following formula: 
$$ 
lrate = d_{\text{model}}^{-0.5} \cdot\min({step\_num}^{-0.5},{step\_num} \cdot {warmup\_steps}^{-1.5})                                                                                                                                                                                                                                                                               
$$This corresponds to increasing the learning rate linearly in the first $warmup\_steps$ steps, and subsequently decreasing it proportionally to the square root of the number of steps. We use $warmup\_steps=4000$. 

> NOTE: This part is very important. It is required to train with this model setup.

```python

class NoamOpt:
"Optim wrapper that implements rate."

def __init__(self, model_size, factor, warmup, optimizer):
self.optimizer = optimizer
self._step = 0
self.warmup = warmup
self.factor = factor
self.model_size = model_size
self._rate = 0

def step(self):
"Update parameters and rate"
self._step += 1
rate = self.rate()
for p in self.optimizer.param_groups:
p['lr'] = rate
self._rate = rate
self.optimizer.step()

def rate(self, step = None):
"Implement `lrate` above"
if step is None:
step = self._step
return self.factor * \
(self.model_size ** (-0.5) *min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
return NoamOpt(model.src_embed[0].d_model, 2, 4000,
torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
```

> Here are some example curves for this model for different model sizes and optimized hyperparameters.

```python
# Three settings of the lrate hyperparameters.
opts = [NoamOpt(512, 1, 4000, None),
NoamOpt(512, 1, 8000, None),
NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
None
```

![svg](2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_files/2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_68_0.svg)

## Regularization
### Label smoothing

During training, we used a label smoothing value of $\epsilon_{ls}=0.1$ [(cite)](https://arxiv.org/abs/1512.00567). Although smoothing the label confuses the model, it improves accuracy and BLEU score.

> We use KL div loss to implement label smoothing. Instead of using a one-hot distribution, we create a distribution that sets the target distribution to 1-smoothing and assigns the remaining probabilities to other words in the vocabulary.

```python
class LabelSmoothing(nn.Module):
"Implement label smoothing."
def __init__(self, size, padding_idx, smoothing=0.0):
super(LabelSmoothing, self).__init__()
self.criterion = nn.KLDivLoss(size_average=False)
self.padding_idx = padding_idx
self.confidence = 1.0 - smoothing
self.smoothing = smoothing
self.size = size
self.true_dist = None

def forward(self, x, target):
assert x.size(1) == self.size
true_dist = x.data.clone()
true_dist.fill_(self.smoothing / (self.size - 2))
true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
true_dist[:, self.padding_idx] = 0
mask = torch.nonzero(target.data == self.padding_idx)
if mask.dim() > 0:
true_dist.index_fill_(0, mask.squeeze(), 0.0)
self.true_dist = true_dist
return self.criterion(x, Variable(true_dist, requires_grad=False))
```

Let's take a look at an example to see the true probability distribution after smoothing.

```python
#Example of label smoothing.
crit = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
[0, 0.2, 0.7, 0.1, 0], 
[0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()), 
Variable(torch.LongTensor([2, 1, 0])))

# Show the target distributions expected by the system.
plt.imshow(crit.true_dist)
None
```

/Users/niepig/Desktop/zhihu/learn-nlp-with-transformers/venv/lib/python3.8/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args willbe deprecated, please use reduction='sum' instead.
warnings.warn(warning.format(ret))

![svg](2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_files/2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_73_1.svg)

```python
print(crit.true_dist)
```

tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
[0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])

Due to the existence of label smoothing, if the model is particularly confident about a word and outputs a particularly large probability, it will be penalized. As shown in the following code, as the input x increases, x/d will become larger and larger, 1/d will become smaller and smaller, but lossIt’s not always decreasing.

```python
crit = LabelSmoothing(5, 0, 0.1)
def loss(x):
d = x + 3 * 1
predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
])
#print(predict)
return crit(Variable(predict.log()),
Variable(torch.LongTensor([1]))).item()

y = [loss(x) for x in range (1, 100)]
x = np.arange(1, 100)
plt.plot(x, y)

```

[<matplotlib.lines.Line2D at 0x7f7fad46c970>]

![svg](2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_files/2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer_76_1.svg)

# Example

> We can start by trying a simple copy task. Given a set of random input symbols from a small vocabulary, the goal is to generate those same symbols.

## Synthetic Data

```python
def data_gen(V, batch, nbatches):
"Generate random data for a src-tgt copy task."
for i in range(nbatches):
data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
data[:, 0] = 1
src = Variable(data, requires_grad=False)
tgt = Variable(data, requires_grad=False)
yield Batch(src, tgt, 0)
```

## Loss function calculation

```python
class SimpleLossCompute:
"A simple loss compute and train function."
def __init__(self, generator, criterion, opt=None):
self.generator = generator
self.criterion = criterion
self.opt = opt

def __call__(self, x, y, norm):
x = self.generator(x)
loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
y.contiguous().view(-1)) / norm
loss.backward()
if self.opt isnot None:
self.opt.step()
self.opt.optimizer.zero_grad()
return loss.item() * norm
```

## Greedy decoding

```python
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
model.train()
run_epoch(data_gen(V, 30, 20), model,SimpleLossCompute(model.generator, criterion, model_opt))
model.eval()
print(run_epoch(data_gen(V, 30, 5), model, 
SimpleLossCompute(model.generator, criterion, None)))
```

> For simplicity, this code uses greedy decoding to predict the translation.

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
memory = model.encode(src, src_mask)
ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
for i in range(max_len-1):
out = model.decode(memory, src_mask,Variable(ys), 
Variable(subsequent_mask(ys.size(1))
.type_as(src.data)))
prob = model.generator(out[:, -1])
_, next_word = torch.max(prob, dim = 1)
next_word = next_word.data[0]
ys = torch.cat([ys, 
torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
```

tensor([[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

# Real scene example
Since the real data scene of the original jupyter requires multi-GPU training, this tutorial will not include it for the time being. Interested readers can continue to read the [original tutorial](https://nlp.seas.harvard.edu/2018/04/03/attention.html). In addition, since the original URL of the real data is invalid, the original tutorial should not be able to run the code of the real data scene.

# Conclusion

So far, we have implemented a complete Transformer line by line and trained and predicted it using synthetic data. I hope this tutorial can help you.

# Acknowledgements
This article was translated by Zhang Hongxu and edited by Duoduo. The original jupyter comes from Harvard NLP [The annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

<div id="disqus_thread"></div>
<script>
/**
* RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
* LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https: //disqus.com/admin/universalcode/#configuration-variables
*/
/*
var disqus_config = function () {
this.page.url = PAGE_URL; // Replace PAGE_URL with your page's canonical URL variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // REQUIRED CONFIGURATION VARIABLE: EDIT THE SHORTNAME BELOW
var d = document, s = d.createElement('script');

s.src = 'https://EXAMPLE.disqus.com/embed.js'; // IMPORTANT: Replace EXAMPLE with your forum shortname!

s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>pt>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript" rel="nofollow">comments powered by Disqus.</a></noscript>