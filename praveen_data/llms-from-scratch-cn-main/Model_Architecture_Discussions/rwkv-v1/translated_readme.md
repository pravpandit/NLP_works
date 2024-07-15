### Detailed interpretation of RWKV v1 source code

Due to the long time, no corresponding pre-training scripts were found. For v1, the source code analysis is mainly done. The scripts for loading model implementation can be found in v2-v6. This document analyzes the core code of RWKV v1 in detail, including initialization, time mixing module, channel mixing module and multi-head attention mechanism. The code is implemented in PyTorch, which has good readability and extensibility.

---

#### Model initialization

First, we define a function `RWKV_Init` for initializing linear layer and embedding layer.

```python
def RWKV_Init(module, config):
for m in module.modules():
if not isinstance(m, (nn.Linear, nn.Embedding)):
continue
with torch.no_grad():
name = '[unknown weight]'
for name, parameter in module.named_parameters():if id(m.weight) == id(parameter):
break

shape = m.weight.data.shape
gain = 1.0 
scale = 1.0 

if isinstance(m, nn.Linear):
if m.bias is not None:
m.bias.data.zero_()
if shape[0] > shape[1]:
gain = math.sqrt(shape[0] / shape[1])
if shape[0] == config.vocab_size and shape[1] == config.n_embd:
scale = config.rwkv_emb_scale

if isinstance(m, nn.Embedding):
gain = math.sqrt(max(shape[0], shape[1]))
if shape[0] == config.vocab_size and shape[1] == config.n_embd:
scale = config.rwkv_emb_scale

if hasattr(m, 'scale_init'):
scale = m.scale_init

print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale,2):g}'.ljust(4), name)

gain *= scale
if gain == 0:nn.init.zeros_(m.weight)
elif gain > 0:
nn.init.orthogonal_(m.weight, gain=gain)
else:
nn.init.normal_(m.weight, mean=0, std=-gain)
```

This function iterates over all linear and embedding layers in the module and initializes their weights according to certain conditions. For linear layers, if the bias exists, it is initialized to zero; for embedding layers, the gain and scaling factor of the weights are calculated. Different initialization methods are used according to different conditions, such as orthogonal initialization or normal initialization.

---

#### Time Mixing Module

The `RWKV_TimeMix` class implements the time mixing mechanism.

```python
class RWKV_TimeMix(nn.Module):
def __init__(self, config, layer_id):
super().__init__()
assert config.n_attn % config.n_head== 0
self.layer_id = layer_id
self.ctx_len = config.ctx_len
self.n_head = config.n_head
self.head_size = config.n_attn // config.n_head

with torch.no_grad(): 
ww = torch.ones(config.n_head, config.ctx_len)
curve = torch.tensor([-(config.ctx_len - 1 - i) for i in range(config.ctx_len)]) 
for h in range(config.n_head):
if h < config.n_head - 1:
decay_speed = math.pow(config.ctx_len, -(h+1)/(config.n_head-1))
else:
decay_speed = 0.0
ww[h] = torch.exp(curve * decay_speed)
self.time_w = nn.Parameter(ww)

self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_len))
self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_len, 1))
self.time_gamma = nn.Parameter(torch.ones(config.ctx_len, 1))

self.time_shift = nn.ZeroPad2d((0,0,1,-1))

self.key= nn.Linear(config.n_embd, config.n_attn)
self.value = nn.Linear(config.n_embd, config.n_attn)
self.receptance = nn.Linear(config.n_embd, config.n_attn)

self.output = nn.Linear(config.n_attn, config.n_embd)

self.key.scale_init = 0
self.receptance.scale_init = 0
self.output.scale_init = 0

def forward(self, x):
B, T, C = x.size()
TT = self.ctx_len
w = F.pad(self.time_w, (0, TT))
w = torch.tile(w, [TT])
w= w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
w = w[:, :, TT-1:] 
w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)

k = self.key(x)
v = self.value(x)
r = self.receptance(x)

k = torch.clamp(k, max=30, min=-60) 
k = torch.exp(k)
sum_k = torch.cumsum(k, dim=1)

kv = (k * v).view(B, T, self.n_head, self.head_size)

wkv= (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)

rwkv = torch.sigmoid(r) * wkv / sum_k

rwkv = self.output(rwkv)

return rwkv * self.time_gamma[:T, :]
```

This class implements the time mixing mechanism, transforming the input through the time weight matrix `time_w`. The `time_w` matrix is ​​initialized according to the number of heads and the context length, and then the input is transformed and mixed in the time dimension. The three linear layers `key`, `value` and `receptance` generate key, value and reception signals respectively, and calculate the output through the sigmoid function.

---

#### Channel Mixing Module

The `RWKV_ChannelMix` class implements the channel mixing mechanism.

```python
class RWKV_ChannelMix(nn.Module):
def __init__(self, config, layer_id):
super().__init__()
self.layer_id = layer_id
self.time_shift = nn.ZeroPad2d((0,0,1,-1))

hidden_sz = 5 * config.n_ffn // 2 
self.key = nn.Linear(config.n_embd, hidden_sz)
self.value = nn.Linear(config.n_embd, hidden_sz)
self.weight = nn.Linear(hidden_sz, config.n_embd)
self.receptance = nn.Linear(config.n_embd, config.n_embd)

self.receptance.scale_init = 0
self.weight.scale_init = 0

def forward(self, x):
B,T, C = x.size()

x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
k = self.key(x)
v = self.value(x)
r = self.receptance(x)

wkv = self.weight(F.mish(k) * v)

rwkv = torch.sigmoid(r) * wkv

return rwkv
```

This class implements the channel mixing mechanism, transforming the input through three linear layers, `key`, `value`, and `receptance`. The tensors generated by the `key` and `value` layers are transformed by the `mish` activation function and then weighted by the `weight` layer, and finally multiplied with the received signal generated by the `receptance` layer to get the final output.

---

#### Multi-head attention mechanism

The `MHA_rotary` class implements the multi-head attention mechanismsystem, and introduced rotational position encoding.

```python
class MHA_rotary(nn.Module):
def __init__(self, config, layer_id, time_shift = False):
super().__init__()
self.layer_id = layer_id
assert config.n_attn % config.n_head == 0
self .n_head = config.n_head
self.ctx_len = config.ctx_len
self.head_size = config.n_attn // config.n_head

if time_shift:
self.time_shift = nn.ZeroPad2d((0,0,1,-1))

self.query = nn.Linear(config.n_embd, config.n_attn)self.key = nn.Linear(config.n_embd, config.n_attn)
self.value = nn.Linear(config.n_embd, config.n_attn)

self.register_buffer("mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))

self.rotary_ndims = int(self.head_size * 0.5)
self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

self.output = nn.Linear(config.n_attn, config.n_embd)

def forward(self, x):
B, T, C = x.size()

if hasattr(self, 'time_shift'):x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)

q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2

)
v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
cos, sin= self.rotary_emb(q, seq_len=T)
q, k = apply_rotary_pos_emb(q, k, cos, sin)
q = torch.cat((q, query_pass), dim=-1)
k = torch.cat((k, key_pass), dim=-1)

att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
att = F.softmax(att, dim = -1)

x = att @ v
x = x.transpose(1, 2).contiguous().view(B, T, -1)

x = self.output(x)
return x
```

This class implements multipleThe RWKV v1 model implements the basic time mixing, channel mixing, and multi-head attention mechanisms. By transforming and mixing the input in multiple dimensions, complex features can be extracted and represented. The above code snippet shows the core components and main calculation process of the model, which provides a solid foundation for the optimization and improvement of subsequent versions.