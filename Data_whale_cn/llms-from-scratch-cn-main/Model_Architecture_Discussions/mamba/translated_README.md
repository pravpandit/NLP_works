Simple, minimalist implementation of Mamba in one PyTorch file.

Features:
* Same numerical output for forward and backward passes as the official implementation
* Simplified, readable, annotated code

Not included:
* Speed. The official implementation is heavily optimized, and these optimizations are one of the core contributions of the Mamba paper. Most of the implementation is kept simple for readability.
* Proper parameter initialization (although this could be added without sacrificing readability)

## Demo

See [demo.ipynb](demo.ipynb) for an example with prompt completion.

```python
from model import Mamba
from transformers import AutoTokenizer

model = Mamba.from_pretrained('state-spaces/mamba-370m')

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

generate(model, tokenizer, 'Mamba is the')
```
> Mamba is the longest venomous snake in the world, estimated to be over 150 meters long. Due to its huge size and highly venomousBite force, Mamba kills by stabbing the victim (which is more painful than a single bite, but less effective)

150 meters... 🫢 Scary!

## References

The Mamba architecture was proposed by [Albert Gu](https://twitter.com/_albertgu?lang=en) and [Tri Dao](https://twitter.com/tri_dao?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor) in [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).

The official implementation can be found here: https://github.com/state-spaces/mamba/tree/main