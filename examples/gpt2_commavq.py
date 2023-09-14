#!/usr/bin/env python3
# pip3 install tiktoken

import functools, argparse
import numpy as np
from tqdm import trange

np.set_printoptions(linewidth=200)
from typing import Optional, Dict, Tuple

from tinygrad.helpers import Timing, getenv, dtypes, DEBUG
from tinygrad.ops import GlobalCounters
from tinygrad.ops import Device
from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear
from tinygrad.jit import TinyJit
from tinygrad.shape.symbolic import Variable

model_file = '/home/gijs/code_projects/commavq/models/pytorch_model.bin'
MAX_CONTEXT = 20 * 129  # comma


class LayerNorm:
  def __init__(self, dim, eps=1e-5):
    self.eps = eps
    self.weight = Tensor.ones(dim)
    self.bias = Tensor.zeros(dim)

  def __call__(self, x: Tensor):
    return (x.layernorm(eps=self.eps)) * self.weight + self.bias


class Attention:
  def __init__(self, dim, n_heads, linear=Linear):
    self.c_attn = linear(dim, 3 * dim, bias=True)
    self.c_proj = linear(dim, dim, bias=True)
    self.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads

  def __call__(self, x: Tensor, cache_k: Optional[Tensor], cache_v: Optional[Tensor], start_pos: int, mask: Optional[Tensor],
               jit_ctx: Optional[Dict[Variable, int]] = None) -> Tuple[Tensor, Tensor, Tensor]:
    xqkv = self.c_attn(x)
    xq, xk, xv = [xqkv.slice([None, None, (i * self.dim, (i + 1) * self.dim)]) for i in range(3)]
    xq, xk, xv = [x.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim) for x in (xq, xk, xv)]

    bsz, seqlen, _, _ = xq.shape
    # kv caching!
    if start_pos == 0:
      keys, values = xk, xv
    else:
      assert cache_k, "no cache"
      # assert start_pos == cache_k.shape[1] and start_pos == cache_v.shape[1], "cache is wrong shape"
      assert seqlen == xk.shape[1] and seqlen == xv.shape[1], "seqlen is wrong shape?!?"
      keys, values = cache_k.cat(xk, dim=1), cache_v.cat(xv, dim=1)

    # save the cache
    cache_k, cache_v = keys, values
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    output = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2).reshape(bsz, seqlen, -1)
    return self.c_proj(output), cache_k, cache_v


class FeedForward:
  def __init__(self, dim, hidden_dim, linear=Linear):
    self.c_fc = linear(dim, hidden_dim, bias=True)
    self.c_proj = linear(hidden_dim, dim, bias=True)

  def __call__(self, x: Tensor) -> Tensor:
    return self.c_proj(self.c_fc(x).gelu())


class TransformerBlock:
  def __init__(self, dim, n_heads, norm_eps, linear=Linear):
    self.attn = Attention(dim, n_heads, linear)
    self.mlp = FeedForward(dim, 4 * dim, linear)
    self.ln_1 = LayerNorm(dim, norm_eps)
    self.ln_2 = LayerNorm(dim, norm_eps)

  def __call__(self, x: Tensor, cache_k: Optional[Tensor], cache_v: Optional[Tensor], start_pos: int, mask: Optional[Tensor],
               jit_ctx: Optional[Dict[Variable, int]] = None, realize=True):
    if start_pos > 0 and mask is None and getenv("JIT"):
      start_pos_var = Variable("start_pos", 1, MAX_CONTEXT)
      cache_k = cache_k.reshape(cache_k.shape[0], start_pos_var, cache_k.shape[2], cache_k.shape[3])
      cache_v = cache_v.reshape(cache_v.shape[0], start_pos_var, cache_v.shape[2], cache_v.shape[3])
      # need this because we don't reshape back to int shape in the jitted path and we don't have the correct var_vars in cache
      cache_k.lazydata.var_vals[start_pos_var] = start_pos
      cache_v.lazydata.var_vals[start_pos_var] = start_pos

    output, cache_k, cache_v = self.attn(self.ln_1(x), cache_k, cache_v, start_pos, mask, jit_ctx=jit_ctx)
    h = x + output
    h = (h + self.mlp(self.ln_2(h)))
    if realize:
      return h.realize(), cache_k.realize(), cache_v.realize()
    return h, cache_k, cache_v
    # return (h + self.mlp(self.ln_2(h))).realize(), cache_k.realize(), cache_v.realize()


class Transformer:
  def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, linear=Linear, max_seq_len=1024):
    self.wte = Embedding(vocab_size, dim)
    self.wpe = Embedding(max_seq_len, dim)
    self.h = [TransformerBlock(dim, n_heads, norm_eps, linear) for _ in range(n_layers)]
    self.kv_caches = None
    self.ln_f = LayerNorm(dim, norm_eps)
    self.lm_head = linear(dim, vocab_size, bias=False)
    self.n_layers = n_layers

    self.embed_jitted = TinyJit(self.embed)
    self.postprocess_jitted = TinyJit(self.postprocess)
    self.h_jitted = [TinyJit(h.__call__) for h in self.h]

    self.embed_jitted_start = TinyJit(self.embed)
    self.postprocess_jitted_start = TinyJit(self.postprocess)
    self.h_jitted_start = [TinyJit(h.__call__) for h in self.h]

    self.bigger_jit_jit = TinyJit(self.bigger_jit)

  def bigger_jit(self, tokens, pos, start_pos, kv_caches, jit_ctx: Optional[Dict[Variable, int]] = None):
    h = self.embed(tokens, pos, realize=False)

    for i, (hi, (cache_k, cache_v)) in enumerate(zip(self.h, kv_caches)):
      h, cache_k, cache_v = hi(h, cache_k, cache_v, start_pos=start_pos, mask=None, jit_ctx=jit_ctx, realize=False)
      kv_caches[i] = (cache_k, cache_v)
    logits = self.lm_head(self.ln_f(h)).realize()

    for i in range(len(kv_caches)):
      kv_caches[i] = (kv_caches[i][0].realize(), kv_caches[i][1].realize())
    return logits, kv_caches

  def embed(self, tokens, pos, realize=True):
    tok_emb = self.wte(tokens).cast(dtypes.float16)
    pos_emb = self.wpe(pos).cast(dtypes.float16)
    h = tok_emb + pos_emb
    if not realize:
      return h
    return h.realize()

  def postprocess(self, x, temperature: Optional[float]):
    logits = self.lm_head(self.ln_f(x))
    if temperature is not None: return (logits[:, -1, :] / (temperature + 1e-10)).softmax().flatten().realize()
    return logits.realize()

  def __call__(self, tokens: Tensor, start_pos: int, temperature: Optional[float] = None):
    _bsz, seqlen = tokens.shape
    if not hasattr(self, 'allpos'): self.allpos = Tensor.arange(0, MAX_CONTEXT).reshape(1, -1).realize()
    if getenv("JIT"):
      start_pos_var = Variable("start_pos", 0, MAX_CONTEXT)
      pos = self.allpos.shrink(((0, self.allpos.shape[0]), (start_pos_var, start_pos_var + seqlen)))
      pos.lazydata.var_vals[start_pos_var] = start_pos
      if start_pos == 0:
        self.kv_caches = [(None, None) for _ in range(self.n_layers)]
        embed_jitted, h_jitted, postprocess_jitted = self.embed_jitted_start, self.h_jitted_start, self.postprocess_jitted_start
        mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=self.wte.weight.dtype).triu(start_pos + 1).realize()

        h = embed_jitted(tokens, pos)

        for i, (hi, (cache_k, cache_v)) in enumerate(zip(h_jitted, self.kv_caches)):
          h, cache_k, cache_v = hi(h, cache_k, cache_v, start_pos=start_pos, mask=mask, jit_ctx={start_pos_var: start_pos})
          self.kv_caches[i] = (cache_k, cache_v)
      else:
        print("Continue > 0")
        assert seqlen == 1
        # embed_jitted, h_jitted, postprocess_jitted = self.embed_jitted, self.h_jitted, self.postprocess_jitted
        # mask = None
        logits, self.kv_caches = self.bigger_jit_jit(tokens, pos, start_pos, self.kv_caches, jit_ctx={start_pos_var: start_pos})
        return logits
      return postprocess_jitted(h, temperature)
    else:
      pos = self.allpos.shrink(((0, self.allpos.shape[0]), (start_pos, start_pos + seqlen)))
      mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=dtypes.float32).triu(start_pos + 1).realize()
      h = self.embed(tokens, pos)
      for i, (hi, (cache_k, cache_v)) in enumerate(zip(self.h, self.kv_caches)):
        h, cache_k, cache_v = hi(h, cache_k, cache_v, start_pos=start_pos, mask=mask)
        self.kv_caches[i] = (cache_k, cache_v)
      return self.postprocess(h, temperature)


# **** files and arguments ****

VOCAB_SIZE = 50257
MODEL_PARAMS = {
  'gpt2': dict(n_layers=12, n_heads=12, dim=768, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 124M params
  'comma_gpt2': dict(n_layers=24, n_heads=16, dim=1024, norm_eps=1e-5, vocab_size=1025),  # 124M params
  'gpt2-medium': dict(n_layers=24, n_heads=16, dim=1024, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 350M params
  'gpt2-large': dict(n_layers=36, n_heads=20, dim=1280, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 774M params
  'gpt2-xl': dict(n_layers=48, n_heads=25, dim=1600, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 1558M params
}


def get_url(model_size):
  return f'https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin'


class GPT2:
  @staticmethod
  # def build(model_size="gpt2"):
  def build(model_size="comma_gpt2", max_seq_len=2580):
    import tiktoken
    from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict
    tokenizer = tiktoken.get_encoding("gpt2")
    # JIT=1 python examples/gpt2.py --prompt="Hello." --count=10 --temperature=0 --timing
    params = MODEL_PARAMS[model_size]
    # model = Transformer(**params, max_seq_len=1024)
    model = Transformer(**params, max_seq_len=max_seq_len)
    # weights = torch_load(fetch_as_file(get_url(model_size)))
    weights = torch_load(model_file)
    weights = {k.replace("transformer.", ""): v for k, v in weights.items()}
    # special treatment for the Conv1D weights we need to transpose
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    for k in weights.keys():
      if any(k.endswith(w) for w in transposed):
        weights[k] = Tensor(weights[k].numpy().T)
    # lm head and wte are tied
    weights['lm_head.weight'] = Tensor(weights['wte.weight'].numpy())
    # weights['lm_head.weight'] = Tensor(weights['transformer.wte.weight'].numpy())
    if getenv("FP16"):  # todo create pr for this
      for k, v in weights.items():
        weights[k] = v.cpu().cast(dtypes.float16).realize()
    load_state_dict(model, weights)
    return GPT2(model, tokenizer)

  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  def greedy_until(self, prompt: str, max_length: int, temperature: float, timing: bool = False):
    toks = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    start_pos = 0
    for _ in trange(max_length, disable=(timing == True)):
      GlobalCounters.reset()
      if args.timing: print("")
      st = GlobalCounters.time_sum_s
      with Timing("total ", enabled=timing):
        with Timing(f"ran model in ", on_exit=(lambda et: f", {(GlobalCounters.time_sum_s - st) * 1e3:.2f} ms on GPU" +
                                                          f", {GlobalCounters.global_ops * 1e-9:.2f} GOPS, {GlobalCounters.global_mem * 1e-9:.2f} GB" +
                                                          f", {GlobalCounters.global_mem * 1e-9 / (GlobalCounters.time_sum_s - st):.2f} GB/s") if DEBUG else None,
                    enabled=timing):
          probs = self.model(Tensor([toks[start_pos:]]), start_pos, temperature)
        probs_np = probs.numpy()
        tok = int(np.random.choice(len(probs_np), p=probs_np))
      start_pos = len(toks)
      toks.append(tok)
      output = self.tokenizer.decode(toks)
    return output


# **** main code ****

if __name__ == "__main__":
  Tensor.no_grad = True
  print(f"using {Device.DEFAULT} backend")

  parser = argparse.ArgumentParser(description='Run GPT2 in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--prompt', type=str, default="What is the answer to life, the universe, and everything?", help="Phrase to start with")
  parser.add_argument('--count', type=int, default=100, help="Max number of tokens to generate")
  parser.add_argument('--temperature', type=float, default=0.8, help="Temperature in the softmax")
  parser.add_argument('--model_size', type=str, default="gpt2-medium", help="Size of model to use [gpt2, gpt2-medium, gpt2-large, gpt2-xl]")
  parser.add_argument('--timing', action='store_true', help="Print timing per token")
  args = parser.parse_args()

  print(f"using {args.model_size}")
  gpt2 = GPT2.build(args.model_size)
  print('Generating text...')
  y = gpt2.greedy_until(args.prompt, args.count, args.temperature, timing=args.timing)
  print(y)