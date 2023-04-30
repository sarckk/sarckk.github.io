---
title: Grokking Transformers
date: '2023-04-10'
categories:
  - AI
tags:
  - AI
  - LLM
  - NLP
---


Today's Large Language Models (LLM) are based on Transformers, a deep learning model architecture for sequence-to-sequence transformations based on the attention mechanism. While it was originally proposed and used in Natural Language Processing (NLP) tasks like language translation, it turns out that a lot of things that we care about can be modelled in terms of sequences, making transformers a useful model in a wide variety of applications beyond NLP, such as [image processing](https://arxiv.org/abs/2103.14030) and [reinforcement learning](https://arxiv.org/abs/2106.01345). Given the overwhelming success of transformers in deep learning and the outsized impact that transformer-based generative AI (e.g. GPT) has had -- and will likely continue to have -- on our society, I thought I should finally take time to read and understand the paper ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) that first proposed Transformers. That paper is now almost 6 years old(!) but better late than never, right?

There are already [many](https://jalammar.github.io/illustrated-transformer/) [tutorials](https://www.youtube.com/watch?v=n9sLZPLOxG8) [covering](https://www.youtube.com/watch?v=n9sLZPLOxG8) [transformers](https://www.youtube.com/watch?v=ptuGllU5SQQ) online, so this article is mostly for my own learning -- this is already [well documented](https://ideas.time.com/2011/11/30/the-protege-effect/), but I find that writing in a pedagogical style helps immensely in solidifying in my learnings and is almost always worth the effort. If anyone else stumbles across this post and finds it helpful, that's an added bonus!

This post will cover the technical details behind the Transformer model. The core concept behind transformers -- self- and cross- attention -- really isn't too hard too grasp, but I've found that actually getting your hands dirty and implementing the model in Pytorch elevates your understanding of the material. Personally, I ran into many issues while trying to write and train the model that I wouldn't have known had I stopped at reading the paper or other tutorials online.


# Table of Contents <!-- omit from toc --> 
- [All about transformations](#all-about-transformations)
- [Diving into Transformers: the architecture](#diving-into-transformers-the-architecture)
  - [Diving deeper: the embedding layer](#diving-deeper-the-embedding-layer)
    - [Step 1: Tokenization](#step-1-tokenization)
    - [Step 2: Converting tokens to vocabulary indices](#step-2-converting-tokens-to-vocabulary-indices)
    - [Step 3: Embedding](#step-3-embedding)
    - [Step 4: Adding Positional Encoding](#step-4-adding-positional-encoding)
  - [Encoder](#encoder)
    - [Attention is all you need](#attention-is-all-you-need)
    - [A closer look at attention](#a-closer-look-at-attention)
    - [Matrix formulation of attention](#matrix-formulation-of-attention)
    - [Back to Positional Encodings](#back-to-positional-encodings)
    - [Multi-Head Attention](#multi-head-attention)
    - [Masking in the encoder](#masking-in-the-encoder)
    - [Feed-Forward Network](#feed-forward-network)
    - [Encoder: the remaining bits](#encoder-the-remaining-bits)
  - [Decoder](#decoder)
    - [Difference #1: Cross-Attention](#difference-1-cross-attention)
    - [Difference #2: Masking in self-attention](#difference-2-masking-in-self-attention)
  - [Linear + Softmax Layer](#linear--softmax-layer)
- [Training](#training)
    - [Loss function](#loss-function)
    - [Regularization](#regularization)
    - [Model hyperparameters](#model-hyperparameters)
    - [Optimizer](#optimizer)
    - [Learning-rate scheduling](#learning-rate-scheduling)
- [Inference](#inference)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)
- [Where to go from here](#where-to-go-from-here)

# All about transformations
A Transformer -- as its name suggests -- *transforms* an input sequence $(x_1,x_2,...,x_S)$ into an output sequence $(y_1,y_2,...,y_T)$. Because this formulation is so general, it doesn't say what $x_1$ and $y_1$ should represent -- it could be a word, a sub-word, a character, a pixel, or a token representing any arbitrary thing. However, I'll be talking about Transformers in the context of NLP here, because that's what it was originally invented for. So if we're talking about machine translation, the input sequence could be a sequence of words in one language (e.g. English) and the output could be a sequence of words in the target language (e.g. German):

![Transformer transforms](https://sarckk.github.io/media/transformer1.svg)

In the diagram above, each element in a sequence represents a word, but in practice, it is common for this to a smaller unit than a word (e.g. a sub-word) depending on the tokenizer you use. 

# Diving into Transformers: the architecture
Now let's talk about what a Transformer actually looks like. From the original paper:

<p align="center" width="100%">
<img src="https://sarckk.github.io/media/transformer_architecture.png" width=450/>
</p>

If you're like me, this might be a bit overwhelming to take in at first. In reality, there's only 4 major components to a transformer architecture: the embedding layer, the encoder, the decoder, and the final linear+softmax layers that transform the output of the decoder into probabilities. Here's the same diagram with some annotations overlaid on top:

<p align="center" width="100%">
<img src="https://sarckk.github.io/media/transformer_arch_illustrated.png" width=400/>
</p>

At a high level, here's the journey that our input sequence takes to be transformed into an output sequence:
1) Go through input embedding layer which projects each element $x_i$ in a sequence of length **S** into a higher dimensional vector. 
2) Add "Positional Encoding" vector to each element in the sequence (which remember, is now a high-dimensional vector). We'll talk about this in more detail later.
3) Go through the encoder (orange block in the diagram above) **N** times. These **N** encoders have the same architecture but do not share weights.At the end of this step, we get a tensor that compactly represents the input sequence. 
4) On the decoder side, we pass in a sequence of length **T**, which goes through the same embedding layer + positional encoding as we had for the input sequence.  
5) The output embedding goes through a stack of **N** decoders (again no sharing of weights), each of which uses the tensor we got from **Step 3** in some way. At the end of this step, we get another tensor.
6) We pass this tensor through a final Linear + Softmax layer to obtain **T** probabilities, where again **T** is the length of the sequence we passed into the decoder.
7) We convert those probabilities into actual tokens (e.g. by taking the token with the highest probability), giving us an output sequence of length **T**.

So from an input sequence of length **S** and decoder input of length **T**, we got -- as the final product of the Transformer -- another sequence of length **T**.

It's okay if some of these steps do not make sense yet. For example, I was confused as to why a sequence of length **T** had to be passed into the decoder to get another sequence of the same length **T**: if we are just passing in an input sentence and expect the model to output the translated text, what are we passing to the decoder? WTF? Don't worry, this will become clear when we talk about training and inference later in this article.

Now, let's talk about each of these components in greater detail during **training**. Then we'll talk about what happens at **inference time**.

## Diving deeper: the embedding layer 
Let's start from the very beginning. The original Transformer in the 2017 paper was trained on the task of translating English sentences to German, using the WMT 2014 English-German dataset. This dataset contains ~4.5 million pairs of English sentence and its corresponding translation in German. We'll use this example to explain what happens in a Transformer for the rest of the article.

### Step 1: Tokenization
The first step is to tokenize each sentence into a sequence of tokens. During training, the input to the Transformer encoder is thus sequence(s) of tokens generated from the English sentences. The paper mentions that they used [byte-pair](https://en.wikipedia.org/wiki/Byte_pair_encoding) encoding (see page 7) for tokenization, but for simplicity I'll assume that the sentences are tokenized word by word.

<p align="center" width="100%">
<img src="https://sarckk.github.io/media/transformer_embed_stage1.svg"/>
</p>

Above is an illustration of what this would look like for an example mini-batch of 3 English sentences. Note that we have 3 special tokens: `<bos>` denoting the beginning of sentence, `<eos>` marking the end of a sentence, and `<pad>` representing an "empty" token to make all the sequences in the tensor of the same length (i.e. the maximum sequence length across all sentences in the mini-batch). 

### Step 2: Converting tokens to vocabulary indices
Before we can pass this into the encoder, we need to convert these sequences of strings into a numerical representation instead so we can do some computation and make GPUs go *brrr*. To do this, we create -- from the dataset -- a mapping from all possible tokens (in this case, words) to its index in a vocabulary. So, the word "The" might have an index of **11**, which uniquely identifies that token. We also leave out some indices for the special tokens that we introduced (`<bos>`, `<eos>` and `<pad>`): for example, their indices might be 0, 1 and 2 respectively.

<p align="center" width="100%">
<img src="https://sarckk.github.io/media/transformer_embed_stage2.svg"/>
</p>

### Step 3: Embedding
Once we have the input tensor, we turn each word (which by this point is a number representing its index in our vocabulary) into a high-dimensional vector that we call an **embedding vector**. This can be implemented using PyTorch's `nn.Embedding()` module. In the original paper, the embedding dimension is **512**, but this is a hyperparameter that we can tune for our model through experiments. Note that in the paper, they also multiply the embedding weights by `sqrt` of the embedding dimension (see page 5). 

After this step, we have a tensor of shape (batch size, (max) input sequence length, embedding dimension). For the rest of the article, I'll use the short form `B` for batch size, `S` for the maximum input sequence length, and `D` for the embedding dimension. In our example, `B=3`, `S=9` and `D=512`.

<p align="center" width="100%">
<img src="https://sarckk.github.io/media/transformer_embed_stage3.png"/>
</p>

### Step 4: Adding Positional Encoding
I'll skim over this step for now, because it only really makes sense when we start looking at how attention is computed. Without too much detail though, here we basically add a tensor with the same shape `(B,S,D)` to our tensor from **Step 2**. This tensor that we add encodes information about the relative order of each word in a sentence, since this is information that we'd like the Transformer to consider in the computation of the final output probabilities. Again, I'll come back to this later in the article.

In the end, the input to the encoder is a tensor of shape `(B,S,D)`. The same thing happens for the decoder, using the target language sentences instead. 

----
## Encoder 
We've finally reached the encoder. The encoding step consists of **N** independent encoder units stacked on top of each other. These encoders are identical in architecture, but do not share weights between them and are thus separately updated during backpropagation.

The encoder unit itself comprises 2 parts:
1) A Self-Attention module, followed by
2) A Feed-Forward network. 

We'll start with this high level picture of the encoder and gradually fill in more details.

### Attention is all you need
The **core idea** behind the Transformer is to replace recurrence and convolutions that made up previous sequence-to-sequence models with one entirely based on the attention mechanism. In simple terms, the attention mechanism is basically just taking a bunch of dot products between sequences. And **self-attention** is just particular case of attention where the sequences that we're concerned with is actually all the same -- just one sequence.

Remember that at this point, our example input to the encoder is a tensor of shape `(3,9,512)`. To make the explanation easier, let's look at what happens for **one** single sentence out of 3 total sentences in this mini-batch: when we actually pass through the entire batch with 3 sentences, logically it will be as if we pass through each sentence separately and merge the 3 outputs together.

Let's look at just one sentence: "This jacket is too small for me". After the embedding layer, we have a tensor of shape `(9,512)`. To encode this tensor, we essentially pass all 9 of the 512-dimensional embedding vector to the self-attention module **at the same time**:

<p align="center" width="100%">
<img src="https://sarckk.github.io/media/transformer_encoder_simple.svg" width=400/>
</p>

The goal of the self-attention module is to figure out how the words in the sentence (or more generally, tokens in a sequence) relate to each other. 

Remember, the sentence is "This jacket is too small for me". When we look at the adjective `"small"`, we want to understand what object it is referring to. Clearly, we know that it is referring to the `"jacket"`, but the Transformer model has to learn this. In other words, it has to learn how each word relates to another. In vector space, we have a concept for computing the similarity between vectors: **dot product**. 

Dot products form the basis of the attention mechanism.

### A closer look at attention 
Computing attention involves 3 inputs: query(s), keys(s), and value(s) where these are all vectors. More formally, we have:

- Queries $q_1,...,q_T$ where $q_i \in$ $\mathbb{R}^{d_k}$, where $d_k$ is the dimension of the query vector, and $T$ is the number of queries
- Keys $k_1,...k_K$ where $k_i \in$ $\mathbb{R}^{d_k}$, and $K$ is the number of key-value pairs
- Value $v_1,...,v_K$ where $v_i \in$ $\mathbb{R}^{d_v}$, where $d_v$ is the dimension of the value vector, which is not necessarily equal to $Q$, although in our case of English-German translation, it is.

Note that $T$, the number of queries doesn't necessarily have to equal $K$, the number of key-value pairs, but the number of keys must be the same as the number of values (for them to form a key-value pair). Furthermore, the query and key vectors must have the same dimension, $d_k$ so we can do a dot product.

Given this formulation, the attention function on $q_i$ does the following:
- Get dot product of query and key vectors to get a scalar value: $\alpha_{ij} = q_i \cdot k_i$
- Normalize each dot product $\alpha_{ij}$ by performing [softmax](https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax) across all $j$, where $j=0,...,K$ , where $K$ is the number of key-value pairs. This gives us weights $w_{ij}$ for all $j$.
- Output weighted sum of all $v_j$ where weight of $v_j$ is $w_{ij}$.

**Back to our example sentence**, we have $x_1,...,x_9$ where $x_i$ is a 512-dimensional embedding vector representing each word in the sentence `"This jacket is too small for me"` plus the `<bos>` and `<eos>` tokens. We obtain our query, key and value vectors from $x_i$ by multiplying it each time with a different matrix:

$$
\begin{aligned}
k_i &= W^Kx_i, \text{where } W^K \in \mathbb{R}^{d_k \times d_k} \\\\
q_i &= W^Qx_i, \text{where } W^Q \in \mathbb{R}^{d_k \times d_k} \\\\
v_i &= W^Vx_i, \text{where } W^V \in \mathbb{R}^{d_v \times d_v} \\\\
\end{aligned}
$$

In our case, $d_k=d_v=512$. We have $W^K$, $W^Q$ and $W^Q$ matrices that linearly project each key, query and value vectors -- this allows for more flexibility in both how the model chooses to define "similarity" between words (by updating $K$ and $Q$), as well as what the final weighted sum represents (by updating $V$) in latent space. In Pytorch code, these matrices are implemented as [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) modules with `bias=False`. 

<figure align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="https://sarckk.github.io/media/transformer_self_attention.svg" />
<figcaption><strong>Above: </strong>Illustration of how self-attention works for two tokens "jacket" and "small". For each token, we obtain the contribution by all other tokens and sum them up to obtain the final output for that token.</figcaption>
</figure>

Now that we have $k_i$, $q_i$ and $v_i$, we just compute the corresponding output for $x_i$ using the steps outlined earlier, computing the sum of all vectors weighed by the dot products. Here, since $q_i$,$k_i$ and $v_i$ are all derived from $x_i$, we give it a special name: **self-attention**. 


### Matrix formulation of attention
As you can see, attention is computed using dot products between any two words within a sequence, allowing the Transformer to learn long-range dependencies in a sequence more easily. One downside of this, though, is that the computation of attention scores is quadratic in the length of the input sequence $N$. This quadratic $O(N^2)$ complexity is an issue because it means it will take a lot of compute for long sequences. 

Fortunately, we can represent the computation as a product of a few matrix multiplications, which is easily parallelizable on GPU/TPUs. Given matrices $Q$, $K$ and $V$ containing rows of query, key and value vectors respectively, the general formulation of attention in matrix form is as follows:

\begin{equation}
 Attention(Q,K,V) = softmax(QK^T)V
\end{equation}

Again, the $Q$, $K$ and $V$ matrices are computed using the corresponding weight matrices $W^Q$, $W^K$, and $W^V$: for example, if we have a matrix $X$ where each row is an embedding vector in our sequence, then we'd have $Q=XW^Q$, $K=XW^K$ and $V=XW^V$ for the self-attention sublayer in the encoder. As we'll see later when we get to cross-attention in the decoder, $Q$, $K$ and $V$ do not necessarily need to come from the same single matrix $X$.

<figure align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="https://sarckk.github.io/media/transformer_matrix_attention.png"/>
<figcaption>Self-attention performed as matrix multiplications</figcaption>
</figure>

The authors in the Transformers paper also apply a scaling factor of $\frac{1}{d_k}$ to the matrix of dot products (numerator) to prevent the products from becoming too large, which can "\[push\] the softmax function into regions where it has extremely small gradients" ([Viswani et al, 2017, pg 4](https://arxiv.org/pdf/1706.03762.pdf)):

\begin{equation}
 Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
\end{equation}

This ability for parallelization is a part of why the Transformer has been so successful -- previous models based on recurrence, for example, cannot be parallelized because the computation of its state at time $t$, $h_t$ necessarily depends on the computation of its previous state at time $t-1$, $h_{t-1}$.

Another important consequence of relying heavily on attention is that we can visualize the attention weights, which can aid in debugging as well as interpreting and explaining the model output.

<br />
<figure align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="https://raw.githubusercontent.com/jessevig/bertviz/master/images/head-view.gif" width=300>
<figcaption><strong>Above:</strong> Transformer attention can be visualized, giving us some visibility into what these models learn. Source: BertViz github repo (<a href="https://github.com/jessevig/bertviz">https://github.com/jessevig/bertviz</a>). BertViz is a tool for visualizing attention in Transformer language models
</figcaption>
</figure>

### Back to Positional Encodings 
Now we can finally talk about why we need positional encodings. We've seen that (self-)attention basically comes down to taking a bunch of dot products and outputting a new vector with this information. The problem is, by simply taking dot products, we lose information about the relative order of these words in a sentence. And we know that the position of a word in a sentence matters. 

To encode information about the position of each token in the sequence, we add **positional encodings** to the input embeddings. In practice, there are many ways to generate this -- including having the network learn this during training -- but the authors use the following formula:

$$
\begin{aligned}
PE_{(pos,2i)} &= \sin(pos/10000^{2i/d_{emb}}) \\\\
PE_{(pos,2i+1)} &= \cos(pos/10000^{2i/d_{emb}})
\end{aligned}
$$

where $i$ is the index along the embedding dimension and $pos$ is the position of the token in the sequence. Both are 0-indexed. By having sine and cosine functions of varying periods, we are able to inject information about position in continuous form. 

<figure align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="https://sarckk.github.io/media/pos_encoding.png" width=400>
<figcaption>Illustration of positional encodings</figcaption>
</figure>

I don't have much to add on positional encodings, though I'll point out that the periodic nature of sinusoids used here has some nice properties, like placing more emphasis on **relative** -- as opposed to absolute -- order.

<details>
<summary>Implementation of Positional Encoding layer</summary>
There are many ways to implement this, but I've chosen to do it this way:

```python
class PositionalEncoding(nn.Module):
  def __init__(self, emb_dim: int, max_seq_len: int = 5000):
    super().__init__()
    assert emb_dim % 2 == 0, "Embedding dimension must be divisble by 2"
    self.dropout = nn.Dropout(0.1)
    
    pos = torch.arange(max_seq_len)[:, None] # [seq_len, 1]
    evens = 10000. ** (-torch.arange(0,emb_dim,step=2) / emb_dim)
    evens = evens[None, :] # [1, ceil(emb_dim/2)]
    evens = pos * evens # [seq_len, ceil(emb_dim/2)]
    pe = rearrange([evens.sin(), evens.clone().cos()], 't h w -> h (w t)') # interleave even and odd parts
      
    self.register_buffer('pe', pe) # [max_seq_len, emb_dim]
  
  def forward(self, 
              src: Tensor # [bsz, seq_len, emb_dim]
              ) -> Tensor:
    assert src.shape[-1] == self.pe.shape[1], f"Expected embedding dimension of {self.pe[1]} but got {src.shape[-1]} instead."
    out = src + self.pe[None,:src.size(1),:]
    return self.dropout(out) # See Page 7 of original paper, under section "Regularization"
```

Note that the  `self.register_buffer('pe', pe)` line is important because while the positional encodings do not have trainable parameters, this adds the encoding to the model's parameters and ensures that it is saved during `torch.save()`.
</details>
<br/>

### Multi-Head Attention 
In the paper, the authors use **Multi-Head Attention (MHA)**. In MHA we have multiple "heads" that each performs the attention computation that we *just* talked about. However, each head $h_i$ has its own linear projection matrices $K_i$, $Q_i$, and $V_i$, and these matrices project the key, query and value vectors to a **lower** dimensional space than we had with single matrices.

For example, if the dimension of matrix $K$ in **Single-Head Attention** was $512 \times 512$, then the dimension of $K_1$ and $K_2$ in a **2-Head Attention** would each be $512 \times 256$, thus projecting to a 256-dimensional space instead of 512-dimensional. 

After all the heads compute its own value of $\text{Attention}(X,Q_i,K_i,V_i)$ in parallel, we concatenate the outputs to obtain an output of the same shape as we had in the case of single-head attention. This is followed by a final linear projection to $d_{emb}$-dimensional space using weights $W_i^O$ where $d_{emb}$ is the embedding dimension. For our example input of shape `(9,512)`, MHA produces an output of the same shape.

<br/>
<figure align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="https://sarckk.github.io/media/transformer_mha.png"/>
<figcaption><strong>Above: </strong>Multi-head attention example with 3 heads</figcaption>
</figure>

The intuition behind why having multiple heads improves performance is that by having independently trainable linear projections per head, the model is able to simultaneously attend to different aspects of the language (for example, for a model trained on LaTeX documents, it might have one head that learns to attend to a presence of a `\end` command if a `\begin` command appears in the sequence, and another head that relates words in terms of their semantic relevance in text).

<details>
<summary>PyTorch implementation of Mult-Head Attention</summary>

```python
class MultiHeadAttention(nn.Module):
  def __init__(self,
               emb_dim: int,
               n_heads: int):
    super().__init__()
    assert emb_dim % n_heads == 0, "Embedding dimension must be divisble by number of heads"
    self.n_heads = n_heads
    self.emb_dim = emb_dim
    self.head_dim = emb_dim // n_heads
    # This projects each word vector into a new vector space (and we have n_heads amount of different vector spaces)
    self.weight_query = nn.Linear(self.emb_dim, self.emb_dim, bias=False) 
    self.weight_key = nn.Linear(self.emb_dim, self.emb_dim, bias=False) 
    self.weight_value = nn.Linear(self.emb_dim, self.emb_dim, bias=False) 
    self.out_project = nn.Linear(self.emb_dim, self.emb_dim)
  
  def forward(self, 
              query: Tensor, # (B, q_seq_len, D)
              key: Tensor, # (B, kv_seq_len, D)
              value: Tensor, # (B, kv_seq_len, D)
              mask: Optional[Tensor] = None, # (B, 1, 1, kv_seq_len) or (B, 1, q_seq_len, q_seq_len] where q_seq_len == kv_seq_len for self-attention
              ) -> Tensor:
    bsz, q_seq_len, _ = query.shape

    Q = self.weight_query(query)
    K = self.weight_key(key)
    V = self.weight_value(value) 
    Q,K,V = [x.view(bsz, -1, self.n_heads, self.head_dim).transpose(1,2) for x in (Q,K,V)]

    attn_weights = torch.einsum('bhqd,bhkd->bhqk',[Q,K]) # (B, h, q_seq_len, kv_seq_len] 
    attn_weights /= math.sqrt(self.head_dim) 

    if mask is not None:
      attn_weights += mask

    # softmax across last dim as it represents attention weight for each embedding vector in sequence
    softmax_attn = F.softmax(attn_weights, dim=-1) 
    out = torch.einsum('bhql,bhld->bhqd',[softmax_attn, V]) # (B, h, q_seq_len, D/h]
    out = out.transpose(1,2).reshape(bsz, -1, self.n_heads * self.head_dim) # (B, q_seq_len, D)
    return self.out_project(out)
```
</details>
<br/>

### Masking in the encoder 
The last important detail to mention at this point for the encoder is **masking**. Recall that in our input sequence, we used a special token for padding, `<pad>`. Because these are just dummy tokens added to ensure all sequences in a batch are of the same length, during attention computation we'd like to exclude the embedding vectors corresponding to these padding tokens from the weighted sum, by setting their weights to 0. 

To do this, we can't just set the weights in the corresponding positions to 0 *after* softmax, because then the weights will no longer sum to 1. Instead, we can apply a mask to the dot products **before** softmax such that after softmax, their values become 0 -- we do this by adding negative infinity ($-\infty$) to positions corresponding to the padding tokens:

<br/>
<figure align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="https://sarckk.github.io/media/transformer_masking.svg" width=500>
<figcaption>
<strong>Above: </strong>Example self-attention scores (pre-softmax) calculated for sentence "The weather is great today". Some scores are omitted. Weights that are highlighted red come from dot products with the &ltpad&gt token embedding vectors and thus needs to be set to 0 after applying softmax. This is achieved by setting these scores to negative infinity before softmax.
</figcaption>
</figure>
<br/>

In PyTorch, you can create a padding mask like so:

```python
def create_padding_mask(xs: Tensor, # (B, S)
                        pad_idx: int 
                        ) -> Tensor:
  batch_size, seq_len = xs.shape
  mask = torch.zeros(xs.shape).to(device)
  mask_indices = xs == pad_idx
  mask[mask_indices] = float('-inf')
  return mask.reshape(batch_size,1,1,seq_len) # (B, 1, 1, S)
```

The `create_padding_mask()` function takes a PyTorch tensor of shape `(B,S)` and the index of the padding token in vocabulary and returns a mask of shape `(B,1,1,S)`. As I established earlier in the article, `B` is the batch size and `S` is the sequence length. There are 2 additional dimensions in the output because of the way we apply the m ask in MHA:

```python
# attn_weights has shape (B, n_heads, query_seq_len, key_value_seq_len)
attn_weights += mask
```

Since `mask` has shape `(B,1,1,S)`, we [broadcast](https://pytorch.org/docs/stable/notes/broadcasting.html) across the 2nd and 3rd dimensions. The 2nd dimension broadcasts across the number of attention heads, while the 3rd dimension broadcasts the number of query vectors. While for self-attention, the number of query vectors equals the number of key and value vectors since they all get generated from the same source vectors, this isn't true in [**cross-attention**](), which we'll get to later when we look at the decoder. This is why we don't generate a padding mask of shape `(B,1,S,S)`, although we technically can for self-attention. 

### Feed-Forward Network 
Recall that self-attention is only the first part of a Transformer encoder. The issue with only having self-attention is that it is a linear transformation with respect to each element/position in a sequence; as we have seen, self-attention is basically a weighted sum (linear) where the weights are computed from dot products (also linear). And we know that nonlinearities are important in deep learning because it allows neural networks to approximate a wide range of functions (or all continous functions, as the [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) tells us).

So the creators of the Transformer introduce a fully connected feed-forward network after attention. This feed-forward network is applied **position-wise**, meaning it is applied to each element in the sequence independently. Therefore, in addition to introducing nonlinearities, these feed-forward networks can also be thought of as somehow "processing" the individual outputs in the sequence post-attention -- it does this by projecting the input into a higher dimension, applying nonlinearity, and projecting it back into the original dimension.[^1] In the paper, they use a 2-layer network with 1 hidden layer and ReLU activation as the nonlinearity. In PyTorch, this simply implemented as:

```python
feed_foward_net = nn.Sequential(
    nn.Linear(embedding_dimension, hidden_dimension),
    nn.ReLU(),
    nn.Linear(hidden_dimension, embedding_dimension),
) 
```

[^1]: It looks like the exact role that these feed-forward networks play in a transformer is not fully understood; see ["Transformer feed-forward layers are key-value memories." (Geva, Mor, et al., 2020)](https://aclanthology.org/2021.emnlp-main.446.pdf) for a paper that tries to shed light into their function.

In the paper, `hidden_dimension` is set to a value of 2048 (embedding dimension is 512 as mentioned earlier).

<figure align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="https://sarckk.github.io/media/transformer_ffn.svg">
<figcaption>Feed-forward network is applied position-wise. The illustration also shows the shape of the tensor at the stage indicated by the grey dotted lines.</figcaption>
</figure>

To recap: if we have a tensor of shape `(9,512)` at the start of the encoder, after passing through Multi-Head Attention, we get back an output of the same shape. When we pass this through the feed forward net as defined above, we basically pass each of the nine `512`-dimensional vectors through the neural network in parallel and join them together to get back a final output tensor of the same shape `(9,512)`. This works because the last dimension of the input tensor (512) is the same as the dimension of the input features of the network. Note that again, I excluded the batch dimension (i.e. in a mini-batch, the tensors will be of shape `(B,S,D)` instead of `(S,D)` that I have used in this example) because the same analysis holds even for bigger batch sizes.

### Encoder: the remaining bits
Here are the remaining details for the encoder:
- [Layer normalization](https://arxiv.org/abs/1607.06450) is applied to the output of each sublayer. Personally, I was confused by this initially because some illustrations of how layer norm works uses layer norm in the context of Computer Vision and Convolutional Neural Networks, which is slightly different from how it is used in Transformers (be careful, some explanations online confuse between the two as well). For this, I've found the following figure from the paper ["Leveraging Batch Normalization for Vision Transformers" (Yao, Zhuliang, et al., 2021)](https://openaccess.thecvf.com/content/ICCV2021W/NeurArch/papers/Yao_Leveraging_Batch_Normalization_for_Vision_Transformers_ICCVW_2021_paper.pdf) to be helpful in visualising the key difference between layer norm in CNN and in transformers:

</br>
<p align="center" width="100%">
<img src="https://sarckk.github.io/media/layernorm.png"/>
</p>

- Use of **residual connections** around both the self-attention and feed-forward network sublayers. First introduced in 2015 by the famous [ResNet paper](https://arxiv.org/abs/1512.03385), residual connections here basically means instead of the sublayer output being `f(x)`, it is `x + f(x)`, which helps with training by providing a gateway for gradients to pass through more easily during backprop.

---
Let's end this section by revisiting the encoder diagram from the paper:

<figure align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="https://sarckk.github.io/media/transformer_encoder_paper.png" width=250/>
<figcaption style="margin-top:2px;">Transformer encoder, taken from Figure 1 of the Transformer paper</figcaption>
</figure>

Everything shown in the diagram should be familiar to us by now. In particular, note how there are 3 arrows going into the Multi-Head Attention module: these represent key, query and values.

----
## Decoder 
Phew! There was quite a lot to cover for encoders. Fortunately, I've already covered most of the important parts of the Transformer -- the decoding part more or less mirrors what we had in the encoding phase, with a few key differences. 

<figure align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="https://sarckk.github.io/media/transformer_decoder_paper.png" width=250/>
<figcaption style="margin-top:2px;">Transformer decoder, taken from Figure 1 of the Transformer paper</figcaption>
</figure>

The input to the first decoder in the stack is a sequence of numerical representations of output tokens. The term "output" here might be a bit confusing. In the context of neural machine translation, the output here refers to tokens in the target language. Assuming that the target language is German and that we use a word-level tokenizer (i.e. each token is just a German word), then we can say that we pass in the sequence of indices of each German word in the sentence. The rest is the same as encoders: we generate an embedding vector and add positional encoding.

Also similar to the encoding phase, we have $N$ decoder modules, where $N=6$ for the base model in the original paper. Each decoder is similar to the encoder, except there are 2 differences: 
- In a decoder, there is an additional sublayer between self-attention and feed-forward network: **cross-attention**.
- An additional mask is used in the decoder to prevent "looking into the future" in self-attention.

### Difference #1: Cross-Attention 
Remember that in self-attention, we have query, key and value vectors used in attention computation coming from the same embedding vector. In cross-attention, we derive the query vector from one embedding vector and key and value vectors from a different vector. More specifically, in the decoder, the query vector comes from the output of the previous layer (i.e. for the very first decoder, this is the embedding layer; for subsequent decoders, it's the previous decoder), while the key and value vectors are generated from the output of the last encoder. Referring back to [Figure 1](https://sarckk.github.io/post/2023/04/10/grokking-transformers-wip/#DivingintoTransformers:thearchitecture) from the paper, this is illustrated with two arrows coming from the encoder to the cross-attention sublayer of the decoder.

<br/>
<figure align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="https://sarckk.github.io/media/decoder_cross_attention.png" width=450>
<figcaption>
<strong>Above: </strong>Adapted from Figure 1 of the Transformer paper. V and K are generated from the output of the last encoder, while Q is generated from the previous decoder self-attention sublayer.
</figcaption>
</figure>

Again, in the context of the machine translation task that the original Transformer was trained on, this step makes intuitive sense because for each word (or more generally, a token) in the target language sentence, we are essentially querying for the most relevant word(s) in the source language sentence and taking a weighted sum of their vector representations so that in the end we can predict the next word:

### Difference #2: Masking in self-attention 
The other difference between decoders and encoders is in the masking. Within the decoder, the masks used in self-attention and cross-attention are different, too.

**Masking in self-attention**

First, let's talk about masking in self-attention. In the decoding stage, the objective is: for each position, correctly predict what token should be at the next position. Therefore, unlike in the encoder, we can't access information about future positions to make a prediction for any given position in the sequence. This is especially true at inference time, where we do not have access to what comes after the current token by definition (otherwise we wouldn't need the model at all), but during training time, we do have information about the full translated sequence so we need to mask out, for each position, all the positions that come after it. It's best to illustrate this with a figure:

<figure align="center" style="display:flex; flex-direction: column; align-items: center;" id="lookahead-mask">
<img src="https://sarckk.github.io/media/transformer_lookahead_mask.svg" width=500>
<figcaption>
<strong>Above: </strong>Illustration of the look-ahead mask used in self-attention sublayer of the decoder. The weights that are highlighted red are set to negative infinity with the mask. After appplying softmax, these positions will be set to 0.
</figcaption>
</figure>

This is a mask where the upper triangular part are negative infinities. I don't think there's an official name for this mask, so I'll call it the **look-ahead** mask. Here's the code that generates this mask:

```python
def create_decoder_mask(seq_len: int) -> Tensor:
  mask = torch.zeros(seq_len, seq_len).to(device)
  mask_indices = torch.arange(seq_len)[None, :] > torch.arange(seq_len)[:, None] 
  mask[mask_indices] = float('-inf')  
  return mask.reshape(1,1,seq_len,seq_len) # (1, 1, S, S)
```

The function takes in the length of the target sequence and returns a mask with shape `(1,1,S,S)`. Recall that the self-attention mask for the **encoder** had shape `(B,1,1,S)`. In the decoder, I've set the first dimension as `1` for broadcasting, but it can very well be `B` as well. However, the third dimension has to be `S` and not `1`, since the mask used in the decoder is two-dimensional, and thus needed to be $S \times S$. 

When I was researching on how masking works in the decoder, the examples I could find also added a mask to exclude the padding tokens, as we did in the encoder. However, I don't think this is actually needed for self-attention in the decoder. Here's a brain dump of my reasoning:

> For any given position $i$, There are two possibilities: 
> 1. It *isn't* a padding token. Due to our look-ahead mask, we don't consider any tokens that comes afterwards in the weighted sum. Any tokens before it are necessarily not padding tokens because position $i$ isn't a padding token and we can't have a padding token that comes before a non-padding token.
> 2. It *is* a padding token. Again, we don't consider any tokens that comes afterwards. The positions before it might have padding tokens. So the output of attention at position $i$ would wrongly have included information from some padding tokens, but this doesn't matter because in the final loss calculation we ignore all positions with padding tokens (again, more on this later).

I haven't found a resource online that explicitly confirms this, so I could very well wrong -- if so, please let me know by submitting an issue [here](https://github.com/sarckk/sarckk.github.io).


**Masking in cross-attention** 

In cross-attention, because the key and value vectors that we're using to calculate dot products and calculate the weighted sum respectively come from the final output of the encoder, we don't need to mask future tokens, because all of this information should be available to us in the decoding stage. However, we still need to mask out the positions which correspond to padding tokens in the encoder's input like we did for self-attention in the encoder.

Recall the [shape of the padding mask](https://sarckk.github.io/post/2023/04/10/grokking-transformers-wip/#Maskingintheencoder), which was `(B,1,1,S)`, where `S` is the length of the source token sequence. Let's think about the shape of the dot product matrix, $QK^T$. Let `T` be the target token sequence length. Then, $QK^T$ will be a matrix $\in \mathbb{R}^{T \times S}$. When we consider multiple heads and batch size, then the result of our attention weights will be of shape `(B,n_heads,T,S)`. Using broadcasting, we just add the padding mask to these attention weights, and calculate the weighted sum.

That's it!

---
## Linear + Softmax Layer
As a final step, the output from the last decoder is passed to a linear layer that projects the embedding vectors to a dimension given by the vocabulary size of the target language, followed by a softmax layer to convert those values into probabilities that sum to 1. 

<br/>
<figure align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="https://sarckk.github.io/media/transformer_linear_softmax.png" width=250>
<figcaption><strong>Source: </strong>Figure 1 of the Transformer paper. </figcaption>
</figure>

For example, if we translating from English to German, and the dataset we're training on has a German vocabulary size of 37000, then the linear layer will take `emb_dimension` input features (e.g. 512) and have 37000 output features. After softmax, the value at $i^{th}$ dimension of the vector at $k^{th}$ position in the sequence will be probability for the $i^{th}$ word/token in the vocabulary in the $(k+1)^{th}$ position.

# Training 
Now let's talk about training. You can train Transformers on any sequence to sequence tasks but I'll talk about it in the context of machine translation, since this is what the original Transformer was trained for.

In a typical training dataset, like the WMT 2014 English-German dataset, you'll have pairs of (sentence in source langauge, same sentence in target language). As with most NLP tasks, first you'll use a tokenizer to split the sentences into their tokens, build up vocabulary of these tokens, and map the sequence of tokens into their corresponding indices in this vocabulary. Then we embed the tokens using the process [explained earlier](https://sarckk.github.io/post/2023/04/10/grokking-transformers-wip/#Divingdeeper:theembeddinglayer). This is all pretty standard stuff -- the thing I want to highlight is that the input to the decoder (i.e. target sequence) is shifted one to the right, which just means you exclude the last token. In code, you'd do something like:

```python
target_input = target[:,:-1]
```

To give a concrete example:
- The source sequence is `[<bos>, The, weather, today, is, great, <eos>, <pad>, <pad>]`
- The translated target sequence in German is `[<bos>, Das, Wetter, ist, heute, großartig, <eos>, <pad>, <pad>]`
- The input to the encoder is the embedding of `[<bos>, The, weather, today, is, great, <eos>, <pad>, <pad>]`
- The input to the decoder is the embedding of `[<bos>, Das, Wetter, ist, heute,großartig, <eos>, <pad>]` (shifted right)
- The decoder should predict `[Das, Wetter, ist, heute, großartig, <eos>, <pad>, <pad>]` (shifted left)

So the decoder learns to predict, at each position, the token that appears in the next position.

### Loss function
The paper doesn't explicilty mention what loss function is used, but you should be able to use any multi-class classification loss (which is what we're doing when predicting the most probable next token). The implementations I've seen seem to use either cross-entropy or KL divergence loss. In my own implementation, I've used cross-entropy loss. 

A mistake that cost me a lot of time debugging was how `nn.CrossEntropyLoss` works in PyTorch. In PyTorch, this module **performs softmax** before calculating the actual cross entropy loss -- it should really be named something like `nn.SoftmaxCrossEntropyLoss`! Because the figure of the Transformer architecture in the original paper has a softmax layer, this is what I originally implemented, and I was passing these normalized logits directly to `nn.CrossEntropyLoss`, causing issues during training: loss plateauing and my model quickly converging to producing the same tokens. In fairness, the PyTorch docs does mention that it expects an input that ["contains the unnormalized logits for each class (which do not need to be positive or sum to 1, in general)"](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) but who has time to read documentation, am I right? 😏

When calculating the loss, it's important to **ignore the loss contributed by the positions that correspond to the padding tokens**. Referring to <a href="#lookahead-mask">the look-ahead mask in the decoder</a>, the mask prevents the embeddings of padding tokens from being included in the weighted sum in attention in non-padding positions, but we nevertheless still compute the weighted sum for the padding positions. If this was the final output of the last decoder, after passing through the linear layer and thereafter taking softmax, we would have the next-token probabilities at each position, even where we had paddings! So we'd like to exclude these positions from our loss, since we don't really care about padding tokens anyway. In `nn.CrossEntropyLoss`, you can do this by passing the index of `<pad>` to the `ignore_index` argument:

```python
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
```

This loss also specifies a 10% label smoothing regularization -- let's talk about that next.

### Regularization
Section **5.4** of the paper mentions two regularization techniques used during training:

1. **Label smoothing** of 10% is also used in the loss calculation. The idea is simple: instead of the target distribution being one-hot (i.e. the "target" word has probability 1 and the rest of the words have 0), we set the probability of one word to be 0.9 and then distribute the other 0.1 over rest of the words in the vocabulary. This gives the model more flexibility in what token it predicts and presumably improves training. Intuitively, this kind of smoothing makes sense because with languages, there are often many plausible words that can come after some sequence of them.
2. [**Dropout**](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/) is used after each sublayer in the encoders, decoders, as well as to all the embeddings. They use a 10% dropout for the base model.

### Model hyperparameters
Model hyperparameters for the base Transformer model can be found on Table 3 of the paper. This model has around 65M trainable parameters -- comparing this number to the number of trainable params in your own implementation can be a good sanity check during development.

### Optimizer
Nothing special here, just an Adam optimizer with $\beta_1=0.9$, $\beta_2=0.98$ and $\epsilon=10^{-9}$.

### Learning-rate scheduling
The paper uses a variable learning rate during training, given by the following formula:

\begin{equation}
lrate = {d_{model}}^{-0.5} \cdot min(step\texttt{\_}num^{-0.5}, step\texttt{\_}num \cdot warmup\texttt{\_}steps^{-1.5})
\end{equation}

where $warmup\texttt{\_}steps=4000$.

The best way to understand how this works is to look at how the learning rate changes with step count:

<figure align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="https://raw.githubusercontent.com/gordicaleksa/pytorch-original-transformer/main/data/readme_pics/custom_learning_rate_schedule.PNG" width=500/>
<figcaption>Graph showing how learning rate changes as number of training steps increases, for different parameters. <strong>Source: </strong><a href="https://github.com/gordicaleksa/pytorch-original-transformer">https://github.com/gordicaleksa/pytorch-original-transformer</a> </figcaption>
</figure>

If you got the model and training steps right, you should be seeing a sweet training loss curve like this (I trained my model on translating German to English):

<p align="center">
<img src="https://sarckk.github.io/media/train_loss_transformer.svg" width=500/>
</p>

...and it's learning!

<p align="center">
<img src="https://sarckk.github.io/media/transformer_german_to_english.png"/>
</p>

# Inference
Now, let's talk about how the Transformer works at inference time for machine translation. Transformers are **auto-regressive**, meaning that they predict a  next token from all the previous tokens. At inference time, the only data available to us is the sentence(s) in the source language. From this, how do we generate the translated text using our trained model?

This is achieved by starting with the `<bos>` token, to mark the beginning of the translated sentence. Then, follow these steps:

1. Pass the source language sequence to the encoders and the target language sequence (which in the first iteration just includes the `<bos>` token) to the decoders. From the output of the Transformer we get a probability distribution over the vocabulary of the target language.
2. Choose the token with the highest probability and append this token to the target language sequence, giving us a longer sequence.

All we have to do now is to simply repeat steps **1** and **2** until the **last predicted token is `<eos>`, marking the end of sentence**. Viola! We just translated from one language to another.

Note that we still have to use the padding and the look-ahead masks just like we did for training, and in step 2 we would have to use a different look-ahead mask with each iteration since the target sequence length changes. However, more advanced implementations of Transformer can allow [only the last predicted token to be passed in on each iteration](https://datascience.stackexchange.com/questions/80826/transformer-masking-during-training-or-inference), in which case the look-ahead mask doesn't have to be passed in, since there is no "future" to consider.

# Conclusion
There it is -- a thorough walkthrough of the Transformer architecture. I mostly drew examples from English-to-German language translation task, but the generality of the architecture means that this can be adapted for any sequence-to-sequence transformations, such as but not limited to summarization, code generation and Q&A.

As I mentioned in the beginning of this article, I wrote this post mostly for myself, as a sort of recap of what I've learned. That said, even with so many tutorials online on this now-6-years-old technology, I still think what I have here might be useful to anyone who might across it, especially if they are somewhat new to deep learning; most tutorials I've seen on Transformers tend to gloss over some details especially around masking as well as how the encoder/decoder input vectors, and the key,query and value vectors in attention are generated. What I've attempted to do in this post is to document everything that I've had to learn and understand to train a simple Transformer model. I've already learned a lot writing this article, but I hope that it also helps someone out there.

--- 

# Acknowledgements
Here are some resources that I've used to learn about Transformers
- [Stanford CS224N Lecture 9 - Self-Attention and Transformers](https://www.youtube.com/watch?v=ptuGllU5SQQ). Probably the best lecture on Transformers online. This is the video that made attention and Transformers *click* for me.
- [PyTorch implementation of Transformer by Gordic Aleksa (AI Epiphany)](https://github.com/gordicaleksa/pytorch-original-transformer) and the accompanying [video tutorial](https://www.youtube.com/watch?v=n9sLZPLOxG8). I didn't refer to the implementation for the most part, but I did use it as reference to debug an issue I had with my implementation.

# Where to go from here
As an addendum, I'll just include some links to resources that I've come across that might serve as good next steps after understanding the Transformer. These are also things I'd like to read up in the near future:

- [A review of Transformer family of models](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
- [Mechanistic Interpretability research](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J)
  - Related: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [Kaparthy's GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [GPT in 160 lines of numpy](https://jaykmody.com/blog/gpt-from-scratch/)
  - Related: check out my previous blog post on [CNNs from scratch in numpy](https://sarckk.github.io/post/2022/03/20/convolutional-neural-networks-from-scratch/).
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
](https://arxiv.org/abs/1810.04805)
- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)
- [Flash Attention](https://arxiv.org/abs/2205.14135)