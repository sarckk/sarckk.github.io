---
title: Grokking Transformers (WIP)
date: '2023-04-10'
categories:
  - AI
tags:
  - AI
  - LLM
---

Today's Large Language Models (LLM) are based on Transformers, a deep learning model architecture for sequence-to-sequence transformations based on the attention mechanism. While it was originally proposed and used in Natural Language Processing (NLP) tasks like language translation, it turns out that a lot of things that we care about can be modelled in terms of sequences, making transformers a useful model in a wide variety of applications beyond NLP, such as [image processing]() and [reinforcement learning](). Given the overwhelming success of transformers in deep learning and the outsized impact that transformer-based generative AI (e.g. GPT) has had -- and will likely continue to have -- on our society, I thought I should finally take time to read and understand the paper ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) that first proposed Transformers. That paper is now almost 6 years old(!) but better late than never, right?

There are already many writings covering transformers on the internet, so this article is mostly for my own learning -- this is already [well documented](https://ideas.time.com/2011/11/30/the-protege-effect/), but I find that writing in a pedagogical style helps immensely in solidifying in my learnings and is almost always worth the effort. If anyone else stumbles across this post and finds it helpful, that's an added bonus!

This post will cover the technical details behind the Transformer model. The core concept behind transformers -- self- and cross- attention -- really isn't too hard too grasp, but I've found that actually getting your hands dirty and implementing the model in Pytorch elevates your understanding of the material. Personally, I ran into many issues while trying to write and train the model that I wouldn't have known had I stopped at reading the paper or other tutorials online.

## All about transformations
A Transformer -- as its name suggests -- *transforms* an input sequence $(x_1,x_2,...,x_n)$ into an output sequence $(y_1,y_2,...,y_m)$. Because this formulation is so general, it doesn't say what $x_1$ and $y_1$ should represent -- it could be a word, a sub-word, a character, a pixel, or a token representing any arbitrary thing. However, I'll be talking about Transformers in the context of NLP here, because that's what it was originally invented for. So if we're talking about machine translation, the input sequence could be a sequence of words in one language (e.g. Korean) and the output could be a sequence of words in the target language (e.g. English):

![](https://sarckk.github.io/media/transformer_1.svg)

In the diagram above, each element in a sequence represents a word for simplicity, but in practice, it is common for this to a smaller unit than a word, like a subword for example. This depends on the tokenizer you use. 

## Diving into Transformers
Now let's talk about what a Transformer actually looks like. From the original paper:

<p align="center" width="100%">
<img src="https://sarckk.github.io/media/transformer_architecture.png" width=450/>
</p>

If you're like me, this might be a bit overwhelming to take in at first. In reality, there's only 4 major components to a transformer architecture: the embedding layer, the encoder, the decoder, and the final linear+softmax layers that transform the output of the decoder into probabilities. Here's the same diagram with some annotations overlaid on top:

<p align="center" width="100%">
<img src="https://sarckk.github.io/media/transformer_arch_illustrated.png" width=400/>
</p>

At a high level, here's the journey that our input sequence takes to be transformed into an output sequence:
1) Go through input embedding layer which projects each element in a sequence $x_i$ into a higher dimensional vector. 
2) Add "Positional Encoding" vector to each element in the sequence (which remember, is now a high-dimensional vector). We'll talk about this in more detail later.
3) Go through the Encoder (orange block in the diagram above) **N** times. These **N** encoders have the same architecture but do not share weights.At the end of this step, we get a tensor that compactly represents the input sequence. 
4) On the decoder side, we pass in a sequence of length **M**, which goes through the same embedding layer + positional encoding as we had for the input sequence.  
5) The output embedding goes through a stack of **N** decoders (again no sharing of weights), each of which uses the tensor we got from **Step 3** in some way. At the end of this step, we get another tensor.
6) We pass this tensor through a final Linear + Softmax layer to obtain **M** probabilities, where again **M** is the length of the sequence we passed into the decoder.
7) We convert those probabilities into actual tokens (e.g. by taking the token with the highest probability), giving us an output sequence of length **M**.

So from an input sequence of length **N** and decoder input of length **M**, we got -- as the final product of the Transformer -- another sequence of length **M**.

It's okay if some of these steps do not make sense yet, especially on why we need a sequence of length **M** to pass into the decoder to get another sequence of the same length as the output. I was initially confused by this as well: if we are just passing in an input sentence like "" and expect the model to output the translated text, what are we passing into the decoder? WTF? Don't worry, this will become clear when we talk about Transformers at training and inference time, later in this article.

Now, let's talk about each of these components in greater detail during **training**. Then we'll talk about what happens at **inference time**.

## Diving deeper: the embedding layer
Let's start from the very beginning. The original Transformer in the 2017 paper was trained on the task of translating English sentences to German, using the WMT 2014 English-German dataset. This dataset contains ~4.5 million pairs of English sentence and its corresponding translation in German. We'll use this example to explain what happens in a Transformer for the rest of the article.

This means that during training, the input to the transformer (bottom left of Figure 1 above, below the very first decoder) is a sequence(s) of tokens in the English sentence. The paper mentions that they used [byte-pair](https://en.wikipedia.org/wiki/Byte_pair_encoding) encoding (page 7) to tokenize the sentences, but for simplicity I'll assume that the sentences are tokenized word by word:

![](https://sarckk.github.io/media/seq_len_input.svg)

Above is an illustration of what this would look like for a mini-batch of 3 English sentences. Note that we have 3 special tokens: `<BOS>` denoting the beginning of sentence, `<EOS>` marking the end of a sentence, and `<PAD>` representing an "empty" token to make all the sequences in the tensor of the same length (i.e. the maximum sequence length across all sentences in the mini-batch). 

### Step 1: String -> Number
Before we can pass this into the encoder, we need to convert these sequences of strings into a numerical representation instead so we can do some computation and make GPUs go *brrr*. To do this, we create -- from the dataset -- a mapping from all possible tokens (in this case, words) to its index in a vocabulary. So, the word "The" might have an index of 37, which uniquely identifies that token. We also leave out some indices for the special tokens that we introduced (`<BOS>`, `<EOS>` and `<PAD>`): for example, their indices might be 0, 1 and 2 respectively.

### Step 2: Number -> Embedding Vector
Once we have the input tensor, we turn each word (which by this point is a number representing its index in our vocabulary) into a high-dimensional vector that we call an **embedding vector**. In the original paper, the embedding vector is **512-dimensional**, but this is a hyperparameter that we can tune for our model through experiments. 


After this step, we have a tensor of shape (batch size, (max) input sequence length, embedding dimension). For the rest of the article, I'll use the short form `B` for batch size, `T` for the maximum input sequence length, and `D` for the embedding dimension. In our example, `B=3`, `T=9` and `D=512`.

### Step 3: Adding Positional Encoding
We'll actually skip this step for now, because it only really makes sense when we start looking at the Encoder and self-attention. Without too much detail, though, here we basically add a tensor with the same shape `(B,T,D)` to our tensor from **Step 2**. This tensor that we add encodes information about the relative order of each word in a sentence, since this is information that we'd like the Transformer to consider in the computation of the final output probabilities. We'll come back to this shortly.

In the end, the input to the Encoder is a tensor of shape `(B,T,D)`. The same thing happens for the Decoder, except with German sentences in our example of English-to-German translation. 

## Encoder
We've finally reached the Encoders. The encoding step consists of **N** independent Encoder units stacked on top of each other. These Encoders are identical in architecture, but do not share weights between them and are thus separately updated during backpropagation.

Let's zoom in to see what it's made of:

<p align="center" width="100%">
<img src="https://sarckk.github.io/media/encoder_here.png" width=300/>
</p>

The Encoder unit itself comprises 2 parts:
1) First, Self-attention with multiple heads (Mult-Head Attention). This is followed by  
2) A Feed Forward network. 

### Attention is all you need
The **core idea** behind the Transformer is to replace recurrence and convolutions that made up previous sequence-to-sequence models with one entirely based on the attention mechanism. In simple terms, the attention mechanism is basically just taking a bunch of dot products between sequences. And **self-attention** is just particular case of attention where the sequences that we're concerned with is actually all the same -- just one sequence.

Remember that at this point, our example input to the Encoder is a tensor of shape `(3,9,512)`. To make the explanation easier, let's look at what happens for **one** single sentence out of 3 total sentences in this mini-batch: when we actually pass through the entire batch with 3 sentences, logically it will be as if we pass through each sentence separately and merge the 3 outputs together.

Let's look at just one sentence: "This jacket is too small for me". After the embedding layer, we have a tensor of shape `(9,512)`. To encode this tensor, we essentially pass all 9 of the 512-dimensional embedding vector to the self-attention module **at the same time**:

![]()

The goal of the self-attention module is to figure out how the words in the sentence (or more generally, tokens in a sequence) relate to each other. 

Remember, the sentence is "This jacket is too small for me". When we look at the adjective `"small"`, we want to understand what object it is referring to. Clearly, we know that it is referring to the `"jacket"`, but the Transformer model has to learn this. In other words, it has to learn how each word relates to another. In vector space, we have a concept for computing the similarity between vectors: **dot product**. 

Dot products form the basis of the attention mechanism.

### The Attention mechanism
The attention mechanism involves 3 components: query(s), keys(s), and value(s) where these are all vectors. More formally, we have:

- Queries $q_1,...,q_T$ where $q_i \in$ $\R^{d_k}$, where $d_k$ is the dimension of the query vector, and $T$ is the number of queries
- Keys $k_1,...k_K$ where $k_i \in$ $\R^{d_k}$, and $K$ is the number of key-value pairs
- Value $v_1,...,v_K$ where $v_i \in$ $\R^{d_v}$, where $d_v$ is the dimension of the value vector, which is not necessarily equal to $Q$, although in our case of English-German translation, it is.

Note that $T$, the number of queries doesn't necessarily have to equal $K$, the number of key-value pairs, but the number of keys must be the same as the number of values (for them to form a key-value pair). Furthermore, the query and key vectors must have the same dimension, $d_k$ so we can do a dot product.

Given this formulation, the attention function on $q_i$ does the following:
- Get dot product of query and key vectors to get a scalar value: $\alpha_{ij} = q_i \cdot k_i$
- Normalize each dot product $\alpha_{ij}$ by performing [softmax](https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax) across all $j$, where $j=0,...,K$ , where $K$ is the number of key-value pairs. This gives us weights $w_{ij}$ for all $j$.
- Output weighted sum of all $v_j$ where weight of $v_j$ is $w_{ij}$.


### Back to our example
Back to our example sentence, we have $x_1,...,x_9$ where $x_i$ is a 512-dimensional embedding vector representing each word in the sentence `"This jacket is too small for me"` plus the `<BOS>` and `<EOS>` tokens. We obtain our query, key and value vectors from $x_i$ by multiplying it each time with a different matrix:

\begin{equation}
k_i = Kx_i, where K is a d_k \times d_k matrix.
\end{equation}
\begin{equation}
q_i = Qx_i, where Q is a d_k \times d_k matrix.
\end{equation}
\begin{equation}
v_i = Vx_i, where V is a d_v \times d_v matrix.
\end{equation}

In our case, $d_k=d_v=512$. It's important that $K$, $Q$ and $V$ are separate matrices (and therefore independently traininable) because this allows for more flexibility in both how the model chooses to define "similarity" between words (by updating $K$ and $Q$), as well as what the final weighted sum represents (by updating $V$) in latent space. In Pytorch code, these matrices are implemented as `nn.Linear()` modules with `bias=False`. 

Now that we have $k_i$, $q_i$ and $v_i$, we just compute the corresponding output for $x_i$ using the steps outlined earlier, computing the sum of all vectors weighed by the dot products. Here, since $q_i$,$k_i$ and $v_i$ are all derived from $x_i$, we give it a special name: **self-attention**. 

![Illustratiotion of self-attention]()

### Parallelizing attention computation
As you can see, attention is computed using dot products between any two words within a sequence, allowing the Transformer to learn long-range dependencies in a sequence more easily. One downside of this, though, is that the computation of attention scores is quadratic in the length of the input sequence $N$. This quadratic $O(N^2)$ complexity is an issue because it means it will take a lot of compute for long sequences. 

Fortunately, we can represent the computation as a product of a few matrix multiplications, which is easily parallelizable on GPU/TPUs.

In matrix form, we can formulate attention as the interaction between $Q$, $K$ and $V$ matrices for queries, keys and values respectively.

In short, we can formulate the attention mechanism described above as:

\begin{equation}
 Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
\end{equation}

This is the standard matrix formulation of attention that the authors use in the Transformers paper (page 4). Notice that we are scaling the matrix ${QK_T}$ (i.e. the dot-product between all rows of query-matrix $Q$ with all columns of key-matrix $K$) by $\frac{1}{d_k}$, to prevent dot products from becoming too large, "pushing the softmax function into regions where it has extremely small gradients" ([Viswani et al, 2017, pg 4](https://arxiv.org/pdf/1706.03762.pdf)).

This ability for parallelization is a part of why the Transformer has been so successful -- previous models based on recurrence, for example, cannot be parallelized because the computation of its state at time $t$, $h_t$ necessarily depends on the computation of its previous state at time $t-1$, $h_{t-1}$.

### Multiple Heads
Let's talk about multi-head attention -- why the need for multiple heads?

### Back to Positional Encodings
Now we can finally talk about why we need positional encodings. We've seen that (self-)attention basically comes down to taking a bunch of dot products and outputting a new vector with this information. The problem is, by simply taking dot products, we lose information about the relative order of these words in a sentence. We know that the position of a word in a sentence matters a great deal. For example, consider the sentence:

and the sentence:

These are similar but the relative order of the word `""` completely changes the meaning. We'd like the Transformer to 

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

  def check_and_plot_pos_encoding(self): 
    # modified from tensorflow example (https://www.tensorflow.org/text/tutorials/transformer#positional_encoding)

    # Check the shape.
    print(self.pe.numpy().shape)

    # Plot the dimensions.
    plt.pcolormesh(self.pe.numpy(), cmap='RdBu')
    plt.gca().invert_yaxis()
    plt.xlabel('Depth')
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
```

Note that the  `self.register_buffer('pe', pe)` line is important because while the positional encodings do not have trainable parameters, this adds the encoding to the model's parameters and ensures that it is saved during `torch.save()`.
</details>

### Masking
TODO

### Feed Forward Networks
Adding nonlinearity, TODO

### Residual Connections + Layer Normalization
TODO

## Decoder
Phew! There was quite a lot to cover for Encoders. Fortunately, I've already covered most of the important parts of the Transformer, and the decoding part more or less mimics what we had in the encoding phase.

Similar to encoding, we have $N$ Decoder modules -- all independently trained. Each Decoder is similar to the Encoder, except we have an additional stage of Attention computation + feed forward network. The difference is that this attention is called **cross-attention**, where we use the final encoder output (remember this is a tensor of shape `(B,T,D)`) as the key and value tensors for attention, while the queries come from the decoder itself (input for the first decoder, and the output of the previous decoder for subsequent decoders).

### Cross-Attention
Cross-attention is simlar, except we use (TODO)

### Masking
Masking is slightly different for Decoders, too. Instead of... 

## Conclusion
TODO

The code for the article can be found on my [github](https://github.com/sarckk/transformer_series/blob/main/transformers.ipynb). My aim was to get the training working as quickly as possible so it's far from polished.


