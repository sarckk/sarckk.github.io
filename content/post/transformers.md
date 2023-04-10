---
title: Grokking Transformers
date: '2023-04-10'
categories:
  - AI
tags:
  - AI
  - LLM
---

Today's Large Language Models (LLM) are based on Transformers, a deep learning model architecture for sequence-to-sequence transformations based on the self-attention mechanism. While it was originally proposed and used in Natural Language Processing (NLP) tasks like language translation, it turns out that a lot of things that we care about can be modelled in terms of sequences, making transformers a useful model in a wide variety of applications beyond NLP, such as [image processing]() and [reinforcement learning](). Given the overwhelming success of transformers in deep learning and the outsized impact that transformer-based generative AI (e.g. GPT) has had -- and will likely continue to have -- on our society, I thought I should finally take time to read and understand the paper ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) that first proposed Transformers. That paper is now almost 6 years old(!) but better late than never, right?

There are already many writings covering transformers on the internet, so this article is mostly for my own learning -- this is already [well documented](https://ideas.time.com/2011/11/30/the-protege-effect/), but I find that writing in a pedagogical style helps immensely in solidifying in my learnings and is almost always worth the effort. If anyone else stumbles across this post and finds it helpful, that's an added bonus!

This post will cover the technical details behind the Transformer model. The core concept behind transformers -- self- and cross- attention -- really isn't too hard too grasp, but I've found that actually getting your hands dirty and implementing the model in Pytorch elevates your understanding of the material. Personally, I ran into many issues while trying to write and train the model that I wouldn't have known had I stopped at reading the paper or other tutorials online.

## All about transformations
A Transformer -- as its name suggests -- *transforms* an input sequence $(x_1,x_2,...,x_n)$ into an output sequence $(y_1,y_2,...,y_m)$. Because this formulation is so general, it doesn't say what $x_1$ and $y_1$ should represent -- it could be a word, a sub-word, a character, a pixel, or a token representing any arbitrary thing. However, I'll be talking about Transformers in the context of NLP here, because that's what it was originally invented for. So if we're talking about machine translation, the input sequence could be a sequence of words in one language (e.g. Korean) and the output could be a sequence of words in the target language (e.g. English):

![](https://sarckk.github.io/media/transformer_1.svg)

In the diagram above, each element in a sequence represents a word for simplicity, but in practice, it is common for this to a smaller unit than a word, like a subword for example. This depends on the tokenizer you use. 

## Diving into Transformers
Now let's talk about what a Transformer actually looks like. From the original paper:

<img src="https://sarckk.github.io/media/transformer_architecture.png" width=350/>

If you're like me, this might be a bit overwhelming to take in at first. In reality, there's only 4 major components to a transformer architecture: the embedding layer, the encoder, the decoder, and the final linear+softmax layers that transform the output of the decoder into probabilities. Here's the same diagram with some annotations overlaid on top:

<img src="https://sarckk.github.io/media/transformer_arch_illustrated.png" width=400/>


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

<img src="https://sarckk.github.io/media/encoder_here.png" width=300/>

The Encoder unit itself comprises 2 parts:
1) First, Self-attention with multiple heads (Mult-Head Attention) 
2) Followed by a Feed Forward network. 

### Self-attention
This is the core idea behind Transformers -- to replace recurrence and convolutions that made up previous sequence-to-sequence models with one entirely based on the attention mechanism. In simple terms, the attention mechanism is basically just taking a bunch of dot products.


#### Multiple Heads


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

## Conclusion
The code for the article can be found on my [github](https://github.com/sarckk/transformer_series/blob/main/transformers.ipynb). My aim was to get the training working as quickly as possible so it's far from polished.



