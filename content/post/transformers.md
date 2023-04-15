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
I'll skim over this step for now, because it only really makes sense when we start looking at how attention is computed. Without too much detail though, here we basically add a tensor with the same shape `(B,T,D)` to our tensor from **Step 2**. This tensor that we add encodes information about the relative order of each word in a sentence, since this is information that we'd like the Transformer to consider in the computation of the final output probabilities. Again, I'll come back to this later in the article.

In the end, the input to the Encoder is a tensor of shape `(B,T,D)`. The same thing happens for the Decoder, except with German sentences in our example of English-to-German translation. 

## Encoder
We've finally reached the Encoder. The encoding step consists of **N** independent Encoder units stacked on top of each other. These Encoders are identical in architecture, but do not share weights between them and are thus separately updated during backpropagation.

The Encoder unit itself comprises 2 parts:
1) A Self-Attention module, followed by
2) A Feed-Forward network. 

We'll start with this high level picture of the Encoder and gradually fill in more details:
<p align="center" width="100%">
<img src="https://sarckk.github.io/media/encoder_here.png" width=300/>
</p>

### Attention is all you need
The **core idea** behind the Transformer is to replace recurrence and convolutions that made up previous sequence-to-sequence models with one entirely based on the attention mechanism. In simple terms, the attention mechanism is basically just taking a bunch of dot products between sequences. And **self-attention** is just particular case of attention where the sequences that we're concerned with is actually all the same -- just one sequence.

Remember that at this point, our example input to the Encoder is a tensor of shape `(3,9,512)`. To make the explanation easier, let's look at what happens for **one** single sentence out of 3 total sentences in this mini-batch: when we actually pass through the entire batch with 3 sentences, logically it will be as if we pass through each sentence separately and merge the 3 outputs together.

Let's look at just one sentence: "This jacket is too small for me". After the embedding layer, we have a tensor of shape `(9,512)`. To encode this tensor, we essentially pass all 9 of the 512-dimensional embedding vector to the self-attention module **at the same time**:

![]()

The goal of the self-attention module is to figure out how the words in the sentence (or more generally, tokens in a sequence) relate to each other. 

Remember, the sentence is "This jacket is too small for me". When we look at the adjective `"small"`, we want to understand what object it is referring to. Clearly, we know that it is referring to the `"jacket"`, but the Transformer model has to learn this. In other words, it has to learn how each word relates to another. In vector space, we have a concept for computing the similarity between vectors: **dot product**. 

Dot products form the basis of the attention mechanism.

### Attention
Computing attention involves 3 inputs: query(s), keys(s), and value(s) where these are all vectors. More formally, we have:

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

$$\begin{aligned}
k_i = Kx_i, \text{where } K \text{ is a } d_k \times d_k \text{matrix} \\
q_i = Qx_i, \text{where } Q \text{ is a } d_k \times d_k \text{matrix} \\
v_i = Vx_i, \text{where } V \text{ is a } d_v \times d_v \text{matrix}
\end{aligned}$$

In our case, $d_k=d_v=512$. We have $K$, $Q$ and $V$ matrices (and therefore independently traininable) that linearly project each key, query and value vectors -- this allows for more flexibility in both how the model chooses to define "similarity" between words (by updating $K$ and $Q$), as well as what the final weighted sum represents (by updating $V$) in latent space. In Pytorch code, these matrices are implemented as [`nn.Linear()`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) modules with `bias=False`. 

Now that we have $k_i$, $q_i$ and $v_i$, we just compute the corresponding output for $x_i$ using the steps outlined earlier, computing the sum of all vectors weighed by the dot products. Here, since $q_i$,$k_i$ and $v_i$ are all derived from $x_i$, we give it a special name: **self-attention**. 

![Illustration of self-attention]()

### Attention, with matrices
As you can see, attention is computed using dot products between any two words within a sequence, allowing the Transformer to learn long-range dependencies in a sequence more easily. One downside of this, though, is that the computation of attention scores is quadratic in the length of the input sequence $N$. This quadratic $O(N^2)$ complexity is an issue because it means it will take a lot of compute for long sequences. 

Fortunately, we can represent the computation as a product of a few matrix multiplications, which is easily parallelizable on GPU/TPUs. Remember, our input tensor has shape `(9,512)` -- this is essentially a matrix of dimension 9 by 512. Given this matrix $X$ and the aforementioned linear projection matrices $K$, $Q$ and $V$, we can formulate the attention computation as follows:

\begin{equation}
 Attention(X, Q,K,V) = softmax((XQ)(XK)^T)XV
\end{equation}

The authors in the Transformers paper also apply a scaling factor of $\frac{1}{d_k}$ to the matrix of dot products (numerator) to prevent the products from becoming too large, which can "\[push\] the softmax function into regions where it has extremely small gradients" ([Viswani et al, 2017, pg 4](https://arxiv.org/pdf/1706.03762.pdf)):

\begin{equation}
 Attention(X,Q,K,V) = softmax(\frac{(XQ)(XK)^T}{\sqrt{d_k}})XV
\end{equation}

This ability for parallelization is a part of why the Transformer has been so successful -- previous models based on recurrence, for example, cannot be parallelized because the computation of its state at time $t$, $h_t$ necessarily depends on the computation of its previous state at time $t-1$, $h_{t-1}$.

### Back to Positional Encodings
Now we can finally talk about why we need positional encodings. We've seen that (self-)attention basically comes down to taking a bunch of dot products and outputting a new vector with this information. The problem is, by simply taking dot products, we lose information about the relative order of these words in a sentence. And we know that the position of a word in a sentence matters. 

To encode information about the position of each token in the sequence, we add **positional encodings** to the input embeddings. In practice, there are many ways to generate this -- including having the network learn this during training -- but the authors use the following formula:

$$\begin{aligned}
PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{emb}}) \\
PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{emb}})
\end{aligned}$$

where $i$ is the index along the embedding dimension and $pos$ is the position of the token in the sequence. Both are 0-indexed. By having sine and cosine functions of varying periods, we are able to inject information about position in continuous form. 

<p align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="https://sarckk.github.io/media/pos_encoding.png" width=400>
<caption>Illustration of positional encodings</caption>
</p>

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

### Masking
Another important detail to mention at this point is masking.   

### Multi-Head Attention
In the paper, the authors use **Multi-Head Attention (MHA)**. In MHA we have multiple "heads" that each performs the attention computation that we *just* talked about. However, each head $h_i$ has its own linear projection matrices $K_i$, $Q_i$, and $V_i$, and these matrices project the key, query and value vectors to a **lower** dimensional space than we had with single matrices.

For example, if the dimension of matrix $K$ in **Single-Head Attention** was $512 \times 512$, then the dimension of $K_1$ and $K_2$ in a **2-Head Attention** would each be $512 \times 256$, thus projecting to a 256-dimensional space instead of 512-dimensional. 

After all the heads compute its own value of `Attention(X,Q_i,K_i,V_i)` in parallel, we concatenate the outputs to obtain an output of the same shape as we had in the case of single-head attention. This is followed by a final linear projection to $d_{emb}$-dimensional space where $d_{emb}$ is the embedding dimension. For our example input of shape `(9,512)`, MHA produces an output of the same shape.

The intuition behind why having multiple heads improves performance is that by having independently trainable linear projections per head, the model is able to simultaneously attend to different aspects of the language (for example, for a model trained on LaTeX documents, it might have one head that learns to attend to a presence of a `\end` command if a `\begin` command appears in the sequence, and another head that relates words in terms of their semantic relevance in text).

### Feed-Forward Network
Recall that self-attention is only the first part of a Transformer Encoder. The issue with only having self-attention is that it is a linear transformation with respect to each element/position in a sequence; as we have seen, self-attention is basically a weighted sum (linear) where the weights are computed from dot products (also linear). And we know that nonlinearities are important in deep learning because it allows neural networks to approximate a wide range of functions (or all continous functions, as the [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) tells us).

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

To recap: if we have a tensor of shape `(9,512)` at the start of the Encoder, after passing through Multi-Head Attention, we get back an output of the same shape. When we pass this through the feed forward net as defined above, we basically pass each of the nine `512`-dimensional vectors through the neural network in parallel and join them together to get back a final output tensor of the same shape `(9,512)`. This works because the last dimension of the input tensor (512) is the same as the dimension of the input features of the network. Note that again, I excluded the batch dimension (i.e. in a mini-batch, the tensors will be of shape `(B,T,D)` instead of `(T,D)` that I have used in this example) because the same analysis holds even for bigger batch sizes.

### Encoder: the remaining bits
Here are the remaining details for the Encoder:
- Use of **residual connections** around both the self-attention and feed-forward network sublayers. First introduced in 2015 by the famous [ResNet paper](https://arxiv.org/abs/1512.03385), residual connections here basically means instead of the sublayer output being `f(x)`, it is `x + f(x)`, which helps with training by providing a gateway for gradients to pass through more easily during backprop.
- [Layer normalization](https://arxiv.org/abs/1607.06450) is applied to the output of each sublayer. Personally, I was confused by this initially because some illustrations of how layer norm works uses layer norm in the context of Computer Vision and Convolutional Neural Networks, which is slightly different from how it is used in Transformers (be careful, some explanations online confuse between the two as well). For this, I've found the following figure from the paper ["Leveraging Batch Normalization for Vision Transformers" (Yao, Zhuliang, et al., 2021)](https://openaccess.thecvf.com/content/ICCV2021W/NeurArch/papers/Yao_Leveraging_Batch_Normalization_for_Vision_Transformers_ICCVW_2021_paper.pdf) to be helpful in visualising the key difference between layer norm in CNN and in transformers:

<p align="center" width="100%">
<img src="https://sarckk.github.io/media/layernorm.png"/>
</p>

--- 
Let's end this section by revisiting the Encoder diagram from the paper:
<p align="center" width="100%">
<img src="https://sarckk.github.io/media/encoder_here.png"/>
</p>

## Decoder
Phew! There was quite a lot to cover for Encoders. Fortunately, I've already covered most of the important parts of the Transformer -- the decoding part more or less mirrors what we had in the encoding phase, with a few key differences. 

The input to the first decoder in the stack is a sequence of numerical representations of output tokens. The term "output" here might be a bit confusing. In the context of neural machine translation, the output here refers to tokens in the target language. Assuming that the target language is German and that we use a word-level tokenizer (i.e. each token is just a German word), then we can say that we pass in the sequence of indices of each German word in the sentence. The rest is the same as encoders: we generate an embedding vector and add positional encoding.

Also similar to the encoding phase, we have $N$ Decoder modules, where $N=6$ for the base model in the original paper. Each Decoder is similar to the Encoder, except there are 2 differences: 
- In a decoder, there is an additional sublayer between self-attention and feed-forward network: **cross-attention**.
- An additional mask is used in the decoder to prevent "looking into the future" in self-attention.

### Difference #1: Cross-Attention
Remember that in self-attention, we have query, key and value vectors used in attention computation coming from the same embedding vector. In cross-attention, we derive the query vector from one embedding vector and key and value vectors from a different vector. More specifically, in the decoder, the query vector comee from the output of the previous layer (i.e. for the very first decoder, this is the embedding layer; for subsequent decoders, it's the previous decoder), while the key and value vectors are generated from the output of the last encoder. Referring back to [Figure 1]() from the paper, this is illustrated with two arrows coming from the encoder to the cross-attention sublayer of the decoder.

Again, in the context of the machine translation task that the original Transformer was trained on, this step makes intuitive sense because for each word (or more generally, a token) in the target language sentence, we are essentially querying for the most relevant word(s) in the source language sentence and taking a weighted sum of their vector representations so that in the end we can predict the next word (more on training soon):

<p align="center" style="display:flex; flex-direction: column; align-items: center;">
<img src="" width=400>
<caption>Illustration of the concept here</caption>
</p>

### Difference #2: Masking in self-attention
The other difference between decoders and encoders is in the masking used in self-attention. Recall that in an encoder, masking was used to prevent  

## Linear + Softmax Layer
As a final step, the output from the last decoder is passed to a linear layer that projects the embedding vectors to a dimension given by the vocabulary size of the target language, followed by a softmax layer to convert those values into probabilities that sum to 1. For example, if we translating from English to German, and the dataset we're training on has a German vocabulary size of 37000, then the linear layer will take `emb_dimension` input features (e.g. 512) and have 37000 output features. After softmax, the value at $i^{th}$ dimension of the vector at $k^{th}$ position in the sequence will be probability for the $i^{th}$ word/token in the vocabulary in the $(k+1)^{th}$ position.

That concludes the section on the Transformer architecture. 

# Training 

# Conclusion
TODO

The code for the article can be found on my [github](https://github.com/sarckk/transformer_series/blob/main/transformers.ipynb). My aim was to get the training working as quickly as possible so it's far from polished.



