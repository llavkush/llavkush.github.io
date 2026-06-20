*Self-attention is the one engine inside every Transformer. Here it is built from nothing but lists and a square root, with output you can read, then the three-letter version (Q, K, V) the papers use.*

**7 minute read**

---

In [the last post](post.html?p=why-transformers) we said a Transformer is a group chat: every word reads every other word at once. That post answered *why*. This one answers *how*. By the end you'll have built the actual machine, **self-attention**, in plain Python, and watched it figure out that "cat" and "mat" belong together.

No PyTorch. No NumPy required. Just lists, a loop, and one square root.

---

## Situation

People throw around three scary letters: **Q, K, V** (query, key, value). They sound like linear-algebra homework. They're not. The whole thing is one everyday idea: **a search, but instead of one winner, you blend all the results by how well they match.**

Think of how a normal database works versus how attention works:

- **A database:** you ask for `id = 7`, you get *exactly* row 7. One exact match, one answer.
- **Attention:** you ask "anything *like* this?", every row says how similar it is, and you get a **weighted blend** of all of them, more from the close matches, less from the far ones.

That softer "blend by similarity" is the entire trick. Let's earn each letter.

---

## Task

By the end you should be able to say, plainly:

- What are **query, key, value**, in words a non-coder gets?
- How does attention turn "how similar?" into actual numbers that sum to 1?
- Why is there a mysterious **divide by √(dk)**, and what breaks without it?
- How is this the same machine that powers GPT and Claude?

---

## Action: the dating-app analogy for Q, K, V

Every word in the sentence plays three roles at once. Picture a dating app:

- **Query (Q)** = your *dating profile of what you want*: "I'm looking for someone funny and outdoorsy."
- **Key (K)** = each person's *advertised label*: "I am funny and outdoorsy."
- **Value (V)** = what you *actually get* if you match with them: their real self, the content you take away.

Attention is: compare **your query** against **everyone's key** to get match scores, turn those scores into percentages, then blend **everyone's value** by those percentages. A word doesn't pick one match, it takes a weighted smoothie of everybody, heavy on the best matches.

In a Transformer, every word does this *simultaneously*, comparing its query against every word's key, including its own.

---

## Action: match scores are just dot products

How do we measure "how well does this query match this key"? The same **dot product** from the [linear algebra note](post.html?p=linear-algebra-intuition): multiply the matching numbers and add them up. Big number means "pointing the same way", means similar.

```python
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

query = [1, 0, 1]     # "I want traits A and C"
keyA  = [1, 0, 1]     # advertises A and C  -> strong match
keyB  = [0, 1, 0]     # advertises B        -> no overlap

print(dot(query, keyA))   # 2  (high score)
print(dot(query, keyB))   # 0  (no match)
```

Do that for every key and you get a row of scores, one per word. That row is "how much should I care about each word?"  but as raw numbers, not yet percentages.

---

## Action: softmax turns scores into percentages

Raw scores like `[2.1, 0.3, 0.1, 0.8]` aren't usable as "how much attention." We want them to be positive and **sum to 1**, like splitting 100% of your attention across the words. That's exactly what **softmax** does: exponentiate each score (so bigger wins by more) and divide by the total.

```python
from math import exp

def softmax(row):
    m = max(row)                       # subtract the max first (see note below)
    e = [exp(x - m) for x in row]
    s = sum(e)
    return [x / s for x in e]

print([round(p, 3) for p in softmax([2.1, 0.3, 0.1, 0.8])])
# [0.636, 0.105, 0.086, 0.173]  -> sums to 1.0, biggest score gets the most
```

That `- max(row)` line matters. Exponentials explode fast; `exp(1000)` overflows. Subtracting the largest value first keeps the math in a safe range and gives the **identical** answer. Watch it survive absurd inputs:

```python
print([round(p, 3) for p in softmax([100, 200, 300])])
# [0.0, 0.0, 1.0]  -> no crash, no overflow
```

---

## Action: the famous formula, decoded

Put the pieces together and you get the one line every Transformer paper prints:

```text
Attention(Q, K, V) = softmax( Q · Kᵀ / √dk ) · V
```

Read left to right it's just our story:

1. `Q · Kᵀ`: every query dotted with every key, giving an N×N grid of match scores.
2. `/ √dk`: shrink the scores (explained next).
3. `softmax(...)`: turn each row into percentages that sum to 1.
4. `· V`: blend the values by those percentages.

And the **/ √dk** part: when vectors are long (say 64 numbers), dot products get *big*, which shoves softmax into a corner where one word grabs ~100% and the rest get ~0%. That kills learning (the gradients vanish). Dividing by the square root of the vector length (`dk`) keeps scores in a gentle range. It's a volume knob so attention doesn't scream.

---

## Action: the whole engine in pure Python

Here is scaled dot-product self-attention, complete, with no libraries. Then we run it on a real sentence and *read the result*.

```python
from math import exp, sqrt

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def transpose(M):
    return [list(col) for col in zip(*M)]

def matmul(A, B):                       # (n×k) times (k×m) -> (n×m)
    Bt = transpose(B)
    return [[dot(row, col) for col in Bt] for row in A]

def softmax_row(row):
    m = max(row)
    e = [exp(x - m) for x in row]
    s = sum(e)
    return [x / s for x in e]

def attention(Q, K, V):
    dk = len(Q[0])
    scores = matmul(Q, transpose(K))                       # 1. Q · Kᵀ
    scores = [[s / sqrt(dk) for s in row] for row in scores] # 2. / √dk
    weights = [softmax_row(row) for row in scores]          # 3. softmax
    output  = matmul(weights, V)                            # 4. · V
    return output, weights

# A real sentence. Each word is a tiny 4-number embedding.
# (Hand-picked so the relationships are readable: "cat" and "mat" share a vector,
#  and the two "the"s are identical.)
sentence = ["the", "cat", "sat", "on", "the", "mat"]
embed = {
    "the": [1, 0, 0, 0],
    "cat": [0, 1, 0, 1],
    "sat": [0, 0, 1, 0],
    "on":  [0, 0, 0, 1],
    "mat": [0, 1, 0, 1],   # same shape as "cat"
}
X = [embed[w] for w in sentence]

# Self-attention = Q, K, V all come from the SAME sentence.
output, weights = attention(X, X, X)

print("     " + "".join(f"{t:>6}" for t in sentence))
for t, row in zip(sentence, weights):
    print(f"{t:>4} " + "".join(f"{p:6.2f}" for p in row))
```

Output:

```text
        the   cat   sat    on   the   mat
 the   0.23  0.14  0.14  0.14  0.23  0.14
 cat   0.10  0.27  0.10  0.16  0.10  0.27
 sat   0.15  0.15  0.25  0.15  0.15  0.15
  on   0.13  0.21  0.13  0.21  0.13  0.21
 the   0.23  0.14  0.14  0.14  0.23  0.14
 mat   0.10  0.27  0.10  0.16  0.10  0.27
```

Read the table, it actually *worked*:

- The **"cat"** row puts its highest weight (**0.27**) on **"mat"**, and vice versa. They share an embedding, so their queries and keys align, so they attend to each other. The machine discovered they belong together.
- Each **"the"** leans on the *other* **"the"** (**0.23**), because identical words have identical keys.
- Every row sums to 1.0: each word spent 100% of its attention, just spread by similarity.

No training, no rules, no neural network yet, only dot products and softmax. That is the beating heart of every Transformer.

---

## Action: the three-letter version the papers use

In real models, Q, K, V aren't the raw embeddings. Each word's embedding is first multiplied by three **learned** matrices `Wq, Wk, Wv` so the model can decide *what* to look for, advertise, and hand over. The mechanism above is unchanged, you just project first:

```python
# Pseudocode of what a real layer adds on top of attention():
Q = matmul(X, Wq)   # learned "what am I looking for?"
K = matmul(X, Wk)   # learned "what do I advertise?"
V = matmul(X, Wv)   # learned "what do I hand over?"
output, weights = attention(Q, K, V)
```

And **multi-head attention** is just running several of these in parallel, each with its own `Wq/Wk/Wv`, so one head can track grammar while another tracks meaning, then gluing their outputs together. Same engine, several copies.

With NumPy the whole thing collapses to a few lines, which is what `torch.nn.MultiheadAttention` does under the hood:

```python
import numpy as np
def attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V, weights
```

---

## Result

| Term | Everyday meaning | In the code |
|---|---|---|
| Query (Q) | what I'm looking for | dotted against every key |
| Key (K) | what I advertise | the thing compared against |
| Value (V) | what I hand over | the rows blended at the end |
| Q · Kᵀ | match scores, all pairs | `matmul(Q, transpose(K))` |
| / √dk | volume knob for softmax | `s / sqrt(dk)` |
| softmax | scores into percentages | `softmax_row` |
| · V | weighted blend of values | `matmul(weights, V)` |

You built the real thing. When a paper writes `softmax(QKᵀ/√dk)V`, you now see a search that blends its results, not a wall of symbols.

---

## What I learned

The formula `softmax(QKᵀ/√dk)V` looks like a barrier because it's shown *before* the story. Flip it: it's a dating app (compare wants to labels), a percentage split (softmax), and a smoothie of values, with one volume knob (√dk) so nothing screams. Build it once in plain Python, print the weights, and the symbols become labels for moves you've already watched happen.

> Out-of-the-box takeaway: attention is a search where, instead of one winner, every result gets a vote weighted by how well it matches. That's it. Everything else is matrices making it fast.

Next in the series: stacking these attention layers into a full Transformer block, with the feed-forward and residual parts that turn one attention pass into a deep model.

---

**Did Q, K, V finally click here, or do you explain it a different way? I'd love to hear your version.**

#transformers #attention #self-attention #deep-learning
