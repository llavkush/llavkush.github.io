*Why the world dropped RNNs for Transformers, explained with a kitchen, a group chat, and short runnable Python you can paste anywhere. No PhD, no PyTorch.*

**6 minute read**

---

The first time I tried to *really* understand Transformers, every explanation either waved its hands ("attention lets the model focus!") or drowned me in matrices. Neither helped.

What finally made it click was realising the whole thing is one idea (**stop reading one word at a time; read them all at once**) and that I could prove the payoff with a few lines of plain Python.

So this note pairs each idea with code you can actually run. Theory, then the practical bit, right next to it.

---

## Situation

Before 2017, understanding text meant an **RNN** (or its smarter cousin, the **LSTM**). These read a sentence like a ticker tape: **one word at a time, left to right**, carrying a little memory forward.

It worked, until it didn't. Three problems kept stacking up:

- **It was slow.** Word 1,000 can't be touched until words 1–999 are done. No skipping ahead.
- **It forgot.** By the end of a long paragraph, the start had faded, like a game of telephone where early words got muddier each step.
- **It squished everything into one box.** Older models crammed an entire sentence into a single fixed vector before answering: short or long, same tiny box.

Here's the slow, forgetful part as code. Notice the one line that ruins everything:

```python
def rnn_style(words, decay=0.9):
    memory = 0.0
    out = []
    for w in words:                  # strictly one at a time
        memory = decay * memory + w  # <-- this needs the PREVIOUS memory
        out.append(round(memory, 2))
    return out

print(rnn_style([1, 2, 3, 4]))       # each step waited for the one before it
```

That `memory = ... memory ...` line is the villain: step *t* literally cannot start until step *t−1* finishes. On a GPU that can do millions of things at once, you're forcing them into a single-file line.

---

## Task

I wanted to be able to answer, in plain words:

- What was actually *wrong* with reading one word at a time?
- What is **attention**, and why does "every word looks at every other word" change the game?
- Why is **doing things in parallel** the real superpower, not fancy new math?
- What's the **catch**? (There's always a catch.)

---

## Action

In 2017 a paper with a cheeky title (**"Attention Is All You Need"**) threw out the ticker tape. The result, the **Transformer**, now powers GPT, Claude, Llama, image models, Whisper, AlphaFold, basically all of modern AI.

The core swap: instead of a single chef cooking steps in order, picture a **group chat** where every word is a person and **everyone reads everyone else's messages at the same time**. The word *"it"* can instantly glance back at *"dog"*: no telephone, no waiting.

> A Transformer doesn't read a sentence. It lets every word look at every other word, all at once, and decide what matters.

And in code, the contrast is stark: the attention version has **no line that waits on a previous step**:

```python
def attention_style(words):
    total = sum(words)                  # could be split across many cores
    n = len(words)
    return [round(total / n, 2) for _ in words]  # each output sees the WHOLE input

print(attention_style([1, 2, 3, 4]))    # all four seen at once, nothing waits
```

No hidden chain. You could hand each word to a different CPU core and combine the results. That "no waiting" is the entire reason Transformers train **5–10× faster** on the same hardware.

---

## How attention actually decides

"Look at every word" is nice, but *how* does it pick what matters? Each word asks a question and every word offers a label, and the model scores the match. Three names you'll keep seeing:

- **Query**: "what am I looking for?"
- **Key**: "what do I offer?"
- **Value**: "here's my actual content."

The match score between a query and a key is just a **dot product**, the "how aligned are these two?" meter from the [linear algebra note](post.html?p=linear-algebra-intuition). High match → pay more attention → pull in more of that word's value. Here's the whole mechanism in raw Python, no libraries:

```python
from math import exp

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(scores):                       # turn scores into weights that sum to 1
    m = max(scores)
    exps = [exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]

def attention(query, keys, values):
    scores = [dot(query, k) for k in keys] # how well query matches each key
    weights = softmax(scores)              # who deserves attention (sums to 1)
    # blended value = weighted sum of every value
    return [sum(w * v[i] for w, v in zip(weights, values))
            for i in range(len(values[0]))]

# "it" (the query) compared against three words (keys), each carrying a value
query  = [1, 0]
keys   = [[1, 0],  [0, 1],  [0.8, 0.2]]    # word A aligns best with the query
values = [[10, 0], [0, 10], [5, 5]]
print(attention(query, keys, values))      # output leans heavily toward word A's value
```

Run it: the output is dominated by the value of the word whose key best matched the query. That's attention: a similarity vote, then a weighted blend.

---

## Seeing it work on a real sentence

Toy vectors are fine, but let's make attention actually *do its job*: resolve a pronoun. Take *"the dog chased it"*. A human knows **"it"** refers to the **dog**. Can a few lines of self-attention figure that out?

We give each word a tiny 3-number embedding meaning `[animate, object, action]`, then run **self-attention**: every word is a query against *all* words (here keys and values are the words themselves). The output is a table: for each word, how much attention it pays to every other.

```python
from math import exp

def dot(a, b): return sum(x * y for x, y in zip(a, b))
def softmax(s):
    m = max(s); e = [exp(x - m) for x in s]; t = sum(e)
    return [x / t for x in e]

sentence = ["the", "dog", "chased", "it"]
embed = {
    "the":    [0.0, 0.0, 0.0],
    "dog":    [1.0, 0.2, 0.0],   # very animate
    "chased": [0.0, 0.0, 1.0],   # an action
    "it":     [0.9, 0.3, 0.0],   # a pronoun -> looks animate, like "dog"
}
tokens = [embed[w] for w in sentence]

def self_attention(tokens):
    rows = []
    for q in tokens:                       # each word asks: who do I relate to?
        scores  = [dot(q, k) for k in tokens]
        rows.append(softmax(scores))       # attention weights for this word
    return rows

weights = self_attention(tokens)
for w, row in zip(sentence, weights):
    print(f"{w:>7}: " + "  ".join(f"{p:.2f}" for p in row))
```

Output:

```text
    the: 0.25  0.25  0.25  0.25
    dog: 0.13  0.38  0.13  0.35
 chased: 0.17  0.17  0.48  0.17
     it: 0.14  0.37  0.14  0.35      <- "it" leans hardest on "dog"
```

Look at the **"it"** row: ignoring itself, its biggest weight (**0.37**) lands on **"dog"**, because *"it"*'s embedding looks animate, so its query aligns with *"dog"*'s key. The model just **resolved the pronoun**, with no rules and no training, using the same dot-product-and-blend you saw above. Real transformers do exactly this, only with learned embeddings and many "attention heads" running in parallel, but the machinery on screen is the whole idea.

---

## The catch: order, and the N² cost

If everyone reads at once, how does the model know *"dog bites man"* ≠ *"man bites dog"*? It doesn't, so each word gets a **seat number** stapled on, called a **positional encoding**. Order preserved, without going sequential.

The bigger catch: every word looking at every other word means **N × N** comparisons. Double the text, **quadruple** the work. You can see it grow:

```python
def attention_cost(n):
    return n * n        # every word compared with every word

for n in [10, 100, 200, 1000]:
    print(n, "words ->", attention_cost(n), "comparisons")
# 200 words costs 4x of 100 words, not 2x
```

That quadratic cost is why long documents are pricey, and why newer tricks exist (**Flash Attention**, and **state-space models** like *Mamba*) to claw back most of the speed without the N² bill.

---

## The real reason it won: depth, not cleverness

Here's the part most explanations bury. Transformers didn't win because attention is fancier math. They won because the work can be **done in parallel**. Modern chips do millions of operations at once; the slow part is the **longest chain of steps that must happen in order** (the *serial depth*).

A running total ("prefix sum") makes this concrete. Done left-to-right it's **N** steps deep (like an RNN). The **Hillis–Steele** trick gets the same answer in about **log₂(N)** rounds by doubling the stride each round:

```python
def serial_scan(xs):                 # N steps deep, each waits for the last
    out, run = [], 0
    for x in xs:
        run += x
        out.append(run)
    return out

def parallel_scan(xs):               # ~log2(N) rounds; a round updates all at once
    xs = xs[:]
    step = 1
    while step < len(xs):
        xs = [xs[i] + xs[i - step] if i >= step else xs[i]
              for i in range(len(xs))]
        step *= 2                     # 1, 2, 4, 8 ... -> log2(N) rounds
    return xs

data = [1, 2, 3, 4, 5, 6, 7, 8]
print(serial_scan(data))             # 8 steps in a row
print(parallel_scan(data))           # same answer, only ~3 rounds deep
```

Both print the same numbers, but the second one's longest dependent chain is *far* shorter. Shorter chain = faster on real hardware. That, in one demo, is why Transformers took over.

---

## The whole thing on one page

| | **RNN / LSTM** (old) | **Transformer** (new) |
|---|---|---|
| Reads | one word at a time | all words at once |
| Speed | slow (waits on itself) | fast (parallel) |
| Long-range memory | fades / forgets | every word sees every word |
| Word order | built-in (it's sequential) | added via positional encoding |
| The catch | (none) | memory grows with N² |

---

## Result

After working through it this way, the jargon stopped being scary:

- "It's a Transformer" → a group chat for words, all read at once.
- "Attention" → a similarity vote (dot product) then a weighted blend of values.
- "Why is it faster?" → no step waits on the previous one, so chips can parallelise it.
- "Positional encoding" → seat numbers, so order survives.
- "Attention is quadratic" → everyone-reads-everyone is N × N work.

Every one of those came with a few lines of Python you can paste into a REPL and watch behave, no framework, no GPU.

---

## What I learned

The intimidating part of Transformers isn't the math; it's that the math is usually shown *before* the intuition. Flip the order: get the one idea (**parallel beats sequential**), prove it with a tiny script, and the equations become labels for things you already understand.

> Out-of-the-box takeaway: RNNs whispered down a line, one person at a time. Transformers put everyone in a group chat and let them read at once. That single change, parallel instead of sequential, is why modern AI exploded.

---

**Do you have a mental model that finally made attention click for you? I'd love to hear how you explain it.**

#transformers #attention #deep-learning #intuition
