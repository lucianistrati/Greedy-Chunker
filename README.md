# Greedy Chunker

We would like to train an NLP model to recognize entities (e.g. symptoms) in a text. But, some NLP models can only process a limited number of words at a time. Therefore, we need to cut the text into appropriately sized pieces or chunks. Implement the function `chunkify` in `main.py` which accomplishes exactly that.

The output of `chunkify` must fulfill the following constraints: 

1. Chunks must not exceed a maximum number of words `max_chunk_size`
2. Chunks must not be empty (i.e. always contain at least one entity)
3. Chunks must not start/end within entities
4. Each entity must have a minimum context `min_padding` (i.e. number of words surrounding it)
5. Minimize the number of chunks of the given text

The algorithm doesn't need to be optimal with respect to the number of chunks i.e. Constraint 5 is soft. In other words, a heuristic is fine. Lastly, if a text cannot be chunked use some form of error handling.

## Examples

Here are a few examples to illustrate the more complicated constraints. We will use `[]` (square brackets) to denote entities starts/ends.

### Example 1

**Parameters:** `max_chunk_size=4`, `min_padding=0`

**Input:** "He reported [abdominal pain] and [fever]."

The following chunks violate *constraint 3* because we split in the middle of the entity `[abdominal pain]`.

```py
chunks = ["He reported abdominal", "pain and fever."]
```

Here are a few correct solutions, although the second one is preferable since it uses fewer chunks.

```py
chunks = ["He reported abdominal pain", "and fever"]
chunks = ["abdominal pain and fever."]
...
```


### Example 2

**Parameters:** `max_chunk_size=8`, `min_padding=2`

**Input:** "Yesterday she had [abdominal pain] and [fever] for an hour. [Vomiting] was not present."


The following chunks violate the following constraints

- *Constraint 4:* [abdominal pain] has no padding on the right side
- *Constraint 4:* [fever] has only padding of 1 word on its left
- *Constraint 4:* [Vomiting] has no padding on its left

```py
chunks = ["Yesterday she had abdominal pain", "and fever for an hour.", "Vomiting was not present."]
```

Here is a solution. Also, notice how this solution contains one less chunk as well than the wrong solution, therefore satisfying *Constraint 5*.

```py
chunks = ["She had abdominal pain and fever for an", "an hour. Vomiting was not present."]
...
```

# Running

pytest main.py
