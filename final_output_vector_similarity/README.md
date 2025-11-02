# Overview

Find the cosine similarity of the final output vectors with the model's head weights.
Question: Is the cosine similarity the driving factor for the logits, or is it the size of the output vector?

### Plan

We will first start with base model. Given a text, we look at the top 10 next token logit, the top 10 next token based on cosine similarity, and the top 10 based on the output vector size. Then we see how much they overlap. 

For large dataset, let's start with wikitext. But for now, let's test on basic functions.