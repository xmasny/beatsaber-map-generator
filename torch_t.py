from torchtext.data import get_tokenizer
import torch

# Test basic tensor operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
c = a + b

print("Tensor Operations Test:")
print(f"a: {a}")
print(f"b: {b}")
print(f"a + b: {c}")

from torchtext.datasets import IMDB
from torchtext import data

tokenizer = get_tokenizer("basic_english")
tokens = tokenizer("You can now install TorchText using pip!")

print(tokens)
