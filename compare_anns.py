import numpy as np
import faiss
import datasets

dataset, dim = datasets.get_dataset("fashion-mnist-784-euclidean")
# dataset has keys: distances, neighbors, test, train

print("Dimensionality", dim)
print(type(dataset["test"]))
print(dataset["neighbors"][0])
print(dataset["distances"])
print(dataset["distances"][0])
for x in dataset:
    print(x)
    

