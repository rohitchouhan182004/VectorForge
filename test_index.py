from vector_index import ExactVectorIndex


index = ExactVectorIndex(dimension=3)


index.insert(1, [1, 0, 0])
index.insert(2, [0.9, 0.1, 0])
index.insert(3, [0, 1, 0])
index.insert(4, [0, 0, 1])


query = [1, 0, 0]

results = index.search(query, k=3)


print("Search Results:")

for result in results:
    print(result)


print("\nTotal vectors:", index.count())


print("\nDeleting vector 2...")

index.delete(2)

print("Total vectors:", index.count())