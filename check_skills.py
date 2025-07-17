import chromadb

client = chromadb.PersistentClient('./rag_database')
collection = client.get_collection('documents')

# Search specifically for skills-related content
results = collection.query(
    query_texts=["Cisco routers switches skills experience"],
    n_results=5
)

print("🔍 Skills-related search results:")
for i, doc in enumerate(results['documents'][0]):
    distance = results['distances'][0][i]
    relevance = round((1 - distance) * 100, 1)
    print(f"\n{i+1}. Relevance: {relevance}%")
    print(f"Content: {doc}")
    print("-" * 50)
