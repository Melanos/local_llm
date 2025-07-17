import chromadb

client = chromadb.PersistentClient('./rag_database')
collection = client.get_collection('documents')
docs = collection.get()

print(f'Total documents: {len(docs["documents"])}')
print('\nDocument types:')
for i, metadata in enumerate(docs['metadatas']):
    filename = metadata.get('filename', 'Unknown')
    doc_type = metadata.get('type', 'unknown')
    print(f'  {i+1}. {filename} ({doc_type})')

# Check for resume specifically
print('\nLooking for resume content...')
for i, doc in enumerate(docs['documents']):
    if 'Igor' in doc or 'Matsenko' in doc:
        print(f'\nFound resume content in document {i+1}:')
        print(f'Preview: {doc[:200]}...')
        break
else:
    print('No resume content found!')
