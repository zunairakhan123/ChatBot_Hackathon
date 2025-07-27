import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Initialize ChromaDB client and collection
embedding_function = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="icodeguru_files", embedding_function=embedding_function)

def retrieve_from_uploaded_files(user_query, uploaded_files):
    """
    Search for answers in uploaded text or JSON files using ChromaDB.
    """
    all_texts = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower()

        try:
            if file_extension == ".txt":
                content = uploaded_file.read().decode("utf-8")
                all_texts.append(content)
            elif file_extension == ".json":
                content = json.load(uploaded_file)
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            all_texts.append(json.dumps(item))
                        else:
                            all_texts.append(str(item))
                elif isinstance(content, dict):
                    all_texts.append(json.dumps(content))
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

    # Add to vector store
    if all_texts:
        ids = [f"doc_{i}" for i in range(len(all_texts))]
        collection.add(documents=all_texts, ids=ids)

        # Search with query
        results = collection.query(query_texts=[user_query], n_results=3)
        if results and results["documents"]:
            top_docs = results["documents"][0]
            return "\n\n".join(top_docs)

    return "No relevant content found in uploaded files."