import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

data_folder = "data"
db_folder = "chroma_db"

# 1. Loading & Splitting
def load_and_split():
    loader = PyPDFDirectoryLoader(data_folder)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        doc_name = os.path.basename(chunk.metadata.get("source", "guideline.pdf"))
        page_num = chunk.metadata.get("page", 0) + 1
        chunk.metadata["document_name"] = doc_name
        chunk.metadata["page_number"] = page_num
        chunk.metadata["chunk_id"] = f"{doc_name}_p{page_num}_c{i}"

    return chunks

# 2. Vector DB Setup
def get_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_folder
    )
    return vector_db

# 3. Simple Similarity Search
def search(vector_db, query, k=3):
    results = vector_db.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    print("1. Loading documents...")
    chunks = load_and_split()
    print(f"Total chunks: {len(chunks)}")

    print("\n2. Creating vector store...")
    vector_db = get_vector_store(chunks)

    print("\n3. Testing query...")
    query = "What is the recommended FIB-4 score cut-off for MASLD assessment?"
    results = search(vector_db, query, k=3)

    print(f"\nQuery: {query}\n")
    for rank, doc in enumerate(results, 1):
        print(f"Result {rank}:")
        print(f"- Doc: {doc.metadata.get('document_name')}")
        print(f"- Page: {doc.metadata.get('page_number')}")
        print(f"- Chunk ID: {doc.metadata.get('chunk_id')}")
        print(f"- Content Preview: {doc.page_content[:200]}...\n")