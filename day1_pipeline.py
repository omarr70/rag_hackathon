import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

data_folder = "data"
db_folder = "chroma_db"

def create_vector_db():
    if not os.path.exists(data_folder):
        print("Data folder not found")
        return None

    loader = PyPDFDirectoryLoader(data_folder)
    docs = loader.load()
    print(f"Loaded pages: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        source_path = chunk.metadata.get("source", "guideline.pdf")
        doc_name = os.path.basename(source_path)
        page_num = chunk.metadata.get("page", 0) + 1
        
        chunk.metadata["document_name"] = doc_name
        chunk.metadata["page_number"] = page_num
        chunk.metadata["chunk_id"] = f"{doc_name}_p{page_num}_c{i}"
        chunk.metadata["source_url"] = "https://www.who.int/publications"

    print(f"Total chunks created: {len(chunks)}")

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_folder
    )
    print("Vector database created successfully")
    return vector_db

def search_query(db, query):
    results = db.similarity_search(query, k=2)
    print("\n--- Search Results ---")
    for doc in results:
        doc_name = doc.metadata.get("document_name", "Unknown")
        page = doc.metadata.get("page_number", 0)
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        print(f"Doc: {doc_name} | Page: {page} | ID: {chunk_id}")
        print(f"Content: {doc.page_content}\n")

db = create_vector_db()
if db:
    user_query = "What is the recommended dose?"
    search_query(db, user_query)