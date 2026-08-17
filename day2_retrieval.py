from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

db_folder = "chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=db_folder, embedding_function=embedding_model)

test_cases = [
    {
        "query": "What are the cardiometabolic criteria to diagnose MASLD?",
        "expected": "cardiometabolic"
    },
    {
        "query": "What is the recommended FIB-4 score cut-off for screening?",
        "expected": "1.3"
    },
    {
        "query": "What is the safety and effectiveness of resmetirom?",
        "expected": "resmetirom"
    }
]

def evaluate_retrieval(db, cases, k=3):
    total_chunks = len(cases) * k
    correct_chunks = 0

    for i, item in enumerate(cases, 1):
        query = item["query"]
        expected = item["expected"].lower()
        results = db.similarity_search(query, k=k)

        print(f"Q{i}: {query}")
        for rank, doc in enumerate(results, 1):
            is_match = expected in doc.page_content.lower()
            if is_match:
                correct_chunks += 1
            
            status = "CORRECT" if is_match else "IRRELEVANT"
            doc_name = doc.metadata.get("document_name", "Doc")
            page = doc.metadata.get("page_number", "N/A")
            print(f"  [{rank}] [{status}] Source: {doc_name} (Page {page})")
        print()

    precision = (correct_chunks / total_chunks) * 100
    print(f"Precision@{k}: {precision:.2f}% ({correct_chunks}/{total_chunks})")

if __name__ == "__main__":
    evaluate_retrieval(vector_db, test_cases, k=3)