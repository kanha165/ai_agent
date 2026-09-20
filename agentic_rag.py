import os

from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from google import genai


# ==========================================
# 1. API SETUP
# ==========================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ==========================================
# 2. EMBEDDING MODEL
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# 3. CHROMA DATABASE
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="ai_knowledge_v2"
)


# ==========================================
# 4. DOCUMENT DATA
# ==========================================

documents = [

    """
    RAG stands for Retrieval Augmented Generation.
    RAG combines information retrieval with language generation.
    A retriever searches a knowledge base for relevant information.
    The retrieved context is passed to a Large Language Model.
    Vector databases can store embeddings for semantic search.
    """,

    """
    Agentic RAG combines retrieval augmented generation
    with AI agents. The agent can decide when to retrieve
    information, evaluate retrieved context, and perform
    another search when the information is insufficient.
    """,

    """
    Python is a high-level programming language.
    Python is widely used in artificial intelligence,
    machine learning, data science and automation.
    """,

    """
    Machine learning is a subset of artificial intelligence.
    Machine learning algorithms learn patterns from data
    and use those patterns to make predictions or decisions.
    """,

    """
    Deep learning is a subset of machine learning.
    Deep learning uses neural networks with multiple layers.
    Deep learning is widely used for computer vision,
    natural language processing and speech recognition.
    """,

    """
    Vector databases store vector representations called
    embeddings. They allow semantic similarity search.
    Examples of vector databases include ChromaDB,
    FAISS, Pinecone and Weaviate.
    """
]


# ==========================================
# 5. DOCUMENT METADATA
# ==========================================

metadatas = [

    {
        "topic": "RAG",
        "source": "rag_guide"
    },

    {
        "topic": "Agentic RAG",
        "source": "agentic_rag_guide"
    },

    {
        "topic": "Python",
        "source": "python_guide"
    },

    {
        "topic": "Machine Learning",
        "source": "ml_guide"
    },

    {
        "topic": "Deep Learning",
        "source": "deep_learning_guide"
    },

    {
        "topic": "Vector Database",
        "source": "vector_database_guide"
    }
]


# ==========================================
# 6. ADD DOCUMENTS
# ==========================================

if collection.count() == 0:

    print(
        "\nCreating documents with metadata..."
    )

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.add(
        ids=[
            f"doc_{i}"
            for i in range(len(documents))
        ],

        documents=documents,

        embeddings=embeddings,

        metadatas=metadatas
    )

    print(
        "Documents added successfully."
    )

else:

    print(
        f"\nCollection already contains "
        f"{collection.count()} documents."
    )


# ==========================================
# 7. VECTOR SEARCH
# ==========================================

def search_knowledge_base(
    query: str
):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,

        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_documents = results[
        "documents"
    ][0]

    retrieved_metadatas = results[
        "metadatas"
    ][0]

    distances = results[
        "distances"
    ][0]

    if not retrieved_documents:

        return []


    results_list = []


    for document, metadata, distance in zip(
        retrieved_documents,
        retrieved_metadatas,
        distances
    ):

        results_list.append({

            "document": document,

            "topic": metadata.get(
                "topic",
                "unknown"
            ),

            "source": metadata.get(
                "source",
                "unknown"
            ),

            "distance": distance
        })


    return results_list


# ==========================================
# 8. TEST SEARCH
# ==========================================

if __name__ == "__main__":

    query = input(
        "\nEnter search query: "
    )

    results = search_knowledge_base(
        query
    )


    print(
        "\n========== SEARCH RESULTS =========="
    )


    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nDOCUMENT {i}"
        )

        print(
            f"Topic  : {result['topic']}"
        )

        print(
            f"Source : {result['source']}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print(
            f"\nContent:\n"
            f"{result['document']}"
        )