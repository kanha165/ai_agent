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
# 4. KNOWLEDGE BASE
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
# 5. METADATA
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
# 6. INITIALIZE KNOWLEDGE BASE
# ==========================================

if collection.count() == 0:

    print(
        "\nCreating knowledge base..."
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
        "Knowledge base created."
    )

else:

    print(
        f"\nKnowledge base contains "
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


    search_results = []


    for document, metadata, distance in zip(
        retrieved_documents,
        retrieved_metadatas,
        distances
    ):

        search_results.append({

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


    return search_results


# ==========================================
# 8. FORMAT CONTEXT
# ==========================================

def format_context(
    results
):

    context = []


    for i, result in enumerate(
        results,
        start=1
    ):

        context.append(
            f"""
DOCUMENT {i}

TOPIC:
{result['topic']}

SOURCE:
{result['source']}

CONTENT:
{result['document']}
"""
        )


    return "\n".join(context)


# ==========================================
# 9. STRICT CONTEXT EVALUATION
# ==========================================

def evaluate_context(
    question: str,
    context: str
):

    prompt = f"""
You are a strict RAG retrieval evaluator.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Determine whether the retrieved context
actually contains enough relevant information
to answer the user's question.

Rules:

1. Return ENOUGH only when the context
   directly contains information needed
   to answer the question.

2. If the documents are about unrelated
   topics, return NOT_ENOUGH.

3. Do not use your own general knowledge.

4. Do not assume missing information.

5. If the user asks about blockchain and
   the context only contains Python,
   Machine Learning or Deep Learning,
   return NOT_ENOUGH.

Return ONLY:

ENOUGH

or

NOT_ENOUGH
"""

    chat = client.chats.create(
        model="gemini-3.6-flash"
    )

    response = chat.send_message(
        message=prompt
    )

    decision = response.text.strip().upper()

    print(
        f"\nEvaluator: {decision}"
    )

    if decision == "ENOUGH":

        return "ENOUGH"

    return "NOT_ENOUGH"


# ==========================================
# 10. GENERATE BETTER QUERY
# ==========================================

def generate_better_query(
    question: str,
    context: str
):

    prompt = f"""
You are a search query optimizer.

USER QUESTION:
{question}

PREVIOUS CONTEXT:
{context}

The previous search did not return
enough relevant information.

Create a more precise search query
for the knowledge base.

Focus directly on the user's question.

Return ONLY the improved search query.
"""

    chat = client.chats.create(
        model="gemini-3.6-flash"
    )

    response = chat.send_message(
        message=prompt
    )

    return response.text.strip()


# ==========================================
# 11. AGENTIC RETRIEVAL
# ==========================================

def agentic_retrieve(
    question: str
):

    current_query = question

    last_results = []


    for attempt in range(3):

        print(
            f"\n========== RETRIEVAL "
            f"ATTEMPT {attempt + 1} =========="
        )

        print(
            f"Search Query: {current_query}"
        )


        # ----------------------------------
        # SEARCH
        # ----------------------------------

        results = search_knowledge_base(
            current_query
        )

        last_results = results


        if not results:

            print(
                "\nNo documents found."
            )

            return []


        # ----------------------------------
        # SHOW RESULTS
        # ----------------------------------

        print(
            f"\nRetrieved {len(results)} "
            f"documents."
        )


        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nDocument {i}"
            )

            print(
                f"Topic: "
                f"{result['topic']}"
            )

            print(
                f"Source: "
                f"{result['source']}"
            )

            print(
                f"Distance: "
                f"{result['distance']:.4f}"
            )


        # ----------------------------------
        # FORMAT CONTEXT
        # ----------------------------------

        context = format_context(
            results
        )


        # ----------------------------------
        # EVALUATE
        # ----------------------------------

        decision = evaluate_context(
            question,
            context
        )


        # ----------------------------------
        # ENOUGH
        # ----------------------------------

        if decision == "ENOUGH":

            print(
                "\nAgent decision:"
                " Context is sufficient."
            )

            return results


        # ----------------------------------
        # NOT ENOUGH
        # ----------------------------------

        print(
            "\nAgent decision:"
            " Context is insufficient."
        )


        # ----------------------------------
        # MAX ATTEMPTS
        # ----------------------------------

        if attempt == 2:

            print(
                "\nMaximum retrieval attempts "
                "reached."
            )

            return last_results


        # ----------------------------------
        # BETTER QUERY
        # ----------------------------------

        current_query = generate_better_query(
            question,
            context
        )

        print(
            f"\nImproved Query:"
            f" {current_query}"
        )


    return last_results


# ==========================================
# 12. GENERATE FINAL ANSWER
# ==========================================

def generate_answer(
    question: str,
    results
):

    if not results:

        return (
            "Information not available "
            "in the knowledge base."
        )


    context = format_context(
        results
    )


    prompt = f"""
You are a knowledge-base RAG assistant.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Answer the question using ONLY the
retrieved context.

Rules:

1. Do not use outside knowledge.

2. Do not invent information.

3. Keep the answer clear and concise.

4. If the context does not contain
   enough information, say:

Information not available in the
knowledge base.

After the answer, provide the sources
that contain information used in
the answer.

Format:

ANSWER:
<answer>

SOURCES:
- <source>
- <source>
"""


    chat = client.chats.create(
        model="gemini-3.6-flash"
    )

    response = chat.send_message(
        message=prompt
    )

    return response.text


# ==========================================
# 13. COMPLETE AGENT
# ==========================================

def run_agent(
    question: str
):

    results = agentic_retrieve(
        question
    )


    answer = generate_answer(
        question,
        results
    )


    return answer


# ==========================================
# 14. RUN
# ==========================================

if __name__ == "__main__":

    question = input(
        "\nYou: "
    )


    answer = run_agent(
        question
    )

    print(
        "\n========== FINAL ANSWER =========="
    )

    print(answer)