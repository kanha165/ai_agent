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
    name="ai_knowledge"
)


# ==========================================
# 4. DOCUMENTS
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
# 5. ADD DOCUMENTS
# ==========================================

if collection.count() == 0:

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.add(
        ids=[
            f"doc_{i}"
            for i in range(len(documents))
        ],
        documents=documents,
        embeddings=embeddings
    )

    print("Documents added to ChromaDB.")

else:

    print(
        f"ChromaDB already contains "
        f"{collection.count()} documents."
    )


# ==========================================
# 6. VECTOR SEARCH
# ==========================================

def search_knowledge_base(query: str) -> str:

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_documents = results["documents"][0]

    if not retrieved_documents:

        return "NO_INFORMATION_FOUND"

    output = []

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        output.append(
            f"DOCUMENT {i}:\n{document}"
        )

    return "\n\n".join(output)


# ==========================================
# 7. STRICT CONTEXT EVALUATOR
# ==========================================

def evaluate_context(
    question: str,
    context: str
) -> str:

    prompt = f"""
You are a strict retrieval evaluator.

Your job is to determine whether the
retrieved context actually contains
information that can answer the user's
question.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Rules:

1. Return ENOUGH only if the context
   directly contains relevant information
   needed to answer the question.

2. If the documents are about a different
   topic, return NOT_ENOUGH.

3. Do not use your general knowledge.

4. Do not assume missing information.

5. If the question is about blockchain and
   the documents only discuss Python,
   Machine Learning or Deep Learning,
   return NOT_ENOUGH.

6. Return ONLY one of these two values:

ENOUGH

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
        f"\nRaw Evaluator Response: "
        f"{decision}"
    )

    if decision == "ENOUGH":

        return "ENOUGH"

    return "NOT_ENOUGH"


# ==========================================
# 8. GENERATE BETTER QUERY
# ==========================================

def generate_better_query(
    question: str,
    context: str
) -> str:

    prompt = f"""
You are a search query optimizer.

The user's question is:

{question}

The previous retrieved context was:

{context}

The previous search did not provide
enough relevant information.

Create a more precise search query
that could find information specifically
related to the user's question.

Return ONLY the improved search query.
Do not explain anything.
"""

    chat = client.chats.create(
        model="gemini-3.6-flash"
    )

    response = chat.send_message(
        message=prompt
    )

    return response.text.strip()


# ==========================================
# 9. AGENTIC RETRIEVAL
# ==========================================

def agentic_retrieve(
    question: str
) -> str:

    current_query = question

    for attempt in range(3):

        print(
            f"\nRetrieval Attempt: {attempt + 1}"
        )

        print(
            f"Search Query: {current_query}"
        )

        # ----------------------------------
        # Search
        # ----------------------------------

        context = search_knowledge_base(
            current_query
        )

        print(
            "\nContext Retrieved:"
        )

        print(context)

        # ----------------------------------
        # Evaluate
        # ----------------------------------

        decision = evaluate_context(
            question,
            context
        )

        print(
            f"\nContext Evaluation: "
            f"{decision}"
        )

        # ----------------------------------
        # Context is enough
        # ----------------------------------

        if decision == "ENOUGH":

            print(
                "\nAgent decided that "
                "context is sufficient."
            )

            return context

        # ----------------------------------
        # Context is NOT enough
        # ----------------------------------

        print(
            "\nAgent decided that "
            "context is insufficient."
        )

        # ----------------------------------
        # Last attempt
        # ----------------------------------

        if attempt == 2:

            print(
                "\nMaximum retrieval attempts "
                "reached."
            )

            return context

        # ----------------------------------
        # Generate better query
        # ----------------------------------

        current_query = generate_better_query(
            question,
            context
        )

        print(
            f"\nImproved Query: "
            f"{current_query}"
        )

    return context


# ==========================================
# 10. FINAL ANSWER
# ==========================================

def generate_answer(
    question: str,
    context: str
) -> str:

    prompt = f"""
Answer the user's question using ONLY
the retrieved context.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Rules:

- Do not use outside knowledge.
- Do not invent information.
- If the context does not contain the
  answer, clearly say that the information
  is not available in the knowledge base.
"""

    chat = client.chats.create(
        model="gemini-3.6-flash"
    )

    response = chat.send_message(
        message=prompt
    )

    return response.text


# ==========================================
# 11. MAIN AGENT
# ==========================================

def run_agent(
    question: str
) -> str:

    context = agentic_retrieve(
        question
    )

    answer = generate_answer(
        question,
        context
    )

    return answer


# ==========================================
# 12. RUN
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