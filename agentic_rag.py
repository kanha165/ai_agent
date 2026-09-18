import os
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types


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
# 7. TOOL DECLARATION
# ==========================================

search_tool = types.FunctionDeclaration(
    name="search_knowledge_base",
    description=(
        "Search the internal knowledge base "
        "using semantic vector search. "
        "Use this tool when the user asks "
        "about RAG, Agentic RAG, Python, "
        "Machine Learning, Deep Learning, "
        "or Vector Databases."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "query": types.Schema(
                type="STRING",
                description=(
                    "The search query to use "
                    "for finding relevant information."
                )
            )
        },
        required=["query"]
    )
)


# ==========================================
# 8. TOOL
# ==========================================

search_tool_config = types.Tool(
    function_declarations=[
        search_tool
    ]
)


# ==========================================
# 9. AGENT
# ==========================================

def ask_agent(question: str):

    response = client.models.generate_content(
        model="gemini-3.6-flash",

        contents=question,

        config=types.GenerateContentConfig(
            tools=[
                search_tool_config
            ]
        )
    )

    return response


# ==========================================
# 10. FUNCTION CALL EXECUTION
# ==========================================

def run_agent(question: str):

    response = ask_agent(question)

    # --------------------------------------
    # Check whether Gemini requested a tool
    # --------------------------------------

    if not response.function_calls:

        return response.text

    # --------------------------------------
    # Get the first function call
    # --------------------------------------

    function_call = response.function_calls[0]

    function_name = function_call.name
    function_args = function_call.args

    print(
        f"\nTool Called: {function_name}"
    )

    print(
        f"Tool Arguments: {function_args}"
    )

    # --------------------------------------
    # Execute our Python function
    # --------------------------------------

    if function_name == "search_knowledge_base":

        tool_result = search_knowledge_base(
            function_args["query"]
        )

    else:

        tool_result = "Unknown tool."

    # --------------------------------------
    # Send tool result back to Gemini
    # --------------------------------------

    final_response = client.models.generate_content(
        model="gemini-3.6-flash",

        contents=[
            question,
            response.candidates[0].content,
            types.Part.from_function_response(
                name=function_name,
                response={
                    "result": tool_result
                }
            )
        ],

        config=types.GenerateContentConfig(
            tools=[
                search_tool_config
            ]
        )
    )

    return final_response.text


# ==========================================
# 11. RUN
# ==========================================

if __name__ == "__main__":

    question = input(
        "\nYou: "
    )

    answer = run_agent(
        question
    )

    print(
        "\nAgent:"
    )

    print(answer)