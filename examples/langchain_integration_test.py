# Standard Library Imports
import sys
import os

# Adjust path to import from src
# This assumes the script is run from the 'examples' directory
# or the project root where 'src' is visible.
# Better approach: Install your package or adjust PYTHONPATH.
# For simplicity here, we add the project root relative to this script.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Imports from your local MLX wrappers
from src.generation.mlx_chat_model import MLXChatModel
from src.retrieval.mlx_embeddings import MLXEmbeddings

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate


# --- Usage Example (Requires actual model paths) ---

if __name__ == "__main__":
    # IMPORTANT: Replace with actual paths to your MLX compatible models
    # e.g., from https://huggingface.co/mlx-community
    # Use absolute paths or paths relative to where you run the script
    MODEL_PATH = "/Users/dewitt/Desktop/Theseus AI Suit/Theseus DataCraft/models/gemma-3-4b-it-qat-4bit/" # Example - REPLACE
    EMBEDDING_MODEL_PATH = "/Users/dewitt/Desktop/Theseus AI Suit/Theseus DataCraft/models/mxbai-embed-large-v1" # Example - REPLACE

    # Check if paths exist (simple check)
    if not os.path.exists(MODEL_PATH):
         print(f"ERROR: Chat model path not found: {MODEL_PATH}")
         print("Please update MODEL_PATH in examples/langchain_integration_test.py")
         # sys.exit(1) # Optional: exit if models aren't found

    if not os.path.exists(EMBEDDING_MODEL_PATH):
         print(f"ERROR: Embedding model path not found: {EMBEDDING_MODEL_PATH}")
         print("Please update EMBEDDING_MODEL_PATH in examples/langchain_integration_test.py")
         # sys.exit(1) # Optional: exit if models aren't found


    print("\n--- Testing MLXChatModel ---")
    try:
        # Make sure the model path exists before initializing
        if os.path.exists(MODEL_PATH):
            chat_model = MLXChatModel(model_path=MODEL_PATH, max_tokens=100, temp=0.1)

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a concise AI assistant speaking like a pirate."),
                ("human", "Tell me about Apple Silicon."),
            ])
            chain = prompt | chat_model

            print("Invoking chain...")
            response = chain.invoke({})
            print("\nResponse:")
            print(response.content) # Access content directly
            print("-" * 20)

            print("Streaming response...")
            full_streamed_response = ""
            for chunk in chain.stream({}):
                print(chunk.content, end="", flush=True)
                full_streamed_response += chunk.content
            print("\n--- End of Stream ---")
            print(f"Full streamed response length: {len(full_streamed_response)}")
            print("-" * 20)
        else:
            print("Skipping MLXChatModel test due to missing model path.")

    except Exception as e:
        print(f"\nError during MLXChatModel test: {e}")
        print("Check if MODEL_PATH is correct and the model is compatible.")


    print("\n--- Testing MLXEmbeddings ---")
    try:
         # Make sure the model path exists before initializing
         if os.path.exists(EMBEDDING_MODEL_PATH):
            embeddings = MLXEmbeddings(model_path=EMBEDDING_MODEL_PATH, normalize=True)

            docs_to_embed = [
                "Ahoy, Apple Silicon be usin' Unified Memory, savvy?",
                "It lets the CPU an' GPU share the same treasure chest o' memory.",
                "Less copyin' means faster plunderin' for them big AI models, arrr!"
            ]
            doc_vectors = embeddings.embed_documents(docs_to_embed)
            print(f"Embedded {len(doc_vectors)} documents.")
            if doc_vectors and isinstance(doc_vectors[0], list) and doc_vectors[0]:
                 print(f"Dimension of first vector: {len(doc_vectors[0])}")
                 # print(f"First vector (first 5 dims): {doc_vectors[0][:5]}")
            elif doc_vectors:
                 print("Embedding result seems malformed for the first document.")


            query = "What be unified memory?"
            query_vector = embeddings.embed_query(query)
            print("Embedded query.")
            if query_vector and isinstance(query_vector, list):
                 print(f"Dimension of query vector: {len(query_vector)}")
                 # print(f"Query vector (first 5 dims): {query_vector[:5]}")
            else:
                 print("Query embedding result seems malformed.")
         else:
              print("Skipping MLXEmbeddings test due to missing model path.")

    except Exception as e:
        print(f"\nError during MLXEmbeddings test: {e}")
        print("Check if EMBEDDING_MODEL_PATH is correct and the model is compatible.")