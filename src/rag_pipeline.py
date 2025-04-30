from generation.mlx_chat_model import UserSessionManager
# src/rag_pipeline.py - Vereinfachte Version ohne User-KV-Cache Management

import os
import asyncio
import time
from dotenv import load_dotenv
from typing import Optional, List, Tuple, AsyncIterator # Korrigierte/Erweiterte Imports für Typing

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import warnings

# --- .env loading and module imports ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
try:
    from retrieval.mlx_embeddings import MLXEmbeddings
    from generation.mlx_chat_model import MLXChatModel
except ImportError:
    warnings.warn("Konnte Module nicht aus retrieval/generation importieren, versuche direkten Import.")
    from mlx_embeddings import MLXEmbeddings
    from mlx_chat_model import MLXChatModel


# --- Config loading ---
DATA_SOURCE = os.getenv("DATA_SOURCE", "./docs/default_document.pdf")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./default_chroma_db")
MODEL_PATH = os.getenv("MODEL_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

if not MODEL_PATH:
    raise ValueError("MODEL_PATH nicht in .env Datei oder Umgebungsvariablen gefunden.")
if not EMBEDDING_MODEL:
    raise ValueError("EMBEDDING_MODEL nicht in .env Datei oder Umgebungsvariablen gefunden.")

# Dynamische Überprüfung und Anpassung des DATA_SOURCE Pfades
if DATA_SOURCE and not os.path.exists(DATA_SOURCE):
     script_dir = os.path.dirname(__file__)
     project_root = os.path.abspath(os.path.join(script_dir, '..'))
     abs_source_path = os.path.join(project_root, DATA_SOURCE)
     if not os.path.exists(abs_source_path):
         warnings.warn(f"DATA_SOURCE Pfad '{DATA_SOURCE}' (oder '{abs_source_path}') existiert nicht. Stelle sicher, dass er korrekt in .env gesetzt ist.")
         DATA_SOURCE = None
     else:
         DATA_SOURCE = abs_source_path
elif not DATA_SOURCE:
     warnings.warn(f"Kein DATA_SOURCE in .env gefunden oder Pfad leer.")
     DATA_SOURCE = None


# --- Hilfsfunktionen (bleiben gleich) ---
def load_and_split_documents(source_path):
    """Lädt Dokumente mit PyPDFLoader und splittet sie."""
    if source_path is None:
         warnings.warn("Kein gültiger DATA_SOURCE Pfad vorhanden, Überspringe das Laden.")
         return []
    print(f"Lade Dokument aus: {source_path}")
    loader = PyPDFLoader(source_path)
    try:
        documents = loader.load()
    except ImportError:
         print("Fehler: pypdf nicht gefunden. Bitte installieren: pip install pypdf")
         raise
    except Exception as e:
        print(f"Fehler beim Laden des PDFs {source_path}: {e}")
        raise

    if not documents:
        print("Warnung: Keine Dokumente aus der PDF-Datei geladen.")
        return []

    print(f"{len(documents)} Seite(n) aus PDF geladen.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)
    print(f"Dokument in {len(splits)} Chunks aufgeteilt.")
    return splits

def create_vectorstore(splits, embedding_model_path, persist_path, collection_name="langchain"):
    """Erstellt die VectorStore für eine spezifische Collection."""
    if not splits:
        print(f"[{collection_name}] Fehler: Keine Dokument-Chunks zum Erstellen der VectorStore vorhanden.")
        return None
    print(f"[{collection_name}] Initialisiere Embeddings mit Modell: {embedding_model_path}")
    embeddings = MLXEmbeddings(model_path=embedding_model_path)
    print(f"[{collection_name}] Erstelle Chroma VectorStore (Collection: {collection_name})...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_path
    )
    print(f"[{collection_name}] VectorStore erstellt und in {persist_path} gespeichert.")
    return vectorstore

def load_vectorstore(embedding_model_path, persist_path, collection_name="langchain"):
    """Lädt eine bestehende VectorStore für eine spezifische Collection."""
    if not os.path.exists(persist_path):
        print(f"VectorStore-Basisverzeichnis {persist_path} nicht gefunden.")
        return None
    print(f"[{collection_name}] Initialisiere Embeddings zum Laden der DB mit Modell: {embedding_model_path}")
    embeddings = MLXEmbeddings(model_path=embedding_model_path)
    print(f"[{collection_name}] Versuche, bestehende VectorStore aus {persist_path} (Collection: {collection_name}) zu laden...")
    try:
        vectorstore = Chroma(
            persist_directory=persist_path,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        # Prüfen, ob die Collection tatsächlich Daten enthält
        if vectorstore._collection.count() == 0:
            print(f"[{collection_name}] Warnung: Geladene Collection ist leer.")
            # Optional: Hier None zurückgeben, um Neuerstellung zu erzwingen, wenn gewünscht.
            # return None
        print(f"[{collection_name}] VectorStore-Objekt für Collection '{collection_name}' initialisiert.")
        return vectorstore
    except Exception as e:
        print(f"[{collection_name}] Fehler/Warnung beim Initialisieren der VectorStore aus {persist_path} (Collection: {collection_name}): {e}")
        print(f"[{collection_name}] Collection existiert möglicherweise noch nicht oder ist korrupt.")
        return None

def format_docs(docs):
    """Formatiert die abgerufenen Dokumente für den Prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- Angepasste setup_rag_chain: Gibt nur die Chain zurück ---
def setup_rag_chain(vectorstore, chat_model_path) -> Optional[RunnableSequence]:
    """Baut die RAG-Chain zusammen."""
    print("Konfiguriere Retriever...")
    # Retriever arbeitet auf der spezifischen Collection des übergebenen vectorstore-Objekts
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print(f"Initialisiere Chat-Modell: {chat_model_path}")
    try:
        # Erstelle die Chat-Modell Instanz
        chat_model = MLXChatModel(model_path=chat_model_path, enable_kv_cache=True)
    except Exception as e:
        print(f"Fehler beim Initialisieren des Chat-Modells {chat_model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None # Fehlerfall

    template = """Du bist ein hilfreicher Assistent. Beantworte die folgende Frage präzise und ausschließlich basierend auf dem bereitgestellten Kontext. Formuliere eine klare Antwort und wiederhole dich nicht.

    KONTEXT:
    {context}

    FRAGE: {question}

    ANTWORT:"""
    prompt = PromptTemplate.from_template(template)

    print("Baue RAG-Chain...")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat_model # Die Instanz wird hier verwendet
        | StrOutputParser()
    )
    print("RAG-Chain ist bereit.")
    return rag_chain # Nur die Chain zurückgeben

# --- Asynchrone Hauptfunktion angepasst ---
async def amain():
    """Asynchrone Hauptlogik."""

    user_ids = ["user_1", "user_2"]
    current_user_id = user_ids[0]
    print(f"Aktueller User: {current_user_id}")

    user_vector_stores = {}
    user_rag_chains = {}
    user_sessions = UserSessionManager()
    # user_chat_models nicht mehr benötigt, da Caching intern ist (oder entfernt)
    # user_tokenizers auch nicht separat nötig für Chain-Betrieb

    async def load_resources_for_user(user_id):
        def create_user_resources():
            # Vectorstore laden oder neu erstellen
            vectorstore = load_vectorstore(EMBEDDING_MODEL, VECTORSTORE_PATH, collection_name=user_id)
            if vectorstore is None and DATA_SOURCE is not None:
                doc_splits = load_and_split_documents(DATA_SOURCE)
                if doc_splits:
                    vectorstore = create_vectorstore(doc_splits, EMBEDDING_MODEL, VECTORSTORE_PATH, collection_name=user_id)

            if not vectorstore:
                print(f"[{user_id}] Fehler: Keine Vectorstore verfügbar.")
                return {"vectorstore": None, "model": None, "chain": None}

            # Chat-Modell instanziieren
            chat_model = MLXChatModel(model_path=MODEL_PATH, enable_kv_cache=True)

            # RAG Chain bauen
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            prompt = PromptTemplate.from_template("""
Du bist ein hilfreicher Assistent. Die folgende Unterhaltung hat bereits stattgefunden:
{chat_history}

Nutze ausschließlich den bereitgestellten Kontext für deine Antwort.

KONTEXT:
{context}

FRAGE: {question}

ANTWORT:""")

            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            chain = ConversationalRetrievalChain.from_llm(
                llm=chat_model,
                retriever=retriever,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt}
            )

            return {"vectorstore": vectorstore, "model": chat_model, "chain": chain}

        user_bundle = user_sessions.get_or_create(user_id, create_user_resources)
        rag_chain = user_bundle.get("chain")
        return rag_chain

    # --- Haupt-Abfrageschleife ---
    while True:
        try:
            # --- Befehle angepasst: 'reset cache' entfernt ---
            print(f"\nAktiver User: {current_user_id}. Optionen: 'exit', 'switch user'")
            query = await asyncio.get_event_loop().run_in_executor(None, input, f"[{current_user_id}] > ")

            if query.lower() == 'exit':
                break
            elif query.lower() == 'switch user':
                 current_user_id = user_ids[1] if current_user_id == user_ids[0] else user_ids[0]
                 print(f"Wechsle zu User: {current_user_id}")
                 continue
            # --- 'reset cache' Logik entfernt ---

            # Lade Ressourcen für den aktuellen User
            rag_chain = await load_resources_for_user(current_user_id)

            # Führe Abfrage nur aus, wenn Chain bereit ist
            if query and rag_chain:
                start_time = time.time()
                total_tokens = 0 # Token Zählung vereinfacht/ungenau
                full_response = ""
                use_streaming = True

                # --- Config für Chain-Aufruf: user_id wird gesetzt ---
                run_config = {"configurable": {"user_id": current_user_id}}

                if use_streaming:
                    print(f"[{current_user_id}] Verarbeite Anfrage (Streaming)...")
                    print("\nAntwort:")
                    input_data = {"question": query}
                    async for chunk in rag_chain.astream(input_data, config=run_config):
                        print(chunk, end="", flush=True)
                        full_response += str(chunk)
                        # Vereinfachte Token-Zählung (sehr ungenau!)
                        total_tokens += len(chunk)
                    print()
                else:
                    print(f"[{current_user_id}] Verarbeite Anfrage (Invoke)...")
                    response = await rag_chain.ainvoke({"question": query}, config=run_config)
                    print("\nAntwort:", response)
                    full_response = response
                    # Vereinfachte Token-Zählung
                    total_tokens = len(full_response)


                # --- Metrik-Berechnung und Ausgabe (Token-Zählung ist jetzt ungenauer) ---
                end_time = time.time()
                duration = end_time - start_time
                tps = total_tokens / duration if duration > 0 else 0

                print(f"\n--------------------\n[{current_user_id}] Gesamtdauer: {duration:.2f}s")
                if total_tokens > 0:
                     print(f"[{current_user_id}] Generierte Zeichen (ungefähre Token): {total_tokens}")
                     print(f"[{current_user_id}] Zeichen/Sekunde (geschätzt): {tps:.2f}")
                else:
                     print(f"[{current_user_id}] Keine Zeichen zur TPS-Berechnung gezählt/generiert.")
                print("-" * 20)

            elif not query:
                 pass
            else:
                print(f"[{current_user_id}] Fehler: RAG-Chain nicht verfügbar für User {current_user_id}. Kann Anfrage nicht bearbeiten.")

        except KeyboardInterrupt:
            print("\nAbbruch durch Benutzer.")
            break
        except Exception as e:
            print(f"\nEin Fehler trat während der Abfrage auf: {e}")
            import traceback
            traceback.print_exc()

# --- Standard Python Entry Point ---
if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\nProgramm beendet.")