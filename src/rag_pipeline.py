# src/rag_pipeline.py

import os
from dotenv import load_dotenv # Importieren

# Importiere PyPDFLoader statt TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Korrigierter Import für Chroma
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate # Importiere das normale PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import warnings # Für Deprecation Warnings etc.

# Lade Variablen aus der .env Datei im Projekt-Root
# Wichtig: load_dotenv() sollte vor dem ersten Zugriff auf os.getenv aufgerufen werden.
# Der Pfad geht davon aus, dass .env im Parent-Verzeichnis von src liegt.
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Importiere deine eigenen Module
try:
    from retrieval.mlx_embeddings import MLXEmbeddings
    from generation.mlx_chat_model import MLXChatModel
except ImportError:
    warnings.warn("Konnte Module nicht aus retrieval/generation importieren, versuche direkten Import.")
    from mlx_embeddings import MLXEmbeddings
    from mlx_chat_model import MLXChatModel


# --- Konfiguration aus Umgebungsvariablen ---
DATA_SOURCE = os.getenv("DATA_SOURCE", "./docs/default_document.pdf") # Default-Wert hinzufügen
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./default_chroma_db") # Default-Wert hinzufügen
MODEL_PATH = os.getenv("MODEL_PATH") # Kein Default, da kritisch
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") # Kein Default, da kritisch

# Überprüfen, ob kritische Pfade geladen wurden
if not MODEL_PATH:
    raise ValueError("MODEL_PATH nicht in .env Datei oder Umgebungsvariablen gefunden.")
if not EMBEDDING_MODEL:
    raise ValueError("EMBEDDING_MODEL nicht in .env Datei oder Umgebungsvariablen gefunden.")
if not os.path.exists(DATA_SOURCE):
     warnings.warn(f"DATA_SOURCE Pfad '{DATA_SOURCE}' existiert nicht. Stelle sicher, dass er korrekt in .env gesetzt ist.")


# --- Hilfsfunktionen / Logik ---
# (load_and_split_documents, create_vectorstore, load_vectorstore, format_docs, setup_rag_chain bleiben unverändert)
# ... Kopiere die Funktionsdefinitionen von oben hierher ...
def load_and_split_documents(source_path):
    """Lädt Dokumente mit PyPDFLoader und splittet sie."""
    if not os.path.exists(source_path):
        # Versuche relativen Pfad vom Skriptverzeichnis aus, falls absoluter Pfad fehlt
        script_dir = os.path.dirname(__file__)
        # Geht davon aus, dass .env und docs/ im Root liegen, src/ ist ein Unterordner
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        abs_source_path = os.path.join(project_root, source_path)

        if not os.path.exists(abs_source_path):
            raise FileNotFoundError(f"Die Datei {source_path} (oder {abs_source_path}) wurde nicht gefunden.")
        # Verwende den absoluten Pfad, um sicherzugehen
        source_path = abs_source_path

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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    )
    splits = text_splitter.split_documents(documents)
    print(f"Dokument in {len(splits)} Chunks aufgeteilt.")
    return splits

def create_vectorstore(splits, embedding_model_path, persist_path):
    """Erstellt die VectorStore mit den Dokument-Splits und speichert sie."""
    if not splits:
        print("Fehler: Keine Dokument-Chunks zum Erstellen der VectorStore vorhanden.")
        return None
    print(f"Initialisiere Embeddings mit Modell: {embedding_model_path}")
    embeddings = MLXEmbeddings(model_path=embedding_model_path)
    print("Erstelle Chroma VectorStore...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_path
    )
    print(f"VectorStore erstellt und in {persist_path} gespeichert.")
    return vectorstore

def load_vectorstore(embedding_model_path, persist_path):
    """Lädt eine bestehende VectorStore vom Speicherort."""
    if not os.path.exists(persist_path):
        print(f"VectorStore-Verzeichnis {persist_path} nicht gefunden.")
        return None
    print(f"Initialisiere Embeddings zum Laden der DB mit Modell: {embedding_model_path}")
    embeddings = MLXEmbeddings(model_path=embedding_model_path)
    print(f"Lade bestehende VectorStore aus {persist_path}...")
    try:
        vectorstore = Chroma(persist_directory=persist_path, embedding_function=embeddings)
        print("VectorStore erfolgreich geladen.")
        return vectorstore
    except Exception as e:
        print(f"Fehler beim Laden der VectorStore aus {persist_path}: {e}")
        print("Möglicherweise müssen Sie das Verzeichnis löschen und neu erstellen lassen.")
        return None

def format_docs(docs):
    """Formatiert die abgerufenen Dokumente für den Prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_chain(vectorstore, chat_model_path):
    """Baut die RAG-Chain zusammen."""
    print("Konfiguriere Retriever...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"Initialisiere Chat-Modell: {chat_model_path}")
    try:
        chat_model = MLXChatModel(model_path=chat_model_path)
    except Exception as e:
        print(f"Fehler beim Initialisieren des Chat-Modells {chat_model_path}: {e}")
        raise
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
        | chat_model
        | StrOutputParser()
    )
    print("RAG-Chain ist bereit.")
    return rag_chain


# --- Hauptlogik ---
if __name__ == "__main__":
    # 1. VectorStore laden oder erstellen
    # Verwende die Variablen, die aus .env geladen wurden
    vector_store = load_vectorstore(EMBEDDING_MODEL, VECTORSTORE_PATH)

    if vector_store is None:
        print("Keine bestehende VectorStore gefunden oder Laden fehlgeschlagen. Versuche neu zu erstellen...")
        try:
            doc_splits = load_and_split_documents(DATA_SOURCE)
            if doc_splits: # Nur erstellen, wenn Splits vorhanden sind
                vector_store = create_vectorstore(doc_splits, EMBEDDING_MODEL, VECTORSTORE_PATH)
            else:
                 print("Konnte keine Chunks aus dem Dokument extrahieren. Abbruch.")
                 vector_store = None # Sicherstellen, dass vector_store None bleibt
        except FileNotFoundError as e:
            print(e) # Fehlermeldung aus load_and_split_documents anzeigen
            vector_store = None # Sicherstellen, dass vector_store None bleibt
        except Exception as e:
            print(f"Ein unerwarteter Fehler trat beim Erstellen der VectorStore auf: {e}")
            vector_store = None


    # 2. RAG Chain aufbauen, nur wenn VectorStore vorhanden ist
    rag_chain = None
    if vector_store:
        try:
           # Verwende die Variable, die aus .env geladen wurde
           rag_chain = setup_rag_chain(vector_store, MODEL_PATH)
        except Exception as e:
            print(f"Fehler beim Aufbau der RAG-Chain: {e}")
            # Chain kann nicht verwendet werden
    else:
        print("Start fehlgeschlagen: Keine gültige VectorStore verfügbar.")

    # 3. Abfrageschleife, nur wenn Chain erfolgreich aufgebaut wurde
    if rag_chain:
        # Entscheide hier, ob du streamen oder invoken willst
        use_streaming = True # Setze auf True zum Testen von Streaming

        mode = "Streaming" if use_streaming else "Invoke"
        print(f"\nRAG Pipeline bereit ({mode}). Stelle deine Fragen (tippe 'exit' zum Beenden).")

        while True:
            try:
                query = input("> ")
                if query.lower() == 'exit':
                    break
                if query:
                    if use_streaming:
                        # --- KORREKTE STREAMING-LOGIK ---
                        print("Verarbeite Anfrage (Streaming)...")
                        print("\nAntwort:")
                        full_response = ""
                        for chunk in rag_chain.stream(query):
                            print(chunk, end="", flush=True)
                            full_response += chunk
                        print() # Füge einen Zeilenumbruch am Ende der Antwort hinzu
                        # -------- ENDE STREAMING-LOGIK --------
                    else: # use_invoke
                        print("Verarbeite Anfrage (Invoke)...")
                        response = rag_chain.invoke(query)
                        print("\nAntwort:", response)

                print("-" * 20) # Trennlinie nach jeder Abfrage/Antwort
            except KeyboardInterrupt:
                print("\nAbbruch durch Benutzer.")
                break
            except Exception as e:
                print(f"\nEin Fehler trat während der Abfrage auf: {e}")
    else:
        print("RAG-Chain konnte nicht initialisiert werden. Das Programm wird beendet.")