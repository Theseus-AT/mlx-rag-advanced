# retrieval/mlx_embeddings.py (Version für mlx-embeddings Bibliothek)

# !!! WICHTIG: Stelle sicher, dass du die Bibliothek installiert hast:
# pip install mlx-embeddings

import mlx.core as mx
# mlx.nn wird hier wahrscheinlich nicht direkt benötigt
import numpy as np
from typing import Any, List, Optional
import asyncio
import warnings
from itertools import islice

# --- NEUE IMPORTE ---
try:
    # Importiere load und generate aus der mlx_embeddings Bibliothek
    from mlx_embeddings import load as load_embedding_model
    from mlx_embeddings import generate as generate_embeddings
    # Optional: Falls mlx_embeddings auch einen speziellen Prozessor braucht
    # from mlx_embeddings import DefaultProcessor
except ImportError:
    raise ImportError(
        "Could not import mlx_embeddings library. "
        "Please install it with `pip install mlx-embeddings`"
    )
# --------------------

# LangChain Core Imports
from langchain_core.embeddings import Embeddings

class MLXEmbeddings(Embeddings):
    """
    LangChain Embeddings class using the mlx-embeddings library
    for MLX-native embedding models like nomic-embed-text.
    """
    model_path: str
    model: Any = None
    tokenizer: Any = None # mlx_embeddings.load gibt beides zurück
    # DefaultProcessor wird hier nicht explizit gespeichert,
    # da generate ihn intern verwenden könnte oder er stateless ist.
    # Falls stateful, müsste er hier gespeichert werden.

    # Parameter wie batch_size und normalize werden jetzt potenziell
    # von mlx_embeddings.generate gehandhabt. Wir behalten sie,
    # falls wir sie manuell anwenden müssen oder zur Info.
    batch_size: int = 32 # Überprüfen, ob generate dies unterstützt
    normalize: bool = True # generate gibt bereits normalisierte Embeddings zurück

    def __init__(
        self,
        model_path: str,
        batch_size: int = 32, # Evtl. nicht mehr direkt verwendet
        normalize: bool = True, # Evtl. nicht mehr direkt verwendet
        **kwargs: Any
    ):
        """Initializes the MLXEmbeddings using mlx-embeddings."""
        super().__init__(**kwargs)
        self.model_path = model_path
        # Speichere Parameter, auch wenn sie evtl. nicht direkt von _embed genutzt werden
        self.batch_size = batch_size
        self.normalize = normalize
        self._load_mlx_embedding_model()

    def _load_mlx_embedding_model(self):
        """Loads the MLX embedding model using mlx_embeddings.load."""
        print(f"Loading MLX Embedding model from: {self.model_path} using mlx_embeddings")
        try:
            # Verwende die load Funktion aus mlx_embeddings
            self.model, self.tokenizer = load_embedding_model(self.model_path)
            # Hinweis: Das HF-Beispiel verwendet 'processor', aber die lib lädt 'tokenizer'.
            # Wir gehen davon aus, dass 'tokenizer' korrekt ist oder intern verwendet wird.
            # Falls ein 'processor' benötigt wird, muss dieser hier auch geladen/initialisiert werden.

            # Optional: Führe eine kleine Test-Inferenz durch, um sicherzustellen, dass es funktioniert
            # try:
            #     _ = generate_embeddings(self.model, self.tokenizer, texts=["test"])
            # except Exception as test_e:
            #      print(f"WARNUNG: Test-Inferenz nach dem Laden fehlgeschlagen: {test_e}")

            print("MLX Embedding model and tokenizer loaded successfully via mlx_embeddings.")
            print(f"--- DEBUG: Loaded MLX embedding model type: {type(self.model)} ---")
            print(f"--- DEBUG: Loaded MLX embedding tokenizer type: {type(self.tokenizer)} ---")

        except Exception as e:
            print(f"Error loading MLX embedding model from {self.model_path} using mlx_embeddings: {e}")
            # Optional: Detaillierteren Traceback loggen
            import traceback
            traceback.print_exc()
            raise e # Fehler weitergeben

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings in Batches using mlx_embeddings.generate."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("MLX Embedding model or tokenizer not loaded.")

        def batched(iterable, batch_size):
            """Helper to yield items in batches."""
            it = iter(iterable)
            while batch := list(islice(it, batch_size)):
                yield batch

        all_embeddings = []
        for batch in batched(texts, self.batch_size):
            try:
                output = generate_embeddings(self.model, self.tokenizer, texts=batch)
                if not hasattr(output, 'text_embeds'):
                    raise AttributeError("Output object lacks 'text_embeds' attribute.")
                batch_embeddings = np.array(output.text_embeds).tolist()
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error embedding batch: {e}")
                import traceback
                traceback.print_exc()
                # Füge leere Embeddings für diesen Batch hinzu, um Länge zu erhalten
                all_embeddings.extend([[] for _ in batch])

        return all_embeddings


    # --- Öffentliche Methoden bleiben gleich ---
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        result = self._embed([text])
        # Stelle sicher, dass das Ergebnis nicht leer ist, bevor du darauf zugreifst
        return result[0] if result else []

    # --- Async Methoden bleiben gleich (nutzen run_in_executor) ---
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous version of embed_documents."""
        return await asyncio.get_event_loop().run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous version of embed_query."""
        return await asyncio.get_event_loop().run_in_executor(None, self.embed_query, text)