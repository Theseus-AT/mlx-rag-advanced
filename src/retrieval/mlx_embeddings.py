import mlx.core as mx
# mlx.nn und mlx_lm imports werden hier nicht mehr direkt benötigt
# import mlx.nn as nn
# from mlx_lm.utils import load as mlx_load_model # Nicht mehr benötigt

from typing import Any, List, Optional, Iterator, Dict
import asyncio
import numpy as np # Wird oft von sentence-transformers zurückgegeben

# LangChain Core Imports
from langchain_core.embeddings import Embeddings

# NEU: Sentence Transformers Import
try:
    import sys
    sys.path.append('/path/to/your/site-packages')  # Replace with the actual path to your installed packages
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "Could not import sentence_transformers library. "
        "Please install it with `pip install sentence-transformers`"
    )


class MLXEmbeddings(Embeddings):
    """
    LangChain Embeddings class using Sentence Transformers library.
    Includes debug print and error raising during model load.
    """
    # --- Pydantic Fields ---
    model_path: str
    model: Any = None
    batch_size: int = 32
    normalize: bool = True

    # --- Angepasste __init__ ---
    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
        normalize: bool = True,
        **kwargs: Any
    ):
        """Initializes the SentenceTransformer Embeddings."""
        super().__init__(**kwargs)
        self.model_path = model_path
        self.batch_size = batch_size
        self.normalize = normalize
        self._load_sentence_transformer_model()

    # --- model_post_init bleibt gleich ---
    def model_post_init(self, __context: Any) -> None:
        """Load the model after Pydantic validation."""
        if self.model is None:
             self._load_sentence_transformer_model()

    # --- Angepasste _load Methode mit Debug und raise ---
    def _load_sentence_transformer_model(self):
        """Loads the Sentence Transformer model."""
        print(f"Loading Sentence Transformer model from: {self.model_path}")
        try:
            # Verwende SentenceTransformer zum Laden
            self.model = SentenceTransformer(self.model_path)
            print("Sentence Transformer model loaded successfully.")
            # --- DEBUG ZEILE ---
            print(f"--- DEBUG: Loaded model type: {type(self.model)} ---")
            # --- /DEBUG ZEILE ---
        except Exception as e:
            print(f"Error loading Sentence Transformer model from {self.model_path}: {e}")
            # ---->> WICHTIG: Fehler weitergeben! <<----
            raise e

    # --- _embed Methode verwendet jetzt model.encode ---
    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Helper function to compute embeddings using Sentence Transformers."""
        # Check if model was loaded correctly before attempting to use it
        if self.model is None:
             # This should ideally not be reached if _load_sentence_transformer_model raises errors
             print("ERROR: Embedding model is None, cannot perform embedding.")
             return [[] for _ in texts]
        try:
            raw_embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False
            )
            return raw_embeddings.tolist()
        except Exception as e:
            # Print the specific error during encoding
            print(f"Error during Sentence Transformer encoding step: {e}")
            return [[] for _ in texts]


    # --- Öffentliche Methoden bleiben gleich ---
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        result = self._embed([text])
        return result[0] if result and result[0] else []

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous version of embed_documents."""
        return await asyncio.get_event_loop().run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous version of embed_query."""
        return await asyncio.get_event_loop().run_in_executor(None, self.embed_query, text)