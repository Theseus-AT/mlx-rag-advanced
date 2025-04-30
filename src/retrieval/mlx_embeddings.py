import mlx.core as mx
import mlx.nn as nn
# Assuming necessary functions are imported from mlx_lm or elsewhere
# Check the actual imports needed based on your mlx-lm/model version
from mlx_lm.utils import load as mlx_load_model # May need adjustment for embeddings

from typing import Any, List, Optional, Iterator, Dict
import asyncio

# LangChain Core Imports
from langchain_core.embeddings import Embeddings


class MLXEmbeddings(Embeddings):
    """
    LangChain Embeddings class using an MLX model.
    (Requires user to run and debug with actual models)
    """
    model_path: str
    model: Any = None
    tokenizer: Any = None
    batch_size: int = 32
    pooling_strategy: str = "mean" # 'mean' or 'cls'
    normalize: bool = True

    # Use Pydantic v2 syntax for initialization if using newer Langchain
    def model_post_init(self, __context: Any) -> None:
        self._load_mlx_model()

    def _load_mlx_model(self):
        """Loads the MLX embedding model and tokenizer."""
        print(f"Loading MLX embedding model from: {self.model_path}")
        try:
            # --- IMPORTANT VERIFICATION POINT ---
            # Assuming mlx_load_model works for embedding models too.
            # This might need adjustment depending on how mlx-lm or the specific
            # model format handles embedding models (e.g., Sentence Transformers).
            # You might need a different loading function or manual model setup.
            self.model, self.tokenizer = mlx_load_model(self.model_path)
            mx.eval(self.model.parameters())
            print("MLX embedding model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading MLX embedding model from {self.model_path}: {e}")
            raise e

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Helper function to compute embeddings using MLX."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            try:
                # 1. Tokenize batch
                # Ensure tokenizer handles padding and truncation, and returns tensors
                # The exact output format ('np' vs 'mx') might depend on tokenizer version
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="np" # Often easier to get np then convert
                    # max_length=self.tokenizer.model_max_length # Use model's max length
                )
                input_ids_mx = mx.array(inputs['input_ids'])
                attention_mask_mx = mx.array(inputs['attention_mask'])

                # 2. Run model inference
                # --- IMPORTANT VERIFICATION POINT ---
                # The exact call depends on how the MLX embedding model is structured.
                # It might be directly callable `self.model(input_ids_mx)` or have a
                # specific method like `self.model.encode(...)`.
                # Assuming it returns a dict or object with 'last_hidden_state' or is the state directly.
                outputs = self.model(input_ids_mx) # This call needs verification!

                # Adjust based on actual model output structure:
                if isinstance(outputs, dict) and 'last_hidden_state' in outputs:
                     last_hidden_states = outputs['last_hidden_state']
                elif isinstance(outputs, mx.array):
                     last_hidden_states = outputs
                else:
                     # Try common attribute names if it's an object
                     if hasattr(outputs, 'last_hidden_state'):
                         last_hidden_states = outputs.last_hidden_state
                     else:
                         # Fallback assuming the output *is* the hidden state
                         last_hidden_states = outputs


                mx.eval(last_hidden_states) # Ensure computation

                # 3. Apply pooling strategy
                if self.pooling_strategy == "mean":
                    mask_expanded = mx.expand_dims(attention_mask_mx, axis=-1).astype(last_hidden_states.dtype)
                    sum_embeddings = mx.sum(last_hidden_states * mask_expanded, axis=1)
                    sum_mask = mx.sum(mask_expanded, axis=1)
                    sum_mask = mx.maximum(sum_mask, 1e-9) # Avoid division by zero
                    pooled_embeddings = sum_embeddings / sum_mask
                elif self.pooling_strategy == "cls":
                    # Assumes CLS token embedding is representative
                    pooled_embeddings = last_hidden_states[:, 0]
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

                # 4. Normalize embeddings (optional but common)
                if self.normalize:
                    norm = mx.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
                    pooled_embeddings = pooled_embeddings / mx.maximum(norm, 1e-9)

                # Ensure final computation before converting to list
                mx.eval(pooled_embeddings)
                batch_embeddings = pooled_embeddings.tolist()
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"Error embedding batch starting with '{batch_texts[0][:50]}...': {e}")
                # Handle error for the batch, maybe add None or raise
                # For simplicity, we'll add empty lists for failed batches
                all_embeddings.extend([[]] * len(batch_texts))


        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        result = self._embed([text])
        # Handle potential errors from _embed if it returns [[]]
        return result[0] if result and result[0] else []

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous version of embed_documents."""
        return await asyncio.get_event_loop().run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous version of embed_query."""
        return await asyncio.get_event_loop().run_in_executor(None, self.embed_query, text)