import mlx.core as mx
import mlx.nn as nn
# Assuming necessary functions are imported from mlx_lm
# Check the actual imports needed based on your mlx-lm version
from mlx_lm.utils import load as mlx_load_model
from mlx_lm.generate import generate as mlx_generate
from mlx_lm.generate import generate_step as mlx_generate_step

from typing import Any, List, Optional, Iterator, Dict
import asyncio

# LangChain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, GenerationChunk, ChatGenerationChunk
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun


class MLXChatModel(BaseChatModel):
    """
    LangChain Chat Model wrapping an MLX language model.
    (Requires user to run and debug with actual models)
    """
    model_path: str
    model: Any = None
    tokenizer: Any = None
    adapter_path: Optional[str] = None

    # Generation parameters
    max_tokens: int = 512
    temp: float = 0.7
    top_p: float = 1.0
    # Add other mlx_lm.generate parameters as needed

    # Use Pydantic v2 syntax for initialization if using newer Langchain
    # model_post_init is called after standard Pydantic initialization
    def model_post_init(self, __context: Any) -> None:
        self._load_mlx_model()

    def _load_mlx_model(self):
        """Loads the MLX model and tokenizer using mlx_lm."""
        print(f"Loading MLX model from: {self.model_path}")
        try:
            self.model, self.tokenizer = mlx_load_model(self.model_path, adapter_path=self.adapter_path)
            # Ensure model parameters are loaded (often done implicitly by load, but eval is safe)
            mx.eval(self.model.parameters())
            print("MLX model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading MLX model from {self.model_path}: {e}")
            # Consider raising a more specific error or handling appropriately
            raise e

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Main synchronous generation method using mlx_lm.generate.
        """
        # 1. Format messages into a single prompt string using the loaded tokenizer
        # This assumes the tokenizer has the apply_chat_template method
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            # Fallback or error if template application fails
            print(f"Warning: Tokenizer failed apply_chat_template ({e}). Using simple concatenation.")
            prompt = "\n".join([f"{m.type}: {m.content}" for m in messages])

        print(f"Generating response for prompt (first 100 chars): {prompt[:100]}...")

        # 2. Generate text using mlx_lm.generate
        try:
            # mlx_lm.generate typically handles tokenization if given a prompt string
            response_text = mlx_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                temp=self.temp,
                top_p=self.top_p,
                verbose=False, # Set to True for debugging prints from mlx_generate
                # TODO: Handle 'stop' sequences if mlx_generate supports them directly
                # or implement manual stopping in streaming if needed.
                # Add any other relevant kwargs passed through LangChain
                **kwargs
            )
            # mlx_generate usually handles internal mx.eval calls
        except Exception as e:
            print(f"Error during MLX generation: {e}")
            # Handle generation error appropriately
            raise e

        print(f"Generated response (first 100 chars): {response_text[:100]}...")

        # 3. Wrap the result
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous generation (basic implementation using run_in_executor)."""
        # This assumes the underlying _generate call is CPU/IO bound enough
        # or that MLX's Metal calls don't block the main asyncio loop excessively.
        # True async MLX execution might require deeper integration.
        return await asyncio.get_event_loop().run_in_executor(
            None, self._generate, messages, stop, run_manager, **kwargs
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Streaming generation using mlx_lm.generate_step."""
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"Warning: Tokenizer failed apply_chat_template ({e}). Using simple concatenation.")
            prompt = "\n".join([f"{m.type}: {m.content}" for m in messages])

        # Use the generate_step token iterator from mlx_lm
        # It likely handles tokenization internally if given prompt text
        try:
            token_iterator = mlx_generate_step(
               prompt=prompt, # Pass the string prompt
               model=self.model,
               # Note: Pass tokenizer ONLY if generate_step requires it explicitly
               # check documentation; often it's inferred from the model
               temp=self.temp,
               top_p=self.top_p,
               **kwargs
            )

            yielded_output = ""
            for token_chunk_mx, _ in zip(token_iterator, range(self.max_tokens)):
                   # Ensure computation for this step happens
                   # generate_step might yield evaluated tokens, but eval is safe
                   mx.eval(token_chunk_mx)

                   # Decode the token chunk
                   # Assuming decode works on the raw token IDs yielded by generate_step
                   token_text = self.tokenizer.decode(token_chunk_mx.tolist())

                   yielded_output += token_text

                   # TODO: Implement stop sequence handling more robustly
                   # Check if yielded_output ends with any stop sequence
                   if stop and any(yielded_output.endswith(s) for s in stop):
                       break

                   chunk = ChatGenerationChunk(message=AIMessageChunk(content=token_text))
                   yield chunk
                   if run_manager:
                       # Call callback manager with the token and the chunk
                       run_manager.on_llm_new_token(token_text, chunk=chunk)

        except Exception as e:
            print(f"Error during MLX streaming generation: {e}")
            # Handle streaming error appropriately
            raise e

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "mlx_chat_model"