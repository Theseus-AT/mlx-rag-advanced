import mlx.core as mx
import mlx.nn as nn
# Assuming necessary functions are imported from mlx_lm
# Check the actual imports needed based on your mlx-lm version
from mlx_lm.utils import load as mlx_load_model
from mlx_lm.generate import generate as mlx_generate
# Korrekter Import für Sampler:
from mlx_lm.sample_utils import make_sampler

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
    Includes sampler fix and debug print in _stream.
    """
    model_path: str
    model: Any = None
    tokenizer: Any = None
    adapter_path: Optional[str] = None

    # Generation parameters
    max_tokens: int = 512
    temp: float = 0.7
    top_p: float = 1.0
    # Add other necessary parameters

    def model_post_init(self, __context: Any) -> None:
        """Called after standard Pydantic initialization."""
        self._load_mlx_model()

    def _load_mlx_model(self):
        """Loads the MLX model and tokenizer using mlx_lm."""
        print(f"Loading MLX model from: {self.model_path}")
        try:
            self.model, self.tokenizer = mlx_load_model(self.model_path, adapter_path=self.adapter_path)
            mx.eval(self.model.parameters())
            print("MLX model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading MLX model from {self.model_path}: {e}")
            raise e # Important: re-raise error

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Main synchronous generation method using mlx_lm.generate."""
        try:
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"Warning: Tokenizer failed apply_chat_template ({e}). Using simple concatenation.")
            prompt_text = "\n".join([f"{m.type}: {m.content}" for m in messages])

        print(f"Generating response for prompt (first 100 chars): {prompt_text[:100]}...")
        gen_kwargs = {"temp": self.temp, "top_p": self.top_p, **kwargs}

        try:
            response_text = mlx_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt_text,
                max_tokens=self.max_tokens,
                verbose=False,
                **gen_kwargs
            )
            if stop:
                for stop_seq in stop:
                    if stop_seq in response_text:
                        response_text = response_text.split(stop_seq, 1)[0]
                        break
        except Exception as e:
            print(f"Error during MLX generation: {e}")
            raise e

        print(f"Generated response (first 100 chars): {response_text[:100]}...")
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
        """Asynchronous generation using run_in_executor."""
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
        """Streaming generation using mlx_lm sampler (Corrected Import/Usage + Debug)."""
        # --- DEBUG ZEILE ---
        print("--- DEBUG: Entering CORRECTED _stream method with make_sampler ---")
        # --- /DEBUG ZEILE ---
        try:
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"Warning: Tokenizer failed apply_chat_template ({e}). Using simple concatenation.")
            prompt_text = "\n".join([f"{m.type}: {m.content}" for m in messages])

        sampler_kwargs = {"temp": self.temp, "top_p": self.top_p, **kwargs}
        # Korrekte Sampler Erstellung:
        sampler = make_sampler(**sampler_kwargs)

        prompt_tokens = mx.array(self.tokenizer.encode(prompt_text))

        try:
            yielded_output = ""
            for logits_chunk, _ in zip(self.model.generate(prompt_tokens, self.max_tokens), range(self.max_tokens)):
                next_token, prob = sampler.sample(logits_chunk)
                mx.eval(next_token)
                token_id_list = [next_token.item()]
                token_text = self.tokenizer.decode(token_id_list)
                yielded_output += token_text

                stop_triggered = False
                if stop:
                    for stop_seq in stop:
                        if yielded_output.endswith(stop_seq):
                            stop_triggered = True
                            break
                if stop_triggered:
                    break
                if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                    break

                chunk = ChatGenerationChunk(message=AIMessageChunk(content=token_text))
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(token_text, chunk=chunk)
                sampler.add_token(next_token)
        except Exception as e:
            print(f"Error during MLX streaming generation: {e}")
            raise e # Important: re-raise error

    @property
    def _llm_type(self) -> str:
        return "mlx_chat_model"