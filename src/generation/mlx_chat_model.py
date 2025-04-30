# generation/mlx_chat_model.py

import mlx.core as mx
import mlx.nn as nn
# Assuming necessary functions are imported from mlx_lm
# Check the actual imports needed based on your mlx-lm version
from mlx_lm.utils import load as mlx_load_model
from mlx_lm.generate import generate as mlx_generate
from mlx_lm.generate import stream_generate
# Import make_sampler and make_logits_processors
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from typing import Any, List, Optional, Iterator, Dict, Callable # Added Callable
import asyncio
import warnings # Import warnings module

# LangChain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, GenerationChunk, ChatGenerationChunk
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, HumanMessage
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.prompt_values import StringPromptValue


class MLXChatModel(BaseChatModel):
    """
    LangChain Chat Model wrapping an MLX language model.
    Uses make_sampler for temperature/top_p control and
    make_logits_processors for repetition_penalty.
    """
    model_path: str
    model: Any = None
    tokenizer: Any = None
    adapter_path: Optional[str] = None

    # Generation parameters (Defaults)
    max_tokens: int = 512
    temp: float = 0.7       # For sampler
    top_p: float = 1.0      # For sampler
    repetition_penalty: Optional[float] = None # For logits processor (e.g., 1.1)
    repetition_context_size: int = 20         # For logits processor
    # Add other parameters supported by make_sampler or generate_step if needed

    # Pydantic V2 specific configuration if needed
    # model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Called after standard Pydantic initialization."""
        self._load_mlx_model()

    def _load_mlx_model(self):
        """Loads the MLX model and tokenizer using mlx_lm."""
        print(f"Loading MLX model from: {self.model_path}")
        try:
            self.model, self.tokenizer = mlx_load_model(self.model_path, adapter_path=self.adapter_path)
            print("MLX model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading MLX model from {self.model_path}: {e}")
            raise e

    def _extract_prompt_from_messages(self, messages: List[BaseMessage]) -> str:
        """
        Helper function to extract the combined content string from messages.
        """
        if not messages:
            warnings.warn("Received empty messages list.")
            return ""
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
             if isinstance(messages[0].content, str):
                 return messages[0].content
             else:
                 warnings.warn(f"Unexpected content type in HumanMessage: {type(messages[0].content)}. Converting to string.")
                 return str(messages[0].content)
        warnings.warn("Received multiple messages unexpectedly. Concatenating content.")
        return "\n".join([str(m.content) for m in messages])


    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Main synchronous generation method using mlx_lm.generate."""

        prompt_text = self._extract_prompt_from_messages(messages)
        if not prompt_text:
             return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Error: No prompt text received."))])

        print(f"Generating response for prompt (first 100 chars): {prompt_text[:100]}...")

        # --- Sampler und Logits Processors erstellen ---
        sampler_temp = kwargs.pop('temp', self.temp)
        sampler_top_p = kwargs.pop('top_p', self.top_p)
        sampler = make_sampler(temp=sampler_temp, top_p=sampler_top_p)

        penalty = kwargs.pop('repetition_penalty', self.repetition_penalty)
        penalty_context = kwargs.pop('repetition_context_size', self.repetition_context_size)
        # Add logit_bias handling if needed:
        # logit_bias = kwargs.pop('logit_bias', getattr(self, 'logit_bias', None))
        logits_processors = make_logits_processors(
            logit_bias=None, # Ersetze None durch logit_bias, falls implementiert
            repetition_penalty=penalty,
            repetition_context_size=penalty_context
        )
        # --- Ende Erstellung ---

        # Bereite kwargs für mlx_generate vor
        gen_kwargs = {
            'max_tokens': self.max_tokens,
            # Füge hier andere gültige kwargs für mlx_generate/generate_step hinzu
        }
        gen_kwargs.update(kwargs) # Verbleibende Laufzeit-kwargs überschreiben Defaults
        gen_kwargs.pop('verbose', None) # 'verbose' wird von generate() anders gehandhabt

        try:
            # Übergib Sampler und Logits Processors explizit
            response_text = mlx_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt_text,
                sampler=sampler,
                logits_processors=logits_processors, # <<<< HIER ÜBERGEBEN
                **gen_kwargs
            )

            # Post-generation stop sequence handling
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
        """Streaming generation using mlx_lm.stream_generate."""
        print("--- DEBUG: Entering stream method ---")

        prompt_text = self._extract_prompt_from_messages(messages)
        if not prompt_text:
            yield ChatGenerationChunk(message=AIMessageChunk(content="Error: No prompt text received."))
            return

        # --- Sampler und Logits Processors erstellen ---
        sampler_temp = kwargs.pop('temp', self.temp)
        sampler_top_p = kwargs.pop('top_p', self.top_p)
        sampler = make_sampler(temp=sampler_temp, top_p=sampler_top_p)

        penalty = kwargs.pop('repetition_penalty', self.repetition_penalty)
        penalty_context = kwargs.pop('repetition_context_size', self.repetition_context_size)
        # logit_bias = kwargs.pop('logit_bias', getattr(self, 'logit_bias', None))
        logits_processors = make_logits_processors(
            logit_bias=None, # Ersetze None durch logit_bias, falls implementiert
            repetition_penalty=penalty,
            repetition_context_size=penalty_context
        )
        # --- Ende Erstellung ---


        # Bereite kwargs für stream_generate vor
        stream_kwargs = {
             'max_tokens': self.max_tokens,
             # Füge hier andere gültige kwargs für stream_generate/generate_step hinzu
        }
        stream_kwargs.update(kwargs) # Verbleibende Laufzeit-kwargs überschreiben Defaults
        stream_kwargs.pop('verbose', None)


        try:
            generated_text_so_far = ""
            # Übergib Sampler und Logits Processors explizit
            for chunk_yielded in stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt_text,
                sampler=sampler,
                logits_processors=logits_processors, # <<<< HIER ÜBERGEBEN
                **stream_kwargs
            ):
                # Extrahiere Text aus GenerationResponse Objekt
                if hasattr(chunk_yielded, 'text') and isinstance(chunk_yielded.text, str):
                    text = chunk_yielded.text
                elif isinstance(chunk_yielded, str): # Fallback, falls direkt String geliefert wird
                    text = chunk_yielded
                else:
                     warnings.warn(f"Unexpected chunk type from stream_generate: {type(chunk_yielded)}. Attempting str().")
                     text = str(chunk_yielded)

                if not text:
                    continue

                generated_text_so_far += text

                # Stop-Sequenz-Handling
                stop_triggered = False
                final_chunk_part = text
                if stop:
                    for stop_seq in stop:
                        if stop_seq in generated_text_so_far:
                            stop_index = generated_text_so_far.rfind(stop_seq)
                            prev_length = len(generated_text_so_far) - len(text)
                            if stop_index >= prev_length:
                                 chars_before_stop = stop_index - prev_length
                                 final_chunk_part = text[:chars_before_stop]
                                 stop_triggered = True
                                 break
                    if stop_triggered:
                         if final_chunk_part:
                             chunk = ChatGenerationChunk(message=AIMessageChunk(content=final_chunk_part))
                             yield chunk
                             if run_manager:
                                  run_manager.on_llm_new_token(final_chunk_part, chunk=chunk)
                         break # Exit the main for loop

                chunk = ChatGenerationChunk(message=AIMessageChunk(content=final_chunk_part))
                yield chunk
                if run_manager:
                     run_manager.on_llm_new_token(final_chunk_part, chunk=chunk)

        except Exception as e:
            print(f"Error during MLX streaming generation: {e}")
            raise e

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "mlx_chat_model"