# generation/mlx_chat_model.py (Vereinfacht ohne User-KV-Cache & mit verbose-Fix)

import mlx.core as mx
import mlx.nn as nn
# Stelle sicher, dass die richtigen Funktionen importiert werden
from mlx_lm.utils import load as mlx_load_model
from mlx_lm.generate import generate as mlx_generate
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
# make_prompt_cache wird hier nicht mehr benötigt, da wir das Caching nicht manuell verwalten

from typing import Any, List, Optional, Iterator, Dict, Callable, Tuple, AsyncIterator
import asyncio
import warnings

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, GenerationChunk, ChatGenerationChunk
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun

# --- Hilfsfunktion zum Konvertieren von Nachrichten (Beispiel für Gemma, ggf. anpassen) ---
def _convert_messages_to_prompt(tokenizer, messages: List[BaseMessage]) -> str:
    """Konvertiert LangChain Nachrichten in einen String-Prompt,
       idealerweise unter Nutzung des Tokenizer-Templates."""
    if not messages:
        warnings.warn("Received empty messages list.")
        return ""

    # Prüfe, ob Nachrichten alternierend sind (user/assistant/user/...)
    roles = [msg.type for msg in messages]
    expected = ['human', 'ai'] * ((len(roles) + 1) // 2)
    if roles != expected[:len(roles)]:
        raise ValueError("Conversation roles must alternate human/ai/human/ai...")

    # Versuche, das Chat-Template anzuwenden
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        try:
            # Wichtig: add_generation_prompt=True für Inferenz!
            # Tokenize=False, da mlx_generate/stream_generate das Encoding übernimmt
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            warnings.warn(f"Could not apply chat template due to message mismatch: {e}. Using fallback.")

    # Fallback: Einfache Konkatenation
    prompt_str = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            prompt_str += f"USER: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            prompt_str += f"ASSISTANT: {msg.content}\n"
        elif isinstance(msg, SystemMessage):
            prompt_str += f"SYSTEM: {msg.content}\n"
        else:
            prompt_str += f"{msg.type.upper()}: {msg.content}\n"
    prompt_str += "ASSISTANT:" # Aufforderung zur Antwort
    warnings.warn("Used simple message concatenation as fallback. Consider verifying model compatibility.")
    return prompt_str


class MLXChatModel(BaseChatModel):
    """
    LangChain Chat Model wrapping an MLX language model using mlx-lm library.
    Optionally supports KV-Cache reuse between calls for faster multi-turn chat with `enable_kv_cache=True`.
    """
    model_path: str
    model: Any = None
    tokenizer: Any = None
    adapter_path: Optional[str] = None

    # Generation parameters (Defaults)
    max_tokens: int = 512
    temp: float = 0.7
    top_p: float = 1.0
    repetition_penalty: Optional[float] = None
    repetition_context_size: int = 20

    enable_kv_cache: bool = False

    # --- __init__ erweitert um KV-Cache Store ---
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kv_cache_store = {}  # user_id -> (input_tokens, kv_cache)

    def model_post_init(self, __context: Any) -> None:
        """Lädt das Modell nach der Pydantic-Initialisierung."""
        self._load_mlx_model()

    def _load_mlx_model(self):
        """Lädt das MLX Modell und den Tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            print("MLX model and tokenizer already loaded.")
            return
        print(f"Loading MLX model from: {self.model_path} using mlx-lm")
        try:
            # Verwende die load Funktion aus mlx_lm.utils
            self.model, self.tokenizer = mlx_load_model(self.model_path, adapter_path=self.adapter_path)
            if self.model is None or self.tokenizer is None:
                 raise ValueError("mlx_load_model returned None for model or tokenizer.")
            print("MLX model and tokenizer loaded successfully via mlx-lm.")
        except Exception as e:
            print(f"Error loading MLX model from {self.model_path} using mlx-lm: {e}")
            import traceback
            traceback.print_exc()
            raise e

    # --- Cache-Management-Methoden entfernt ---

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Main synchronous generation method."""

        prompt_text = _convert_messages_to_prompt(self.tokenizer, messages)

        user_id = kwargs.get("user_id")
        use_cache = self.enable_kv_cache and user_id is not None
        cache_input = None
        kv_cache = None
        if use_cache and user_id in self._kv_cache_store:
            cache_input, kv_cache = self._kv_cache_store[user_id]

        if not prompt_text:
             return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Error: No prompt text generated from messages."))])

        print(f"Generating response for prompt (first 100 chars): {prompt_text[:100]}...")

        # Bereite Sampler und Logits Processors vor
        sampler_temp = kwargs.get('temp', self.temp)
        sampler_top_p = kwargs.get('top_p', self.top_p)
        sampler = make_sampler(temp=sampler_temp, top_p=sampler_top_p)

        penalty = kwargs.get('repetition_penalty', self.repetition_penalty)
        penalty_context = kwargs.get('repetition_context_size', self.repetition_context_size)
        logits_processors = make_logits_processors(
            logit_bias=kwargs.get('logit_bias', None), # logit_bias optional aus kwargs
            repetition_penalty=penalty,
            repetition_context_size=penalty_context
        )

        # Bereinige kwargs für mlx_generate
        gen_kwargs = {}
        allowed_mlx_params = {'max_tokens'} # Erlaubte Parameter für mlx_generate
        gen_kwargs['max_tokens'] = kwargs.get('max_tokens', self.max_tokens) # Hole max_tokens oder nutze Default
        for k, v in kwargs.items():
             if k in allowed_mlx_params and k != 'max_tokens': # Nur zusätzliche erlaubte Params
                 gen_kwargs[k] = v

        try:
            gen_inputs = dict(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt_text,
                sampler=sampler,
                logits_processors=logits_processors,
                **gen_kwargs
            )
            if kv_cache is not None:
                gen_inputs["kv_cache"] = kv_cache
            if cache_input is not None:
                gen_inputs["cache_input"] = cache_input

            response_text = mlx_generate(**gen_inputs)

            # Manuelles Stoppen
            if stop:
                 for stop_seq in stop:
                     if stop_seq in response_text:
                         response_text = response_text.split(stop_seq, 1)[0]
                         break
            # Nach Generierung: KV-Cache speichern
            if use_cache and hasattr(self.model, 'kv_cache'):
                self._kv_cache_store[user_id] = (prompt_text, self.model.kv_cache)
        except Exception as e:
            print(f"Error during MLX generation: {e}")
            import traceback
            traceback.print_exc()
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Generation Error: {e}"))])

        print(f"Generated response (first 100 chars): {response_text[:100]}...")
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Streaming generation."""

        prompt_text = _convert_messages_to_prompt(self.tokenizer, messages)

        user_id = kwargs.get("user_id")
        use_cache = self.enable_kv_cache and user_id is not None
        cache_input = None
        kv_cache = None
        if use_cache and user_id in self._kv_cache_store:
            cache_input, kv_cache = self._kv_cache_store[user_id]

        if not prompt_text:
            yield ChatGenerationChunk(message=AIMessageChunk(content="Error: No prompt text generated from messages."))
            return

        # Bereite Sampler und Logits Processors vor
        sampler_temp = kwargs.get('temp', self.temp)
        sampler_top_p = kwargs.get('top_p', self.top_p)
        sampler = make_sampler(temp=sampler_temp, top_p=sampler_top_p)

        penalty = kwargs.get('repetition_penalty', self.repetition_penalty)
        penalty_context = kwargs.get('repetition_context_size', self.repetition_context_size)
        logits_processors = make_logits_processors(
            logit_bias=kwargs.get('logit_bias', None),
            repetition_penalty=penalty,
            repetition_context_size=penalty_context
        )

        # Bereinige kwargs für stream_generate
        stream_kwargs = {}
        allowed_mlx_params = {'max_tokens'} # Erlaubte Parameter für stream_generate
        stream_kwargs['max_tokens'] = kwargs.get('max_tokens', self.max_tokens)
        for k, v in kwargs.items():
             if k in allowed_mlx_params and k != 'max_tokens':
                 stream_kwargs[k] = v

        try:
            stream_inputs = dict(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt_text,
                sampler=sampler,
                logits_processors=logits_processors,
                **stream_kwargs
            )
            if kv_cache is not None:
                stream_inputs["kv_cache"] = kv_cache
            if cache_input is not None:
                stream_inputs["cache_input"] = cache_input

            generated_text_so_far = ""
            for chunk_yielded in stream_generate(**stream_inputs):
                if hasattr(chunk_yielded, 'text') and isinstance(chunk_yielded.text, str):
                    text = chunk_yielded.text
                elif isinstance(chunk_yielded, str):
                    text = chunk_yielded
                else:
                    warnings.warn(f"Unexpected chunk type from stream_generate: {type(chunk_yielded)}. Skipping.")
                    continue

                if not text:
                    continue

                generated_text_so_far += text
                stop_triggered = False
                final_chunk_part = text

                # Manuelles Stoppen prüfen
                if stop:
                    for stop_seq in stop:
                        if generated_text_so_far.endswith(stop_seq):
                            overlap_len = 0
                            for i in range(1, min(len(text), len(stop_seq)) + 1):
                                if text.endswith(stop_seq[:i]):
                                    overlap_len = i
                                else:
                                    break
                            chars_to_keep = len(text) - overlap_len
                            final_chunk_part = text[:chars_to_keep]
                            stop_triggered = True
                            break

                    if stop_triggered:
                        if final_chunk_part:
                            chunk_obj = ChatGenerationChunk(message=AIMessageChunk(content=final_chunk_part))
                            yield chunk_obj
                            if run_manager:
                                run_manager.on_llm_new_token(final_chunk_part, chunk=chunk_obj)
                        # Nach Generierung: KV-Cache speichern
                        if use_cache and hasattr(self.model, 'kv_cache'):
                            self._kv_cache_store[user_id] = (prompt_text, self.model.kv_cache)
                        return

                chunk_obj = ChatGenerationChunk(message=AIMessageChunk(content=final_chunk_part))
                yield chunk_obj
                if run_manager:
                    run_manager.on_llm_new_token(final_chunk_part, chunk=chunk_obj)

            # Nach Generierung: KV-Cache speichern (falls nicht vorher return)
            if use_cache and hasattr(self.model, 'kv_cache'):
                self._kv_cache_store[user_id] = (prompt_text, self.model.kv_cache)

        except Exception as e:
            print(f"Error during MLX streaming generation: {e}")
            import traceback
            traceback.print_exc()
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"\nStreaming Error: {e}"))

    # --- Async Methoden ---
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous generation."""
        # Bereinige kwargs hier, bevor sie an run_in_executor übergeben werden
        kwargs.pop('verbose', None) # Entferne verbose sicherheitshalber
        allowed_kwargs = {k: v for k, v in kwargs.items() if k in ['temp', 'top_p', 'repetition_penalty', 'repetition_context_size', 'max_tokens', 'logit_bias']} # Nur bekannte weitergeben

        sync_result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._generate,
            messages,
            stop,
            run_manager.get_sync() if run_manager else None,
            **allowed_kwargs # Übergebe nur bereinigte kwargs
        )
        return sync_result

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronous streaming generation."""
        # Bereinige kwargs hier, bevor sie an _stream übergeben werden
        kwargs.pop('verbose', None) # Entferne verbose sicherheitshalber
        allowed_kwargs = {k: v for k, v in kwargs.items() if k in ['temp', 'top_p', 'repetition_penalty', 'repetition_context_size', 'max_tokens', 'logit_bias']} # Nur bekannte weitergeben

        sync_iterator = self._stream(
            messages,
            stop,
            run_manager.get_sync() if run_manager else None,
            **allowed_kwargs # Übergebe nur bereinigte kwargs
        )

        while True:
            try:
                # Holen des nächsten Chunks im Executor
                chunk = await asyncio.get_event_loop().run_in_executor(None, next, sync_iterator, None)
                if chunk is None: # Generator ist am Ende
                    break
                # Async Callback aufrufen, falls vorhanden
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.message.content, chunk=chunk)
                yield chunk
                # Kurze Pause, um der Event Loop Luft zu geben (optional, kann Latenz erhöhen)
                # await asyncio.sleep(0)
            except StopIteration:
                break # Normales Ende des Generators
            except Exception as e:
                # Fehler im Async Callback Manager melden
                if run_manager:
                    await run_manager.on_llm_error(e)
                # Fehler weiterwerfen oder einen Fehler-Chunk senden
                yield ChatGenerationChunk(message=AIMessageChunk(content=f"\nAsync Streaming Error: {e}"))
                # Optional: raise e


    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "mlx_chat_model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "max_tokens": self.max_tokens,
            "temp": self.temp,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "repetition_context_size": self.repetition_context_size,
            "enable_kv_cache": self.enable_kv_cache,
        }


# --- UserSessionManager Hilfsklasse ---
class UserSessionManager:
    """
    Einfache Verwaltung mehrerer Nutzerinstanzen mit eigenem Model-Cache, Vectorstore und Memory.
    Diese Klasse ist optional und wird von außen gesteuert.
    """

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_or_create(self, user_id: str, create_fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Gibt die Session für den Nutzer zurück oder erstellt eine neue über die gegebene Factory-Funktion.
        Beispiel für create_fn-Rückgabe:
            {
                "model": MLXChatModel(...),
                "vectorstore": Chroma(...),
                "memory": ConversationBufferMemory(...)
            }
        """
        if user_id not in self.sessions:
            self.sessions[user_id] = create_fn()
        return self.sessions[user_id]

    def has_user(self, user_id: str) -> bool:
        return user_id in self.sessions

    def clear_user(self, user_id: str) -> None:
        self.sessions.pop(user_id, None)

    def list_users(self) -> List[str]:
        return list(self.sessions.keys())

    def reset_all(self) -> None:
        self.sessions.clear()