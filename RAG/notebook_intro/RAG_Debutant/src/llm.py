"""
src/llm.py — Modèles de langage via LangChain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
LangChain expose une interface commune à tous les LLMs.
On peut switcher entre Ollama, HuggingFace, etc. sans changer le pipeline.
 
Backends disponibles :
  OllamaLLM          → LLM local via Ollama (Mistral, LLaMA, Phi...)
  ChatOllama         → Version "Chat" avec messages structurés
  HuggingFaceHub     → API HuggingFace (gratuite pour petits modèles)
  HuggingFacePipeline→ Inférence locale avec transformers
"""
 
import logging
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
 
logger = logging.getLogger(__name__)
 
 
def get_llm(backend: str = config.LLM_BACKEND):
    """
    Retourne un LLM LangChain selon le backend configuré.
 
    Tous les LLMs LangChain exposent la même interface :
      llm.invoke("prompt")           → str
      llm.stream("prompt")           → Iterator[str]
      llm | StrOutputParser()        → compatible LCEL chains
 
    Args:
        backend: "ollama" | "huggingface" | "transformers"
 
    Returns:
        Un BaseLLM ou BaseChatModel LangChain
    """
    if backend == "ollama":
        return _get_ollama()
    elif backend == "huggingface":
        return _get_huggingface()
    elif backend == "transformers":
        return _get_transformers()
    else:
        raise ValueError(f"Backend inconnu : {backend}")

def _get_ollama():
    """
    ChatOllama — LLM local via Ollama.
 
    Installation :
        curl -fsSL https://ollama.ai/install.sh | sh
        ollama pull mistral        # ~4GB, recommandé
        ollama pull llama3.2       # ~2GB, plus léger
        ollama pull phi3           # ~2GB, Microsoft
        ollama pull gemma2:2b      # ~2GB, Google
    """
    from langchain_community.llms import Ollama
 
    try:
        llm = Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.TEMPERATURE,
            num_predict=config.MAX_NEW_TOKENS,
        )
        # Test rapide de connexion
        llm.invoke("Hi")
        logger.info(f"Ollama prêt : {config.OLLAMA_MODEL}")
        return llm
    except Exception as e:
        logger.error(
            f"Ollama non disponible : {e}\n"
            f"→ Installer : https://ollama.ai\n"
            f"→ Puis : ollama pull {config.OLLAMA_MODEL}"
        )
        raise
 
 
def _get_huggingface():
    """
    HuggingFaceHub — API cloud (gratuite pour les petits modèles).
    Token gratuit : https://huggingface.co/settings/tokens
    """
    from langchain_community.llms import HuggingFaceHub
 
    llm = HuggingFaceHub(
        repo_id=config.HF_MODEL_ID,
        huggingfacehub_api_token=config.HF_API_TOKEN or None,
        model_kwargs={
            "temperature": config.TEMPERATURE,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "return_full_text": False,
        },
    )
    logger.info(f"HuggingFaceHub prêt : {config.HF_MODEL_ID}")
    return llm
 
 
def _get_transformers():
    """
    HuggingFacePipeline — Inférence locale avec transformers.
    Recommandé pour : TinyLlama, Phi-2, Qwen2-0.5B (fonctionnent sur CPU).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_community.llms import HuggingFacePipeline
    import torch
 
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # ~600MB, CPU-friendly
    logger.info(f"⚙️  Chargement du modèle Transformers : {model_id}")
 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
 
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        do_sample=True,
    )
 
    llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("HuggingFacePipeline prêt")
    return llm