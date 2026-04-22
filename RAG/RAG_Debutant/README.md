# 🦜🔗 RAG Project — Version LangChain

Pipeline RAG complet en utilisant **LangChain LCEL** comme orchestrateur,
**ChromaDB** comme base vectorielle et **Ollama** comme LLM local.

---

## 🏗️ Architecture LangChain LCEL

```
LCEL = LangChain Expression Language  →  syntaxe : A | B | C

┌─────────────────────────────────────────────────────────────┐
│                    CHAIN SIMPLE                             │
│                                                             │
│   question                                                  │
│      │                                                      │
│      ├──────────────────────────────────────┐               │
│      │                                      │               │
│      ▼                                      ▼               │
│  RunnablePassthrough              Retriever (ChromaDB)      │
│                                       │                     │
│                                       ▼                     │
│                               format_docs()                 │
│                                   (context)                 │
│      │                                │                     │
│      └──────────────┬─────────────────┘                     │
│                     ▼                                       │
│             ChatPromptTemplate                              │
│          (system: RAG_SYSTEM_PROMPT)                        │
│                     │                                       │
│                     ▼                                       │
│                 Ollama LLM                                  │
│             (mistral / llama3...)                           │
│                     │                                       │
│                     ▼                                       │
│             StrOutputParser                                 │
│                     │                                       │
│                     ▼                                       │
│                  answer                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Stack technique

| Composant | LangChain Class | Backend |
|-----------|-----------------|---------|
| Embeddings | `HuggingFaceEmbeddings` | `sentence-transformers` |
| Vector Store | `Chroma` | `chromadb` |
| Retriever | `VectorStoreRetriever` | similarity / MMR |
| LLM | `Ollama` | Mistral, LLaMA, Phi... |
| Prompt | `ChatPromptTemplate` | — |
| Parser | `StrOutputParser` | — |
| Chain | `LCEL` (`|`) | — |

---

## 🗂️ Structure

```
rag-langchain/
├── main.py                      ← Point d'entrée
├── config.py                    ← Configuration centralisée
├── requirements.txt
├── data/
│   ├── raw/                     ← Documents sources
│   └── chroma_db/               ← Base vectorielle (générée)
├── src/
│   ├── embeddings.py            ← HuggingFaceEmbeddings (LangChain)
│   ├── document_loader.py       ← TextLoader, WikipediaLoader, ArxivLoader
│   ├── chunker.py               ← RecursiveCharacterTextSplitter & co.
│   ├── vector_store.py          ← Chroma + as_retriever()
│   ├── llm.py                   ← Ollama / HuggingFaceHub
│   ├── rag_pipeline.py          ← Chain LCEL complète
│   └── evaluator.py             ← Métriques RAGAS (Faithfulness, Relevancy…)
├── scripts/
│   ├── ingest.py                ← Ingestion Wikipedia, ArXiv, HF Datasets
│   └── evaluate.py              ← Évaluation du pipeline en ligne de commande
└── tests/
    ├── test_pipeline.py
    └── test_evaluator.py
```

---

## 🚀 Démarrage rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt # vaut mieux utilise le python 3.11

# 2. Installer Ollama + télécharger un modèle
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral          # ~4GB RAM  (recommandé)
# ou :
ollama pull llama3.2         # ~2GB RAM  (plus léger)
ollama pull phi3             # ~2GB RAM  (Microsoft)

# 3. Ingérer les documents open-source
python scripts/ingest.py --source wikipedia    # Articles Wikipedia NLP/AI
python scripts/ingest.py --source arxiv        # Papers ArXiv
python scripts/ingest.py --source hf           # HuggingFace Datasets (SQuAD)
# ou tout d'un coup :
python scripts/ingest.py --source all

# 4. Poser des questions !
python main.py                                 # Mode interactif
python main.py -q "Qu'est-ce que le RAG ?"    # Question directe
python main.py --stream                        # Streaming token/token

# 5. Évaluer la qualité du pipeline
pip install ragas datasets                     # Installer RAGAS
python scripts/evaluate.py                     # Évaluation complète (4 métriques)
python scripts/evaluate.py --no-reference      # Sans ground truth
python scripts/evaluate.py --simple            # Métriques légères, 100% local
python scripts/evaluate.py --output eval.json  # Sauvegarder le rapport
```

---

## 💡 Exemples de code

### Usage basique
```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()
result = rag.query("Qu'est-ce que le mécanisme d'attention ?")
print(result.answer)
print(result.sources)
```

### Réponse avec sources
```python
out = rag.query_with_sources("Différence BERT vs GPT ?")
print(out["answer"])
for doc in out["source_documents"]:
    print(doc.metadata["title"])
```

### Streaming
```python
for token in rag.stream("Explique les embeddings"):
    print(token, end="", flush=True)
```

### Chain LCEL personnalisée
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Construire votre propre chain
custom_prompt = ChatPromptTemplate.from_template("""
Réponds en français en 3 points maximum.
Contexte : {context}
Question : {question}
""")

custom_chain = (
    {"context": rag.retriever | rag._format_docs,
     "question": RunnablePassthrough()}
    | custom_prompt
    | rag.llm
    | StrOutputParser()
)

answer = custom_chain.invoke("Ma question")
```

---

## ⚙️ Configuration (.env)

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
COLLECTION_NAME=rag_langchain
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
SEARCH_TYPE=mmr                 # "similarity" ou "mmr"
LLM_BACKEND=ollama
OLLAMA_MODEL=mistral
TEMPERATURE=0.1
```

---

## 🔍 Splitters LangChain disponibles

| Splitter | Usage | Import |
|----------|-------|--------|
| `RecursiveCharacterTextSplitter` | **Recommandé** — général | `langchain.text_splitter` |
| `TokenTextSplitter` | Respecter les limites LLM | `langchain.text_splitter` |
| `MarkdownHeaderTextSplitter` | Documentation MD | `langchain.text_splitter` |
| `SemanticChunker` | Découpage par sens | `langchain_experimental` |

---

## 📊 Évaluation du pipeline (RAGAS)

RAGAS est le framework open-source standard pour mesurer la qualité d'un pipeline RAG.

### Les 4 métriques

| Métrique | Ce qu'elle mesure | Ground truth ? |
|---|---|:---:|
| **Faithfulness** | La réponse est-elle fondée sur les chunks récupérés ? (pas d'hallucination) | ✗ |
| **Answer Relevancy** | La réponse répond-elle vraiment à la question posée ? | ✗ |
| **Context Precision** | Les chunks récupérés sont-ils tous utiles (pas de bruit) ? | ✓ |
| **Context Recall** | A-t-on récupéré tous les chunks nécessaires pour répondre ? | ✓ |

Le **score RAGAS global** est la moyenne harmonique des 4 métriques.

```
≥ 0.85  ✅  Excellent  — pipeline production-ready
≥ 0.70  🟡  Bon        — quelques ajustements recommandés
≥ 0.50  🟠  Moyen      — chunking ou retrieval à revoir
< 0.50  🔴  Faible     — pipeline à refactoriser
```

### Utilisation en ligne de commande

```bash
# Évaluation complète avec les 4 métriques RAGAS
python scripts/evaluate.py

# Sans ground truth (Faithfulness + Answer Relevancy seulement)
python scripts/evaluate.py --no-reference

# Métriques simples sans RAGAS (100% local, pas d'API)
python scripts/evaluate.py --simple

# Sauvegarder le rapport JSON pour comparer les versions
python scripts/evaluate.py --output reports/eval_v1.json

# Utiliser vos propres questions (fichier JSON)
python scripts/evaluate.py --questions my_questions.json
```

Format du fichier `my_questions.json` :
```json
[
  {
    "question": "Qu'est-ce que le mécanisme d'attention ?",
    "ground_truth": "Le mécanisme d'attention calcule un score entre chaque paire de tokens..."
  },
  {
    "question": "Quelle est la différence entre BERT et GPT ?",
    "ground_truth": "BERT est bidirectionnel, GPT est auto-régressif..."
  }
]
```

### Utilisation dans le code

```python
from src.rag_pipeline import RAGPipeline
from src.evaluator import RAGEvaluator

rag = RAGPipeline()
evaluator = RAGEvaluator(rag)

questions = ["Qu'est-ce que le RAG ?", "Comment fonctionne ChromaDB ?"]
ground_truths = ["Le RAG combine retrieval et génération...", "ChromaDB stocke des vecteurs..."]

# Évaluation complète
report = evaluator.evaluate(questions, ground_truths)
report.display()
# → Score RAGAS : 0.821 ████████████████░░░░

# Accéder aux scores individuels
print(report.faithfulness)        # 0.91
print(report.answer_relevancy)    # 0.87
print(report.context_precision)   # 0.76
print(report.context_recall)      # 0.80
print(report.ragas_score)         # 0.821

# Sauvegarder en JSON
import json
json.dump(report.to_dict(), open("eval.json", "w"), indent=2)

# Évaluation simple sans RAGAS (100% local)
summary = evaluator.evaluate_simple(questions, ground_truths)
print(summary["avg_retrieval_score"])   # pertinence des chunks
print(summary["avg_latency_ms"])        # temps de réponse moyen
print(summary["avg_token_overlap"])     # chevauchement réponse/référence
```

### Stack technique

| Composant | Outil | Licence |
|---|---|---|
| Framework évaluation | `ragas` | Apache 2.0 |
| Dataset d'éval | `datasets` (HuggingFace) | Apache 2.0 |
| Métriques maison | Token F1, retrieval score | — |

Installation : `pip install ragas datasets`

---

## 📚 Ressources LangChain

- [LangChain LCEL](https://python.langchain.com/docs/expression_language/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/)
- [LangChain Ollama](https://python.langchain.com/docs/integrations/llms/ollama/)
- [RAGAS — Evaluation Framework](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
