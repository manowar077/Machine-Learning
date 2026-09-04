# LLMOps Tools & Technologies: Complete Guide with Indian Examples

## Table of Contents
1. [Large Language Models (Basics)](#llm-basics)
2. [Prompt Engineering](#prompt-engineering)
3. [Prompt Versioning & Governance](#prompt-versioning)
4. [Data, Datasets, and Curation](#data)
5. [Embeddings & Vector Databases](#embeddings-vdb)
6. [RAG (Retrieval-Augmented Generation)](#rag)
7. [Evaluation & Benchmarking](#evaluation)
8. [Guardrails, Safety, and Policy](#guardrails)
9. [Serving & Inference (vLLM, TGI, Ollama)](#serving)
10. [Scaling, Batching, and Caching](#scaling)
11. [Cost Optimization](#cost)
12. [Fine-tuning (LoRA, QLoRA, SFT)](#finetuning)
13. [Distillation & Quantization](#distill-quant)
14. [CI/CD for LLM Systems](#cicd-llm)
15. [Observability & Monitoring](#observability)
16. [Agents & Tool Use](#agents)
17. [Orchestration & Pipelines](#orchestration)
18. [Model & Prompt Registry](#registry)
19. [Deployment Options (Cloud & Self-hosted)](#deployment)
20. [A/B Testing & Online Evaluation](#ab-testing)
21. [Compliance & Governance (India/Global)](#compliance)
22. [Multilingual & Indic LLMs](#indic)
23. [Synthetic Data & Feedback Loops](#synthetic)
24. [Benchmarks & Leaderboards](#benchmarks)
25. [Edge & On-Device LLMs](#edge)
26. [Summary: Choosing the Right Stack](#summary)

---

## 1. Large Language Models (Basics) {#llm-basics}

### What are LLMs?

LLMs are models trained to predict the next token in text, enabling them to answer questions, write content, reason, and follow instructions.

**Simple Analogy**:
- **Traditional ML**: A chef who knows one dish very well.
- **LLM**: A master chef who can cook many dishes by reading recipes and adapting to your tastes.

### Why LLMs for Enterprises in India?
- **Customer Support**: IRCTC, GST, UPI disputes.
- **Document Understanding**: Aadhaar KYC, bank statements, invoices, government circulars (RBI, SEBI).
- **Code & IT Support**: ITSM ticket resolution in large banks/PSUs.
- **Multilingual Reach**: Hindi, Tamil, Telugu, Marathi, Bengali, Hinglish.

### Where LLMs Fit in the Pipeline
```
Business Need → Data → [LLM/RAG/Fine-tune] → API/Agent → Deployment → Monitoring → Feedback → Iteration
```

### Alternatives
1. **Search + Keywords**: Simple but limited understanding.
2. **Rule-based/NLU**: Works for narrow tasks.
3. **Classical ML**: For structured prediction; combine with LLMs.

---

## 2. Prompt Engineering {#prompt-engineering}

### What is Prompt Engineering?
Designing inputs so the LLM gives reliable, safe, and useful outputs.

**Analogy**: Asking a government clerk for a certificate—clear, complete forms yield faster approvals.

### Core Patterns
- **Instruction + Context + Examples** (ICE)
- **Chain-of-Thought** (ask for reasoning steps when allowed)
- **Role & Style** (tone: formal/legal/concise)
- **Output Schema** (JSON/Pydantic) for downstream automation

### Indian Example: GST Helpdesk Answering
```python
from datetime import date

SYSTEM_PROMPT = (
    "You are 'GST-Sahayak', an Indian tax assistant. "
    "Answer strictly from the provided RBI/CBIC circulars and GST Acts. "
    "If unsure, say 'Insufficient information in documents provided'. "
    "Use concise, formal tone. Include section/subsection citations."
)

USER_PROMPT_TEMPLATE = """
Question: {question}

Relevant Documents:
{chunks}

Constraints:
- Indian context only
- Cite sections (e.g., CGST Act Sec 16(2))
- Output JSON with fields: answer, citations, confidence
Date: {today}
"""

def build_prompt(question: str, chunks: list[str]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        question=question,
        chunks="\n".join(f"- {c}" for c in chunks),
        today=date.today().isoformat(),
    )
```

### Tips
- Keep prompts short; move long instructions to system messages.
- Add “Don’t answer if not in context” to reduce hallucinations.
-
Always define a strict output format (JSON) for automation.

---

## 3. Prompt Versioning & Governance {#prompt-versioning}

### Why Version Prompts?
Prompts are production artifacts. Changes affect accuracy, tone, and compliance.

### Practices
- Store prompts in Git with semantic versioning.
- Attach metadata: owner, domain, last reviewer, test coverage.
- Use feature flags to roll out new prompts to a subset of traffic.

### Example: YAML Prompt Registry
```yaml
# prompts/gst_sahayak.yml
name: gst_sahayak
version: 1.4.2
owners: ["tax-team@corp.in"]
reviewed_by: ["legal@corp.in"]
locale: ["en-IN", "hi-IN"]
guardrails: ["pii_redaction", "policy_citation_required"]
prompt:
  system: |
    You are 'GST-Sahayak'... (full system text)
  template: |
    Question: {{question}}
    Relevant Documents:\n{{context}}
    Constraints: ...
tests:
  - id: itc-eligibility-001
    question: "Is ITC allowed for motor vehicles used for transport of persons?"
    must_cite: ["CGST Sec 17(5)"]
    must_contain: ["not allowed", "exceptions"]
    locale: "en-IN"
```

### Rollout via Feature Flags
```python
def get_prompt_version(user_id: str) -> str:
    # 10% users get 1.4.2, rest remain on 1.4.1
    return "1.4.2" if hash(user_id) % 10 == 0 else "1.4.1"
```

---

## 4. Data, Datasets, and Curation {#data}

### Why Data Matters More Than Parameters
Clean, representative, multilingual Indian data drives LLM quality: laws, policies, FAQs, forms, chats.

### Sources (India)
- RBI circulars, GST Acts, Income Tax Dept notifications
- IRCTC rules, Indian Railways manuals
- State government portals, RTI responses
- Bank FAQs (SBI, HDFC, ICICI), NPCI docs (UPI)

### Versioning with DVC
```bash
dvc init
dvc add data/gst_circulars/
git add data/gst_circulars.dvc .gitignore
git commit -m "Track GST circulars v2025-10"
dvc push  # to S3/GCS/MinIO
```

### Redaction & Normalization
- Remove PII (Aadhaar, PAN) with Presidio.
- Normalize Unicode and transliteration (Hinglish ↔ Hindi).
- Deduplicate and chunk documents for RAG.

---

## 5. Embeddings & Vector Databases {#embeddings-vdb}

### What & Why
Embeddings convert text to vectors so similar meanings are close, powering RAG.

### Options
- Open-source: sentence-transformers, jina-embeddings, bge-m3
- Managed: OpenAI, Cohere, Vertex AI
- Vector DBs: FAISS (local), Chroma (local), Pinecone, Milvus, Qdrant, Weaviate

### Example: Index Indian Law Corpus (FAISS)
```python
from sentence_transformers import SentenceTransformer
import faiss, json, numpy as np

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

docs = [json.loads(l) for l in open("data/indian_law_corpus.jsonl")]
texts = [d["text"] for d in docs]
emb = model.encode(texts, normalize_embeddings=True)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(np.array(emb, dtype=np.float32))
faiss.write_index(index, "faiss/indian_law.index")
```

---

## 6. RAG (Retrieval-Augmented Generation) {#rag}

### What is RAG?
LLM answers grounded in your documents, reducing hallucinations and enabling domain accuracy.

### Architecture
```
User → Query → Embed → Retrieve Top-K → Build Prompt → LLM → Post-process → Answer
                         ↑
                     Vector DB
```

### FastAPI RAG for Bank Policy Q&A (Hindi/English)
```python
# rag_api.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
import json, os
import httpx

EMBED_MODEL = os.getenv("EMBED_MODEL", "jinaai/jina-embeddings-v2-base-hi-en")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")  # vLLM/OpenAI-compatible
INDEX_PATH = os.getenv("INDEX_PATH", "faiss/bank_policies.index")
DOCS_PATH = os.getenv("DOCS_PATH", "data/bank_policies.jsonl")

app = FastAPI(title="Bank RAG API", version="1.0")

class Query(BaseModel):
    question: str
    language: str = "en"  # "hi" or "en"
    top_k: int = 5

model = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(INDEX_PATH)
docs = [json.loads(l) for l in open(DOCS_PATH)]

def search(query: str, k: int):
    q = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, idx = index.search(q, k)
    return [(docs[i], float(scores[0][j])) for j, i in enumerate(idx[0])]

def build_messages(question: str, retrieved: list, language: str):
    context = "\n".join([f"[{i}] {d['title']}: {d['chunk']}" for i, (d, _) in enumerate(retrieved)])
    system = (
        "You are 'NagrikSahayak', a bilingual Indian banking assistant. "
        "Answer strictly from the provided context. If missing, say so. "
        "Cite [indices] for each claim."
    )
    user = f"Question ({language}): {question}\n\nContext:\n{context}\n\nFormat: JSON with fields answer, citations, confidence"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

async def llm_call(messages):
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(LLM_ENDPOINT, json={
            "model": os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
            "messages": messages,
            "temperature": 0.2,
            "stream": False
        })
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

@app.post("/rag")
async def rag(query: Query):
    try:
        retrieved = search(query.question, query.top_k)
        messages = build_messages(query.question, retrieved, query.language)
        answer = await llm_call(messages)
        return {"answer": answer, "sources": [d for d, _ in retrieved]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
```

---

## 7. Evaluation & Benchmarking {#evaluation}

### Why Evaluate?
Move from “sounds good” to measurable quality. Catch regressions before release.

### Types
- Offline: RAGAS/DeepEval style metrics
- Online: A/B tests, implicit feedback, CSAT
- Safety: Toxicity, PII leakage, policy adherence

### Example: RAGAS Evaluation
```python
# eval_rag.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
import pandas as pd

df = pd.DataFrame([
    {
        "question": "Is ITC allowed on demo cars?",
        "contexts": ["CGST Sec 17(5) ..."],
        "answer": "ITC generally not allowed with exceptions ...",
        "ground_truth": "Not allowed except when used for further supply ..."
    },
])

report = evaluate(df, metrics=[faithfulness, answer_relevancy, context_relevancy])
print(report)
```

### Golden Sets
Maintain Indian domain golden questions with references and expected rationale.

---

## 8. Guardrails, Safety, and Policy {#guardrails}

### What to Guard?
- PII leakage (Aadhaar, PAN, bank a/c)
- Harmful content, hate speech in Indian languages
- Hallucinations and missing citations
- Prompt injection against tools/databases

### PII Redaction with Presidio
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def redact(text: str) -> str:
    entities = analyzer.analyze(text=text, language='en')
    return anonymizer.anonymize(text, entities).text
```

### Output Schema Validation (Pydantic)
```python
from pydantic import BaseModel, Field, ValidationError

class AnswerSchema(BaseModel):
    answer: str
    citations: list[str] = Field(min_items=1)
    confidence: float

def validate_output(raw: str) -> AnswerSchema:
    import json
    try:
        return AnswerSchema(**json.loads(raw))
    except ValidationError as e:
        raise ValueError(f"Bad output: {e}")
```

### Policy Classifier (Content Moderation)
Use Azure/OpenAI/Vertex moderation endpoints or a local safety classifier before sending response to users.

---

## 9. Serving & Inference (vLLM, TGI, Ollama) {#serving}

### Options
- **vLLM**: Fast, OpenAI-compatible, KV cache, paged attention.
- **TGI** (HuggingFace Text Generation Inference): Production-ready server.
- **Ollama**: Simple local serving on macOS/Linux.

### Start vLLM Server
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --served-model-name bank-assistant \
  --max-num-batched-tokens 8192 \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 --port 8000
```

### Client Call (OpenAI SDK Compatible)
```python
import requests

resp = requests.post(
  "http://localhost:8000/v1/chat/completions",
  json={
    "model": "bank-assistant",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Explain UPI dispute timelines."}
    ]
  }
)
print(resp.json()["choices"][0]["message"]["content"])
```

---

## 10. Scaling, Batching, and Caching {#scaling}

### Techniques
- Request batching and KV cache reuse
- Prompt caching (e.g., Cloudflare Workers AI, Together.ai cache)
- Embedding cache in Redis

### Redis Prompt Cache
```python
import redis, hashlib, json
r = redis.Redis(host='localhost', port=6379, db=0)

def cache_key(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()

def cached_completion(call, messages):
    key = cache_key(messages)
    if (val := r.get(key)):
        return json.loads(val)
    result = call(messages)
    r.setex(key, 3600, json.dumps(result))
    return result
```

---

## 11. Cost Optimization {#cost}

### Levers
- Smaller models with RAG vs huge base models
- Quantization (4/8-bit) and long KV cache windows
- Rerankers before generation to reduce tokens
- Early exit: stop when confidence reached

### Example: Rerank Top-50 → Top-5
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates):
    pairs = [[query, c] for c in candidates]
    scores = reranker.predict(pairs)
    return [c for _, c in sorted(zip(scores, candidates), reverse=True)][:5]
```

---

## 12. Fine-tuning (LoRA, QLoRA, SFT) {#finetuning}

### When to Fine-tune?
- Style/tone control for Indian legal/financial writing
- Domain-specific tools and nomenclature
- Reduce prompt length and latency

### QLoRA with PEFT/TRL (Example)
```python
# finetune_qlora.py
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

base_model = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=True, device_map="auto")

lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","v_proj"]) 
model = get_peft_model(model, lora_cfg)

ds = load_dataset("json", data_files={"train": "data/support_transcripts_train.jsonl", "eval": "data/support_transcripts_eval.jsonl"})

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
    args=SFTConfig(output_dir="outputs/qlora-bank", per_device_train_batch_size=2, gradient_accumulation_steps=8, num_train_epochs=1),
)

trainer.train()
model.save_pretrained("outputs/qlora-bank/adapter")
```

---

## 13. Distillation & Quantization {#distill-quant}

### Why?
Serve on smaller GPUs/CPUs, reduce cost, enable edge.

### Approaches
- Distill teacher → student (e.g., 70B → 7B)
- Quantize to 4/8-bit (bitsandbytes, AutoGPTQ, AWQ)

### Quantize with AutoGPTQ
```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)
model = AutoGPTQForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", quantize_config)
model.save_quantized("mistral-7b-instruct-gptq")
```

---

## 14. CI/CD for LLM Systems {#cicd-llm}

### What to Automate?
- Prompt tests (goldens)
- RAG retrieval quality checks
- Safety gates
- Canary deploy of new prompts/models

### GitHub Actions (Minimal)
```yaml
# .github/workflows/llm_deploy.yml
on: [push]
jobs:
  test-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install deps
        run: pip install -r requirements.txt
      - name: Run golden tests
        run: python tests/test_golden.py
      - name: Safety checks
        run: python tests/test_safety.py
      - name: Deploy (staging)
        run: ./deploy.sh staging
```

---

## 15. Observability & Monitoring {#observability}

### Track
- Latency (queue + infer), tokens, cost
- Error rate, timeouts, saturation
- Hallucination reports, feedback thumbs

### Example: Langfuse Instrumentation
```python
# obs.py
from langfuse import Langfuse
lf = Langfuse(public_key="pk", secret_key="sk")

def log_trace(user_id, messages, response, meta):
    lf.trace(
        name="chat_completion",
        user_id=user_id,
        input=messages,
        output=response,
        metadata=meta,
    )
```

---

## 16. Agents & Tool Use {#agents}

### Why Agents?
Let LLM call tools: search IRCTC rules, query GST database, translate, execute code.

### Example: Simple Tool Call Contract
```python
TOOLS = {
  "get_irctc_rule": lambda topic: f"Rule for {topic}: ...",
  "sql_query": lambda q: [{"count": 42}],
}

def call_tool(name, arg):
    return TOOLS[name](arg)
```

Use frameworks like LangChain or LlamaIndex for structured tool calling and memory.

---

## 17. Orchestration & Pipelines {#orchestration}

### Common Jobs
- Nightly embeddings refresh for updated circulars
- Batch re-evaluation and leaderboard updates
- Periodic safety audits

### Prefect Flow (Example)
```python
from prefect import flow, task

@task
def embed_new_docs():
    ...

@task
def refresh_index():
    ...

@flow
def nightly_refresh():
    embed_new_docs()
    refresh_index()

if __name__ == "__main__":
    nightly_refresh()
```

---

## 18. Model & Prompt Registry {#registry}

### What to Track
- Base model, adapters (LoRA), quantization
- Prompt versions, templates, tests
- Datasets and their hashes

### MLflow (Lightweight)
```python
import mlflow, json

mlflow.start_run(run_name="bank-assistant-rag")
mlflow.log_params({"base_model": "Llama-3.1-8B", "adapter": "qlora-v2"})
mlflow.log_param("prompt_version", "1.4.2")
mlflow.log_artifact("prompts/gst_sahayak.yml")
mlflow.log_artifact("faiss/bank_policies.index")
mlflow.log_metric("rag_faithfulness", 0.82)
mlflow.end_run()
```

---

## 19. Deployment Options (Cloud & Self-hosted) {#deployment}

### Managed APIs
- **AWS Bedrock**: Access Anthropic, Meta, Cohere; private VPC; India region options evolving.
- **Azure OpenAI**: Enterprise controls, Indian data residency options.
- **Google Vertex AI**: Gemini, safety, grounding.

### Self-hosted
- Kubernetes with vLLM/TGI; GPUs on EKS/GKE/AKS
- ECS/Fargate for smaller workloads

### K8s Deployment (vLLM, simplified)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama
spec:
  replicas: 2
  selector:
    matchLabels: {app: vllm}
  template:
    metadata: {labels: {app: vllm}}
    spec:
      containers:
      - name: server
        image: vllm/vllm-openai:latest
        args: ["--model", "meta-llama/Meta-Llama-3.1-8B-Instruct", "--host", "0.0.0.0"]
        ports: [{containerPort: 8000}]
        resources:
          limits: {nvidia.com/gpu: 1, cpu: "2", memory: "8Gi"}
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-svc
spec:
  type: LoadBalancer
  selector: {app: vllm}
  ports:
  - port: 80
    targetPort: 8000
```

---

## 20. A/B Testing & Online Evaluation {#ab-testing}

### Strategy
- Randomly assign users to variant A/B
- Compare business KPIs, CSAT, fallback rates

### Simple Router
```python
def route(user_id: str):
    return "B" if hash(user_id) % 2 else "A"
```

---

## 21. Compliance & Governance (India/Global) {#compliance}

### Key Points
- DPDP Act (India): Consent, purpose limitation, data retention.
- RBI/SEBI guidelines: Sensitive financial data handling.
- Audit trails for prompts, outputs, and access.
- Mask PII in logs; store only hashes.

### Example: Log Scrubbing
```python
def scrub(record: dict) -> dict:
    record["user_input"] = redact(record.get("user_input", ""))
    return record
```

---

## 22. Multilingual & Indic LLMs {#indic}

### Challenges
- Mixed scripts (Devanagari + Latin)
- Code-switching (Hinglish)
- Tokenization inefficiencies

### Tips
- Use multilingual embeddings (hi-en).
- Normalize numerals (Devanagari ↔ Arabic).
- Evaluate per-language metrics.

### Indic Models
- Bharat-LLM, OpenHathi, Indic LLMs (check latest repos)

---

## 23. Synthetic Data & Feedback Loops {#synthetic}

### Synthetic Data
- Generate edge cases (Aadhaar name mismatch scenarios)
- Paraphrase FAQs into Indic languages

### RLAIF/RLHF (High-level)
- Use human or model feedback to prefer better responses.

```python
# pseudo: collect feedback
feedback = {"helpful": True, "reason": "Cited correct GST section"}
```

---

## 24. Benchmarks & Leaderboards {#benchmarks}

### Build Domain Benchmarks
- GST, RBI, IRCTC, Income Tax Q&A
- Track faithfulness, citation rate, answer relevancy

### Leaderboard
- Compare models: GPT-4o, Llama-3.1, Mistral, Qwen (update quarterly)

---

## 25. Edge & On-Device LLMs {#edge}

### When?
- Offline kiosks, privacy constraints, low-latency compliance desks

### How?
- llama.cpp, MLC-LLM, 4-bit quantized 7B models

```bash
ollama run mistral:instruct
```

---

## 26. Summary: Choosing the Right Stack {#summary}

### For Beginners (POC)
1. Start with OpenAI/Azure + simple RAG (FAISS/Chroma)
2. Add prompt tests + basic safety checks
3. Streamlit demo → FastAPI service

### For Small Teams
1. vLLM/TGI self-hosting + Redis cache
2. RAG with Pinecone/Milvus + RAGAS eval
3. CI/CD, basic observability (Langfuse)

### For Enterprises
1. Multiregion K8s + managed APIs (Bedrock/Azure)
2. Fine-tuning (QLoRA), registry, strong guardrails
3. A/B testing, cost optimization, full governance

Remember: Start simple, measure, then scale. 🚀


