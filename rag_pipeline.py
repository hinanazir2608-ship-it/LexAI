# ============================================================
# STEP 2: RAG PIPELINE — The Core Engine
# ============================================================
# Flow: PDF Upload → Chunk → Embed → FAISS Store
#       → User Query → Retrieve Chunks → LLM → Answer
# Stack: LangChain + FAISS + Groq + LLaMA + Sentence Transformers
# ============================================================

import os
import json
import pickle
from pathlib import Path

# PDF Processing
import fitz  # PyMuPDF

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter   # new package in LangChain ≥0.2
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document                          # moved from langchain.schema

# Groq LLM
from groq import Groq

from system_prompt import (
    LEGAL_SYSTEM_PROMPT,
    RISK_ANALYSIS_PROMPT,
    PLAIN_ENGLISH_PROMPT,
    MISSING_CLAUSE_PROMPT,
)

# ============================================================
# CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"  # Best available on Groq
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_CHUNKS = 3

# ✅ Model ek baar load karo — app start hote hi
print("Loading embedding model... please wait")
_EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Embedding model loaded!")


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF using PyMuPDF.
    Returns clean text string.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        doc.close()

        if not full_text.strip():
            raise ValueError("No text could be extracted from this PDF. It may be scanned/image-based.")

        return full_text.strip()

    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {str(e)}")


# ============================================================
# DOCUMENT CHUNKING
# ============================================================

def chunk_document(text: str, filename: str) -> list[Document]:
    """
    Split document text into overlapping chunks for better retrieval.
    Adds metadata to each chunk for citation purposes.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_text(text)

    # Wrap as LangChain Documents with metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    return documents


# ============================================================
# FAISS VECTOR STORE — Build & Save
# ============================================================

def build_vector_store(documents: list[Document], save_path: str) -> FAISS:
    """
    Embed documents and build FAISS vector store.
    Saves to disk for reuse within the session.
    """
    embeddings = _EMBEDDING_MODEL  # already loaded — no download needed

    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(save_path)

    return vector_store


def load_vector_store(save_path: str) -> FAISS:
    """
    Load an existing FAISS vector store from disk.
    """
    embeddings = _EMBEDDING_MODEL  # already loaded — no download needed

    vector_store = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store


# ============================================================
# RETRIEVAL — Find Relevant Chunks
# ============================================================

def retrieve_relevant_chunks(query: str, vector_store: FAISS, k: int = TOP_K_CHUNKS) -> str:
    """
    Retrieve top-k most relevant chunks for a given query.
    Returns combined context string with source citations.
    """
    results = vector_store.similarity_search(query, k=k)

    context_parts = []
    for i, doc in enumerate(results):
        context_parts.append(
            f"[Excerpt {i+1} from {doc.metadata.get('source', 'document')}]:\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# GROQ LLM CALL — The Brain
# ============================================================

def call_groq_llm(system_prompt: str, user_message: str, temperature: float = 0.1) -> str:
    """
    Call Groq's LLaMA model with a system prompt and user message.
    Low temperature = more factual, less hallucination.
    """
    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,  # Keep low to reduce hallucination
        max_tokens=2048,
    )

    return response.choices[0].message.content


# ============================================================
# MAIN RAG FUNCTIONS — One per Feature
# ============================================================

def rag_legal_qa(query: str, vector_store: FAISS) -> str:
    """
    Legal Q&A: Answer a user question grounded in the document.
    """
    context = retrieve_relevant_chunks(query, vector_store)

    user_message = f"""
The attorney has uploaded a legal document. Based ONLY on the following
excerpts from that document, answer their question.

DOCUMENT EXCERPTS:
{context}

ATTORNEY'S QUESTION:
{query}

Remember: Only use information from the document excerpts above.
If the answer isn't in the excerpts, say so clearly.
"""

    return call_groq_llm(LEGAL_SYSTEM_PROMPT, user_message, temperature=0.1)


def rag_risk_analysis(vector_store: FAISS, full_text: str) -> dict:
    """
    Risk Analysis: Scan the full document for risky clauses.
    Returns parsed JSON with risks, missing clauses, score.
    """
    # For risk analysis, use full text (chunked if too long)
    # Take first 6000 chars to stay within context limits
    text_for_analysis = full_text[:6000]

    user_message = f"""
Analyze this legal contract for risks and missing clauses:

CONTRACT TEXT:
{text_for_analysis}

Return ONLY valid JSON as specified. No extra text.
"""

    raw_response = call_groq_llm(RISK_ANALYSIS_PROMPT, user_message, temperature=0.0)

    # Parse JSON safely
    try:
        # Strip any accidental markdown fences
        clean = raw_response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        # Fallback if model doesn't return clean JSON
        return {
            "risks": [],
            "missing_clauses": [],
            "overall_risk_score": "UNKNOWN",
            "summary": raw_response,
            "parse_error": True,
        }


def rag_plain_english_summary(full_text: str) -> str:
    """
    Plain English Summary: Simplify the whole contract.
    """
    text_for_summary = full_text[:5000]

    user_message = f"""
Translate this legal contract into plain English:

CONTRACT TEXT:
{text_for_summary}
"""

    return call_groq_llm(PLAIN_ENGLISH_PROMPT, user_message, temperature=0.2)


def rag_missing_clauses(full_text: str) -> dict:
    """
    Missing Clause Detection: Check what's absent from the contract.
    """
    text_for_check = full_text[:6000]

    user_message = f"""
Review this contract for missing or inadequate clauses:

CONTRACT TEXT:
{text_for_check}

Return ONLY valid JSON as specified.
"""

    raw_response = call_groq_llm(MISSING_CLAUSE_PROMPT, user_message, temperature=0.0)

    try:
        clean = raw_response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "present_clauses": [],
            "missing_clauses": [],
            "inadequate_clauses": [],
            "parse_error": True,
            "raw": raw_response,
        }


# ============================================================
# FULL PIPELINE — Called when user uploads a PDF
# ============================================================

def process_uploaded_pdf(pdf_path: str, user_id: str, upload_folder: str) -> dict:
    """
    Full pipeline: PDF → Text → Chunks → Vector Store
    Returns metadata about the processed document.
    """
    filename = Path(pdf_path).name
    vector_store_path = os.path.join(upload_folder, f"vs_{user_id}")

    # Step 1: Extract text
    full_text = extract_text_from_pdf(pdf_path)

    # Step 2: Chunk
    documents = chunk_document(full_text, filename)

    # Step 3: Build & save FAISS store
    build_vector_store(documents, vector_store_path)

    # Step 4: Save raw text for features that need it
    text_path = os.path.join(upload_folder, f"text_{user_id}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return {
        "success": True,
        "filename": filename,
        "total_chunks": len(documents),
        "vector_store_path": vector_store_path,
        "text_path": text_path,
        "char_count": len(full_text),
    }


def rag_risk_score(full_text: str) -> dict:
    """
    Generate a numerical risk score 0-100 for the contract.
    100 = extremely risky, 0 = very safe
    """
    text_for_analysis = full_text[:6000]

    prompt = """
You are a legal risk scoring expert. Analyze this contract and return ONLY valid JSON:
{
  "overall_score": 72,
  "verdict": "HIGH RISK",
  "categories": {
    "liability": {"score": 80, "label": "Unlimited Liability Clauses"},
    "termination": {"score": 60, "label": "Unfair Termination Terms"},
    "payment": {"score": 40, "label": "Payment Terms"},
    "ip_ownership": {"score": 90, "label": "IP Ownership Issues"},
    "confidentiality": {"score": 30, "label": "Confidentiality Terms"},
    "dispute_resolution": {"score": 70, "label": "Dispute Resolution"}
  },
  "top_risks": [
    "One-sided termination clause with no notice period",
    "Unlimited liability exposure for service provider",
    "IP ownership transfers entirely to client"
  ],
  "safe_points": [
    "Clear payment schedule defined",
    "Confidentiality terms are reasonable"
  ]
}

Score meaning:
0-30 = LOW RISK (green)
31-60 = MEDIUM RISK (yellow)  
61-80 = HIGH RISK (orange)
81-100 = CRITICAL RISK (red)

Return ONLY valid JSON. No extra text.
"""

    user_message = f"Score this contract for risk:\n\n{text_for_analysis}"
    raw = call_groq_llm(prompt, user_message, temperature=0.0)

    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "overall_score": 50,
            "verdict": "UNKNOWN",
            "categories": {},
            "top_risks": [],
            "safe_points": [],
            "parse_error": True
        }


# ============================================================
# CONTRACT COMPARISON
# ============================================================

COMPARISON_PROMPT = """
You are a legal contract comparison expert. You are given TWO contracts.
Compare them thoroughly and return ONLY valid JSON:

{
  "summary": "2-3 sentence overview of key differences",
  "verdict": "Document 1 is more favorable" or "Document 2 is more favorable" or "Both are balanced",
  "verdict_reason": "Why one is better for the attorney/client",
  
  "added_clauses": [
    {
      "clause": "clause name",
      "in_document": "Document 2",
      "content": "brief description of the clause",
      "impact": "HIGH or MEDIUM or LOW"
    }
  ],
  
  "removed_clauses": [
    {
      "clause": "clause name",
      "was_in_document": "Document 1",
      "content": "brief description",
      "impact": "HIGH or MEDIUM or LOW"
    }
  ],
  
  "modified_clauses": [
    {
      "clause": "clause name",
      "doc1_version": "how it reads in document 1",
      "doc2_version": "how it reads in document 2",
      "change_type": "More Restrictive or More Lenient or Neutral",
      "impact": "HIGH or MEDIUM or LOW"
    }
  ],
  
  "risk_comparison": {
    "doc1_risk": "HIGH or MEDIUM or LOW",
    "doc2_risk": "HIGH or MEDIUM or LOW",
    "doc1_score": 65,
    "doc2_score": 40
  },
  
  "recommendations": [
    "specific recommendation for the attorney"
  ]
}

Return ONLY valid JSON. No extra text.
"""

def rag_compare_contracts(text1: str, text2: str) -> dict:
    """
    Compare two contract texts and return structured differences.
    """
    # Limit text to avoid context overflow
    t1 = text1[:4000]
    t2 = text2[:4000]

    user_message = f"""
Compare these two legal contracts:

=== DOCUMENT 1 ===
{t1}

=== DOCUMENT 2 ===
{t2}

Return ONLY valid JSON as specified.
"""

    raw = call_groq_llm(COMPARISON_PROMPT, user_message, temperature=0.0)

    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "summary": raw,
            "verdict": "UNKNOWN",
            "verdict_reason": "",
            "added_clauses": [],
            "removed_clauses": [],
            "modified_clauses": [],
            "risk_comparison": {
                "doc1_risk": "UNKNOWN",
                "doc2_risk": "UNKNOWN",
                "doc1_score": 0,
                "doc2_score": 0
            },
            "recommendations": [],
            "parse_error": True
        }


# ============================================================
# MULTI-DOCUMENT UPLOAD
# ============================================================

def process_multiple_pdfs(pdf_paths: list, user_id: str, upload_folder: str) -> dict:
    """
    Process multiple PDFs and merge them into one FAISS vector store.
    Returns combined metadata.
    """
    all_documents = []
    all_texts = []
    filenames = []

    for pdf_path in pdf_paths:
        filename = Path(pdf_path).name
        filenames.append(filename)

        # Extract text
        full_text = extract_text_from_pdf(pdf_path)
        all_texts.append(full_text)

        # Chunk with document label
        chunks = chunk_document(full_text, filename)
        all_documents.extend(chunks)

    # Build combined vector store
    vector_store_path = os.path.join(upload_folder, f"vs_multi_{user_id}")
    build_vector_store(all_documents, vector_store_path)

    # Save combined text
    combined_text = "\n\n=== NEXT DOCUMENT ===\n\n".join(all_texts)
    text_path = os.path.join(upload_folder, f"text_multi_{user_id}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(combined_text)

    return {
        "success": True,
        "filenames": filenames,
        "total_chunks": len(all_documents),
        "vector_store_path": vector_store_path,
        "text_path": text_path,
        "doc_count": len(pdf_paths),
    }