# PhrasIQ RAG Pipeline

This repository contains a Retrieval-Augmented Generation (RAG) pipeline designed to answer complex questions across mixed financial data (CSV and unstructured text).

## System Architecture

### 1. Chunking Strategy
- **Structured Data (CSV)**: Each row in the `bu_financials_q3.csv` is treated as a single "Document". We convert the structured fields (Revenue, EBITDA, Variance, etc.) into a natural language description. This ensures that quantitative relationships (like specific variances) are semantically indexable.
- **Unstructured Data (Text)**: The `commentary_excerpts.txt` is split using the `--- DOCUMENT N ---` markers. Each excerpt is kept whole to preserve the internal context of the business unit's commentary.

### 2. Model
- **OpenAI `gpt-5-mini`**: The system exclusively uses OpenAI's `gpt-5-mini` for high-reasoning performance and reliability.

### 3. Embeddings
- **OpenAI `text-embedding-ada-002`**: All documents (both structured CSV rows and unstructured commentary) are embedded using OpenAI via `OpenAIEmbeddings()` from LangChain. This model produces 1536-dimensional vectors that capture the semantic meaning of each document chunk, enabling accurate similarity-based retrieval even across heterogeneous data formats (financial metrics vs. narrative text).

### 4. Retrieval Method
- We utilize **FAISS** (Facebook AI Similarity Search) as our vector database. It is highly efficient for local development and provides fast similarity searching via Euclidean distance (default) or Inner Product.
- **Top-K Retrieval**: The system retrieves the top 5 most relevant documents to provide a comprehensive context for the LLM.

### 5. Grounding & Fallback
- **LCEL Chain**: The pipeline is built using LangChain Expression Language (LCEL). It uses a `RunnableParallel` structure to return both the **answer** and the **retrieved context documents**.
- **Source Citation**: The metadata from the retrieved documents is extracted and displayed to the user, identifying whether the answer came from the metric table or a specific commentary document.
- **Fallback Logic**: The system prompt explicitly instructs the LLM: *"If the answer is not contained within the context, clearly state that you do not have enough information to answer."*

## Part 2: Hallucination Mitigation Analysis


### 1. Mitigation Strategy: The Auditor Pattern (Internal Verification)
I implemented a **Multi-Step Verification (Self-Correction)** chain:
1. **Generator (Draft)**: Produces an initial answer based on context.
2. **Auditor (Lead)**: A second AI pass that explicitly audits the draft against the source context to produce a **Comprehensive Audited Response**. This output contains:
    - **### Final Polished Answer**: The clean, corrected result.
    - **### Verification Report**: A detailed breakdown of the verdict (Accurate/Hallucination) and the specific supporting checks performed.

### 2. What it shows
It provides maximum transparency. During a demo, it proves exactly *why* the AI decided to trust or reject a piece of information, while still providing a clean answer at the top. This approach shows the "internal reasoning" to the user, which is a powerful demo tool, even if it might be condensed for a consumer-facing app.

## Usage

1. Install dependencies: `pip install pandas langchain-openai FAISS-cpu langchain-community`
2. Set up your `.env` with `OPENAI_API_KEY`.
3. Run the Part 1 RAG: `python part1.py`
4. Run the Part 2 Mitigated RAG: `python part2.py`
5. **Interactive Demo (Recommended)**:
   ```bash
   streamlit run app.py
   ```
   *The app includes a sidebar toggle to switch between the Standard and Mitigated architectures live.*

## App Screenshots

### Standard RAG — Part 1
<img width="1112" height="561" alt="Screenshot 2026-04-14 214814" src="https://github.com/user-attachments/assets/4480ea06-e734-48e9-86aa-a445cc18a1f5" />

> The **Standard RAG (Part 1)** mode runs a single-pass retrieval chain. The user's question is embedded, the top-K most relevant documents are retrieved from FAISS, and the LLM generates a direct answer grounded in those sources. The right panel shows the cited source documents used to construct the response.

---

### Mitigated RAG — Part 2 (Auditor Pattern)
<img width="1114" height="554" alt="Screenshot 2026-04-14 214930" src="https://github.com/user-attachments/assets/1ba8b64e-577d-4336-aed9-c497ea9d054f" />

> The **Mitigated RAG (Part 2)** mode adds a two-step verification pipeline. A **Generator** LLM produces a draft answer, then a **Lead Auditor** LLM reviews it against the original retrieved context for hallucinations or unsupported claims. The output includes a **Final Polished Answer** (clean result) and a **Verification Report** (verdict + supporting checks), providing full transparency for auditable, high-stakes financial queries.
