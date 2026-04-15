# PhrasIQ RAG Pipeline

This repository contains a Retrieval-Augmented Generation (RAG) pipeline designed to answer complex questions across mixed financial data (CSV and unstructured text).

## System Architecture

### 1. Chunking Strategy
- **Structured Data (CSV)**: Each row in the `bu_financials_q3.csv` is treated as a single "Document". We convert the structured fields (Revenue, EBITDA, Variance, etc.) into a natural language description. This ensures that quantitative relationships (like specific variances) are semantically indexable.
- **Unstructured Data (Text)**: The `commentary_excerpts.txt` is split using the `--- DOCUMENT N ---` markers. Each excerpt is kept whole to preserve the internal context of the business unit's commentary.

### 2. Multi-Model Support
- **Cloud (GPT-5)**: The system supports **OpenAI's `gpt-5-mini`** for high-reasoning performance and reliability.
- **Local (Llama2)**: The system integrates with **Ollama** to run `llama2` locally, providing a private alternative for data processing.

### 3. Retrieval Method
- We utilize **FAISS** (Facebook AI Similarity Search) as our vector database. It is highly efficient for local development and provides fast similarity searching via Euclidean distance (default) or Inner Product.
- **Top-K Retrieval**: The system retrieves the top 5 most relevant documents to provide a comprehensive context for the LLM.

### 4. Grounding & Fallback
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

### 2. Tradeoffs
- **What it catches**: It provides maximum transparency. During a demo, it proves exactly *why* the AI decided to trust or reject a piece of information, while still providing a clean answer at the top.
- **What it misses**: This approach shows the "internal reasoning" to the user, which is a powerful pedagogical and demo tool, even if it might be condensed for a consumer-facing app.

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

