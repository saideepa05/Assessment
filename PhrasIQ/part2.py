import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 1. Configuration & Setup
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_financial_data(csv_path):
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        content = (
            f"Business Unit: {row['Business Unit']}, Quarter: {row['Quarter']}. "
            f"Revenue Actual: ${row['Revenue_Actual ($M)']}M, Revenue Plan: ${row['Revenue_Plan ($M)']}M, "
            f"Revenue Variance: {row['Revenue_Variance ($M)']}M ({row['Revenue_Variance_Pct']}). "
            f"EBITDA Actual: ${row['EBITDA_Actual ($M)']}M, EBITDA Plan: ${row['EBITDA_Plan ($M)']}M, "
            f"EBITDA Variance: {row['EBITDA_Variance ($M)']}M ({row['EBITDA_Variance_Pct']}). "
            f"Headcount: {row['Headcount_Actual']} (Plan: {row['Headcount_Plan']}, Var: {row['Headcount_Variance']}). "
            f"Notes: {row['Notes']}"
        )
        metadata = {"source": os.path.basename(csv_path), "type": "financial_metric"}
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def load_commentary_data(txt_path):
    with open(txt_path, 'r') as f:
        content = f.read()
    parts = content.split("--- DOCUMENT ")
    documents = []
    for part in parts:
        if not part.strip(): continue
        header_end = part.find(" ---")
        doc_id = part[:header_end].strip()
        doc_body = part[header_end + 4:].strip()
        metadata = {"source": os.path.basename(txt_path), "doc_id": f"Document {doc_id}", "type": "commentary"}
        documents.append(Document(page_content=doc_body, metadata=metadata))
    return documents

def build_mitigated_rag_system():
    # Detect the correct path for data files
    csv_path = "bu_financials_q3.csv" if os.path.exists("bu_financials_q3.csv") else "PhrasIQ/bu_financials_q3.csv"
    txt_path = "commentary_excerpts.txt" if os.path.exists("commentary_excerpts.txt") else "PhrasIQ/commentary_excerpts.txt"
    
    docs = load_financial_data(csv_path) + load_commentary_data(txt_path)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

    # CHAIN 1: PRELIMINARY GENERATION ---
    gen_system_prompt = (
        "You are a financial analyst. Answer the question based ONLY on the context below. "
        "If the answer is not in the context, say 'I don't know'.\n\nContext:\n{context}"
    )
    gen_prompt = ChatPromptTemplate.from_messages([("system", gen_system_prompt), ("human", "{input}")])

    # CHAIN 2: AUDITOR / COMPREHENSIVE REPORT ---
    audit_system_prompt = (
        "You are a Lead Financial Auditor. You will be given a 'Draft Answer'. "
        "Review it against the 'Source Context' for accuracy and hallucinations.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. You MUST NOT use internal knowledge or training data to answer. Only use the 'Source Context'.\n"
        "2. If the user asks a question about general knowledge (e.g., 'Who is the President?') that is not in the context, "
        "your 'Final Polished Answer' MUST be a refusal (e.g., 'I don't have enough information').\n"
        "3. Your output MUST contain two distinct sections:\n\n"
        "### Final Polished Answer:\n"
        "[Provide the clean, corrected final answer here, strictly grounded in the context. If the info is missing, refuse to answer.]\n\n"
        "### Verification Report:\n"
        "**Verdict**: [Accurate / Hallucination Detected / Missing Info]\n"
        "**Supporting Checks**:\n"
        "- [Check 1 details]\n"
        "- [Check 2 details]\n"
        "\nSource Context:\n{context}"
    )
    audit_prompt = ChatPromptTemplate.from_messages([
        ("system", audit_system_prompt),
        ("human", "Draft Answer to Review: {draft_answer}\n\nOriginal Question: {input}")
    ])

    # INTEGRATED LCEL PIPELINE (2-Step Audit Report) 
    preliminary_chain = (
        RunnableParallel({"context": retriever, "input": RunnablePassthrough()})
        | {
            "draft_answer": gen_prompt | llm | StrOutputParser(),
            "context": lambda x: x["context"],
            "input": lambda x: x["input"]
        }
    )

    final_chain = (
        preliminary_chain
        | {
            "answer": audit_prompt | llm | StrOutputParser(),
            "context": lambda x: x["context"]
        }
    )
    
    return final_chain

def run_mitigation_test(chain):
    # Testing the Failure Mode Query
    question = "What was the CEO's guidance on the Q3 variance and what remediation steps did she commit to?"
    
    print(f"\nQUERY: {question}\n")
    response = chain.invoke(question)
    
    print("--- VERIFICATION REPORT ---")
    try:
        print(response["answer"])
    except UnicodeEncodeError:
        print(response["answer"].encode('ascii', 'ignore').decode('ascii'))
    
    print("\n--- SOURCES CITED ---")
    seen_sources = set()
    for doc in response["context"]:
        sid = doc.metadata.get("doc_id") or doc.metadata.get("source")
        if sid not in seen_sources:
            print(f"- {sid}")
            seen_sources.add(sid)

if __name__ == "__main__":
    chain = build_mitigated_rag_system()
    run_mitigation_test(chain)