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
    """Parses the CSV and converts each row into a descriptive Document."""
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        # Create a descriptive text for the row to make it searchable
        content = (
            f"Business Unit: {row['Business Unit']}, Quarter: {row['Quarter']}. "
            f"Revenue Actual: ${row['Revenue_Actual ($M)']}M, Revenue Plan: ${row['Revenue_Plan ($M)']}M, "
            f"Revenue Variance: {row['Revenue_Variance ($M)']}M ({row['Revenue_Variance_Pct']}). "
            f"EBITDA Actual: ${row['EBITDA_Actual ($M)']}M, EBITDA Plan: ${row['EBITDA_Plan ($M)']}M, "
            f"EBITDA Variance: {row['EBITDA_Variance ($M)']}M ({row['EBITDA_Variance_Pct']}). "
            f"Headcount: {row['Headcount_Actual']} (Plan: {row['Headcount_Plan']}, Var: {row['Headcount_Variance']}). "
            f"Notes: {row['Notes']}"
        )
        metadata = {
            "source": os.path.basename(csv_path),
            "type": "financial_metric",
            "business_unit": row['Business Unit'],
            "quarter": row['Quarter']
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def load_commentary_data(txt_path):
    """Splits the commentary text into separate Documents based on headers."""
    with open(txt_path, 'r') as f:
        content = f.read()
    
    # Split by the document marker
    parts = content.split("--- DOCUMENT ")
    documents = []
    for part in parts:
        if not part.strip():
            continue
        
        # Reconstruct the document ID and content
        header_end = part.find(" ---")
        doc_id = part[:header_end].strip()
        doc_body = part[header_end + 4:].strip()
        
        metadata = {
            "source": os.path.basename(txt_path),
            "doc_id": f"Document {doc_id}",
            "type": "commentary"
        }
        documents.append(Document(page_content=doc_body, metadata=metadata))
    return documents

def build_rag_system():
    # Detect the correct path for data files
    csv_path = "bu_financials_q3.csv" if os.path.exists("bu_financials_q3.csv") else "PhrasIQ/bu_financials_q3.csv"
    txt_path = "commentary_excerpts.txt" if os.path.exists("commentary_excerpts.txt") else "PhrasIQ/commentary_excerpts.txt"
    
    # Load data
    csv_docs = load_financial_data(csv_path)
    txt_docs = load_commentary_data(txt_path)
    all_docs = csv_docs + txt_docs
    
    # Initialize Vector Store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Initialize LLM (GPT-5 exclusively)
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
    # Define Prompt
    system_prompt = (
        "You are a financial analyst assistant. Use the following pieces of retrieved context to answer the question. "
        "If the answer is not contained within the context, clearly state that you do not have enough information to answer. "
        "Do not fabricate data or explanations. "
        "\n\n"
        "Context:\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create Chain using LCEL
    rag_chain = (
        RunnableParallel({
            "context": retriever,
            "input": RunnablePassthrough()
        })
        | {
            "answer": prompt | llm | StrOutputParser(),
            "context": lambda x: x["context"]
        }
    )
    
    return rag_chain

def run_assessment_query(chain):
    question = (
        "Which business unit had the largest unfavorable variance in Q3, and "
        "what factors does the commentary attribute this to?"
    )
    
    print(f"\nQUERY: {question}\n")
    response = chain.invoke(question)
    
    print("--- ANSWER ---")
    print(response["answer"])
    print("\n--- SOURCES ---")
    seen_sources = set()
    for doc in response["context"]:
        source_id = doc.metadata.get("doc_id") or doc.metadata.get("source")
        if source_id not in seen_sources:
            print(f"- {source_id}")
            seen_sources.add(source_id)

if __name__ == "__main__":
    # Test with default OpenAI
    print("--- [TESTING WITH OPENAI] ---")
    chain_openai = build_rag_system()
    run_assessment_query(chain_openai)
