# Standard library
import os

# Third-party packages
import pandas as pd
import openai
from dotenv import load_dotenv
from rouge_score import rouge_scorer

# LangChain core
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain integrations
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

def _load_env_variables():
    load_dotenv()  # Load environment variables from .env file

    return {
        "openai_api_key": os.getenv("openai_api_key"),
        "faiss_index_name": os.getenv("faiss_index_name"),
        "document_store_name": os.getenv("document_store_name"),
        "file_type": os.getenv("file_type", "pdf"),
        "chunk_size": int(os.getenv("chunk_size", 500)), 
        "chunk_overlap": int(os.getenv("chunk_overlap", 100)),
        "model_name": os.getenv("model_name"),
        "temperature": os.getenv("temperature"),
        "question_store_name": os.getenv("question_store_name"),
    }


def load_pdfs(folder_path: str):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            try:
                loader = PyPDFLoader(filepath)
                documents = loader.load()
                documents.extend(documents)
                print(f"Loaded {len(documents)} pages from {filename}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    return documents

def process_documents(openai_api_key, document_store_name, chunk_size, chunk_overlap, faiss_index_name):
    documents = load_pdfs(document_store_name)

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    
    # Convert text chunks into embeddings and store them in FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_index_name)

    print(f"FAISS index saved as '{faiss_index_name}'.")
    return vectorstore.as_retriever()

def run_rag_pipeline(absolute_path, retriever, model_name, temperature):
    # Initialize OpenAI LLM
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    # Build the RAG pipeline
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    
    # Load questions from CSV
    questions_df = pd.read_csv(absolute_path)
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    results = []
    
    for question_no, row in questions_df.iterrows():
        question = row["Question"]
        human_answer = row["Answer"]

        # Construct prompt
        prompt = f"""
            Provide a response to the following question in no more than 20 words.

            {question}

            Answer: """

        # Generate response
        response = qa_chain.invoke({"query": prompt})

        # Compute ROUGE scores
        rouge_scores = scorer.score(human_answer, response["result"])

        # Print results
        print(f"Question {question_no + 1}: {question}")
        print(f"Human Answer: {human_answer}")
        print(f"Response: {response}\n")
        print(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}")
        print(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")
        print("\n------------------------------------------------")

        # Store results as a dictionary
        results.append({
            "question": question,
            "human_answer": human_answer,
            "response": response,
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
        })
    
    return results

def main():
    print("Loading environment variables...")
    env_vars = _load_env_variables()
    retriever = process_documents(openai_api_key=env_vars["openai_api_key"],
                                  document_store_name=env_vars["document_store_name"], 
                                  chunk_size=env_vars["chunk_size"], 
                                  chunk_overlap=env_vars["chunk_overlap"], 
                                  faiss_index_name=env_vars["faiss_index_name"])
    results = run_rag_pipeline(absolute_path=env_vars["question_store_name"], 
                               retriever=retriever,
                               model_name=env_vars["model_name"],
                               temperature=env_vars["temperature"])

if __name__ == "__main__":
    main()