# Document Loading & Splitting
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# import warnings
# warnings.simplefilter("ignore", UserWarning)

# Embedding & Vector Storage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 

# Environment Variables & API Setup
from dotenv import load_dotenv
import openai
import os
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
import pandas as pd

# Retrieval-Based QA Pipeline
from langchain.chains import RetrievalQA
# from langchain_community.chat_models import ChatOpenAI

# ROUGE Scoring for Evaluation
from rouge_score import rouge_scorer
from langchain_community.document_loaders import PyPDFLoader

def _load_env_variables():
    """
    Loads and retrieves environment variables from the .env file.
    Returns them as a dictionary.
    """
    load_dotenv()  # Load environment variables from .env file

    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "faiss_index_name": os.getenv("faiss_index_name", "faiss_index"),
        "document_store_name": os.getenv("document_store_name", "documents/"),
        "file_type": os.getenv("file_type", "pdf"),
        "chunk_size": int(os.getenv("chunk_size", 500)),  # Convert to integer
        "chunk_overlap": int(os.getenv("chunk_overlap", 100))  # Convert to integer
    }

def load_openai_api_key(openai_api_key):
    openai.api_key = openai_api_key  # Retrieve API key

    if openai.api_key:
        print("API Key loaded successfully!")
    else:
        print("Failed to load API Key. Check your .env file.")
    return openai

def load_all_pdfs_from_folder(folder_path: str):
    all_documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            try:
                loader = PyPDFLoader(filepath)
                documents = loader.load()
                all_documents.extend(documents)
                print(f"✅ Loaded {len(documents)} pages from {filename}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")

    return all_documents

def process_documents(document_store_name, file_type, chunk_size, chunk_overlap, faiss_index_name):
    all_documents = load_all_pdfs_from_folder(document_store_name)

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    
    docs = text_splitter.split_documents(all_documents)
    # Convert text chunks into embeddings and store them in FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save the FAISS index for later use
    vectorstore.save_local(faiss_index_name)
    print(f"FAISS index saved as '{faiss_index_name}'.")
    
    # Create a retriever
    retriever = vectorstore.as_retriever()

    print("Hello World End!")
    return retriever

def run_rag_pipeline(absolute_path, retriever, model_name="gpt-4", temperature=0):
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
    openai = load_openai_api_key(env_vars["OPENAI_API_KEY"])
    retriever = process_documents(document_store_name=env_vars["document_store_name"], 
                                  file_type=env_vars["file_type"], 
                                  chunk_size=env_vars["chunk_size"], 
                                  chunk_overlap=env_vars["chunk_overlap"], 
                                  faiss_index_name=env_vars["faiss_index_name"])
    results = run_rag_pipeline("./questions/Questions.csv", retriever)

    # del vectorstore
    # print("Document retrieval pipeline completed.")

if __name__ == "__main__":
    main()