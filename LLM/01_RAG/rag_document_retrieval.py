# Standard library
import os
from typing import Dict, Any, List
import logging
from ast import literal_eval
from datetime import datetime
from datasets import Dataset

# Third-party packages
import pandas as pd
import openai
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    answer_similarity,
)
from ragas import evaluate

# LangChain core
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever

# LangChain integrations
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.INFO)

def _load_env_variables() -> Dict[str, Any]:
    """
    Load environment variables from a `.env` file and return them as a dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing configuration values including:
            - openai_api_key (str | None): API key for OpenAI.
            - faiss_index_name (str | None): Name of the FAISS index.
            - document_store_name (str | None): Name of the document store.
            - file_type (str): File type to load, default is "pdf".
            - chunk_size (int): Number of characters in each document chunk.
            - chunk_overlap (int): Number of overlapping characters between chunks.
            - model_name (str | None): Name of the OpenAI model to use.
            - temperature (str | None): Temperature setting for model response variability.
            - question_store_name (str | None): Name of the question store.
    """
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
        "results_store_name": os.getenv("results_store_name")
    }


def load_pdfs(folder_path: str) -> List[Document]:
    """
    Load all PDF files from the given folder and return a list of Document objects.

    Args:
        folder_path (str): The path to the folder containing PDF files.

    Returns:
        List[Document]: A list of parsed documents from all PDF files.
    """    
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            try:
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
                logger.info(f"Loaded {len(documents)} pages from {filename}")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")

    return documents

def process_documents(    
    openai_api_key: str,
    document_store_name: str,
    chunk_size: int,
    chunk_overlap: int,
    faiss_index_name: str
) -> VectorStoreRetriever:
    documents = load_pdfs(document_store_name)

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    
    # Convert text chunks into embeddings and store them in FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_index_name)

    logger.info(f"FAISS index saved as '{faiss_index_name}'.")
    return vectorstore.as_retriever()

# Convert context column from string to list if needed
def parse_context(val):
    try:
        # If it's a stringified list (e.g., "['a', 'b']")
        return literal_eval(val)
    except Exception:
        # If it's semicolon-separated
        return val.split(";") if isinstance(val, str) else []


def run_rag_pipeline(
    absolute_path: str,
    retriever: VectorStoreRetriever,
    model_name: str,
    temperature: float,
    results_store_name:str)-> List[Dict[str, Any]]:
        
    # Initialize OpenAI LLM
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    # Build the RAG pipeline
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=retriever,
        return_source_documents=True
    )
    
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
        contexts = [doc.page_content for doc in response["source_documents"]]

        # Compute ROUGE scores
        rouge_scores = scorer.score(human_answer, response["result"])

        # Print results
        logger.info("\n------------------------------------------------")
        logger.info(f"Question {question_no + 1}: {question}")
        logger.info(f"Human Answer: {human_answer}")
        logger.info(f"Response: {response['result']}\n")
        logger.info(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}")
        logger.info(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}")
        logger.info(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")
        
        # Store results as a dictionary
        results.append({
            "question": question,
            "ground_truth": human_answer,
            "answer": response["result"],
            "contexts": contexts,
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure
        })

    df = pd.DataFrame(results)
    ragas_dataset = Dataset.from_pandas(df[["question", "answer", "contexts", "ground_truth"]])

    # Evaluate using RAGAS
    ragas_result = evaluate(
        ragas_dataset,
        metrics=[
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
            answer_similarity
        ]
    )
    print(ragas_result.to_pandas())

    # Extract the directory and base filename
    directory, filename = os.path.split(results_store_name)
    name, ext = os.path.splitext(filename)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Create the new path with timestamp
    output_filename = os.path.join(directory, f"{name}_{timestamp}{ext}")

    df = pd.DataFrame(results)
    df.to_csv(output_filename, index=False)

    overall_rouge = {
        "ROUGE-1": df["rouge1"].mean(),
        "ROUGE-2": df["rouge2"].mean(),
        "ROUGE-L": df["rougeL"].mean(),
    }

    print("\nOverall ROUGE Scores:")
    for key, value in overall_rouge.items():
        print(f"{key}: {value:.4f}")

    return None

def run_interactive_mode(
    retriever: VectorStoreRetriever,
    model_name: str,
    temperature: float
) -> None:
    """
    Launches an interactive terminal-based interface for asking questions 
    using a Retrieval-Augmented Generation (RAG) pipeline.

    Args:
        retriever (VectorStoreRetriever): A retriever object that performs document retrieval.
        model_name (str): The name of the OpenAI model to use (e.g., 'gpt-3.5-turbo' or 'gpt-4').
        temperature (float): The temperature value for controlling randomness in LLM responses.

    Returns:
        None
    """
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    logger.info("\nInteractive mode. Type your question or 'exit' to quit.")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() in {"exit", "quit"}:
            logger.info("Exiting...")
            break

        prompt = f"""
        Provide a response to the following question in no more than 20 words.

        {question}

        Answer:"""

        response = qa_chain.invoke({"query": prompt})
        logger.info(f"\nResponse: {response['result']}")


def main() -> None:
    """
    Entry point for the RAG document retrieval application.

    This function:
    - Loads environment variables
    - Builds a retriever from the provided documents
    - Allows the user to choose between batch mode (from CSV) or interactive mode
    - Executes the appropriate pipeline based on user selection
    """
    logger.info("Loading environment variables...")
    env_vars = _load_env_variables()
    retriever = process_documents(openai_api_key=env_vars["openai_api_key"],
                                  document_store_name=env_vars["document_store_name"], 
                                  chunk_size=env_vars["chunk_size"], 
                                  chunk_overlap=env_vars["chunk_overlap"], 
                                  faiss_index_name=env_vars["faiss_index_name"])
    mode = input("Enter 'batch' to run from CSV or 'interactive' to type questions: ").strip().lower()
    
    if mode == "batch":
        results = run_rag_pipeline(absolute_path=env_vars["question_store_name"], 
                                   retriever=retriever,
                                   model_name=env_vars["model_name"],
                                   temperature=float(env_vars["temperature"]),
                                   results_store_name=env_vars["results_store_name"])
    elif mode == "interactive":
        run_interactive_mode(retriever=retriever,
                             model_name=env_vars["model_name"],
                             temperature=float(env_vars["temperature"]))
    else:
        print("Invalid input. Exiting.")    

if __name__ == "__main__":
    main()