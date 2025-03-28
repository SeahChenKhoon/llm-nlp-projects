# Standard library
import logging
import os
from ast import literal_eval
from datetime import datetime
from typing import Any, Dict, List, Tuple

# Third-party packages
import openai
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    answer_similarity,
)

# LangChain core
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever

# LangChain integrations
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker

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
    Loads environment variables from a `.env` file and returns them as a dictionary.

    This function uses `python-dotenv` to load variables from a local `.env` file and
    retrieves key configuration settings used throughout the RAG pipeline.

    Returns:
        Dict[str, Any]: A dictionary containing the following keys:
            - 'openai_api_key' (str): OpenAI API key for authentication (e.g., "sk-proj-...").
            - 'faiss_index_name' (str): Directory name for storing the FAISS index (e.g., "faiss_index").
            - 'document_store_name' (str): Path to the folder containing input documents (e.g., "./documents").
            - 'file_type' (str): Glob pattern for document file types (e.g., "*.pdf").
            - 'model_name' (str): OpenAI model to use (e.g., "gpt-4").
            - 'temperature' (float): Sampling temperature for LLM response variability (e.g., 0.0).
            - 'question_store_name' (str): Path to the CSV file with questions and answers (e.g., "./questions/Questions.csv").
            - 'results_store_name' (str): Path to the output results CSV (e.g., "./results/result.csv").
    """
    load_dotenv()  # Load environment variables from .env file

    return {
        "openai_api_key": os.getenv("openai_api_key"),
        "faiss_index_name": os.getenv("faiss_index_name"),
        "document_store_name": os.getenv("document_store_name"),
        "file_type": os.getenv("file_type", "pdf"),
        "model_name": os.getenv("model_name"),
        "temperature": float(os.getenv("temperature")),
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
    faiss_index_name: str
) -> VectorStoreRetriever:
    """
    Processes PDF documents using semantic chunking, generates embeddings, and stores them in a FAISS index.

    This function:
      - Loads PDF documents from the given directory.
      - Splits them into semantically meaningful chunks using `SemanticChunker`.
      - Converts the chunks into OpenAI embeddings.
      - Stores the embeddings in a local FAISS index for retrieval.

    Args:
        openai_api_key (str): OpenAI API key used to initialize the embedding model.
        document_store_name (str): Path to the folder containing input PDF documents.
        faiss_index_name (str): Name for the FAISS index directory to save locally.

    Returns:
        VectorStoreRetriever: A retriever instance built from the FAISS vector store.
    """
    documents = load_pdfs(document_store_name)

    
    # Convert text chunks into embeddings and store them in FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Use SemanticChunker with percentile-based control
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95  # You can tune this
    )
    docs = text_splitter.split_documents(documents)

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


def load_questions(absolute_path: str) -> pd.DataFrame:
    """
    Loads a CSV file containing question-answer pairs into a pandas DataFrame.

    Args:
        absolute_path (str): The full path to the CSV file. The file is expected to contain
                             at least the columns 'Question' and 'Answer'.

    Returns:
        pd.DataFrame: A DataFrame containing the questions and corresponding reference answers.
    """
    return pd.read_csv(absolute_path)

def build_prompt(question: str) -> str:
    """
    Constructs a prompt for the language model using the given question.

    The prompt instructs the model to provide a concise response of no more than 20 words,
    formatted in a consistent style suitable for inference in a RAG pipeline.

    Args:
        question (str): The input question to be answered.

    Returns:
        str: A formatted prompt string to be used with the language model.
    """
    return f"""
    Provide a response to the following question in no more than 20 words.

    {question}

    Answer: """

def run_qa_chain(qa_chain, prompt: str) -> Tuple[str, List[str]]:
    result = qa_chain.invoke({"query": prompt})
    answer = result["result"]
    contexts = [doc.page_content for doc in result["source_documents"]]
    return answer, contexts

def compute_rouge_scores(human_answer: str, generated_answer: str) -> Dict[str, float]:
    """
    Computes ROUGE scores between a human reference answer and a generated answer.

    This function uses the ROUGE scorer to calculate ROUGE-1, ROUGE-2, and ROUGE-L F1 scores,
    which reflect the lexical overlap between the two texts at different granularities.

    Args:
        human_answer (str): The reference or ground truth answer.
        generated_answer (str): The answer generated by the model.

    Returns:
        Dict[str, float]: A dictionary containing:
            - 'rouge1': ROUGE-1 F1 score (unigram overlap)
            - 'rouge2': ROUGE-2 F1 score (bigram overlap)
            - 'rougeL': ROUGE-L F1 score (longest common subsequence)
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(human_answer, generated_answer)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }

def generate_results(
    qa_chain,
    questions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Runs a RAG QA chain on a set of questions and collects answers, contexts, and ROUGE scores.

    This function iterates over a DataFrame of questions and reference answers, constructs prompts,
    invokes the QA chain to generate answers and retrieve supporting contexts, and evaluates the
    generated answers using ROUGE metrics. It returns a DataFrame containing all results.

    Args:
        qa_chain: A LangChain `RetrievalQA` instance with `return_source_documents=True`.
        questions_df (pd.DataFrame): DataFrame containing the following columns:
            - 'Question': the user query
            - 'Answer': the reference or human-generated answer

    Returns:
        pd.DataFrame: A DataFrame containing:
            - 'question': the original question
            - 'ground_truth': the reference answer
            - 'answer': the model-generated answer
            - 'contexts': list of retrieved document snippets
            - 'rouge1': ROUGE-1 F1 score
            - 'rouge2': ROUGE-2 F1 score
            - 'rougeL': ROUGE-L F1 score
    """
    results = []
    for i, row in questions_df.iterrows():
        question = row["Question"]
        human_answer = row["Answer"]
        prompt = build_prompt(question)
        answer, contexts = run_qa_chain(qa_chain, prompt)
        rouge = compute_rouge_scores(human_answer, answer)

        logger.info("\n------------------------------------------------")
        logger.info(f"Question {i+1}: {question}")
        logger.info(f"Human Answer: {human_answer}")
        logger.info(f"Response: {answer}\n")
        logger.info(f"ROUGE-1: {rouge['rouge1']:.4f}")
        logger.info(f"ROUGE-2: {rouge['rouge2']:.4f}")
        logger.info(f"ROUGE-L: {rouge['rougeL']:.4f}")

        results.append({
            "question": question,
            "ground_truth": human_answer,
            "answer": answer,
            "contexts": contexts,
            **rouge
        })
    return pd.DataFrame(results)

def evaluate_ragas_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates the quality of a RAG (Retrieval-Augmented Generation) pipeline using RAGAS metrics.

    This function converts the input DataFrame into a Hugging Face `Dataset` compatible with RAGAS,
    ensuring the 'contexts' column is parsed as a list. It then evaluates the dataset using
    multiple RAGAS metrics including answer relevancy, context precision, context recall,
    faithfulness, and semantic similarity.

    Args:
        df (pd.DataFrame): DataFrame containing the following columns:
            - 'question': the input query
            - 'answer': the model-generated answer
            - 'contexts': list of retrieved context strings (or a stringified list)
            - 'ground_truth': reference/human answer

    Returns:
        pd.DataFrame: DataFrame containing the computed RAGAS metric scores for each sample.
    """    
    df["contexts"] = df["contexts"].apply(lambda x: x if isinstance(x, list) else literal_eval(x))
    logger.info("Evaluating RAGAS metrics...") 
    ragas_dataset = Dataset.from_pandas(df[["question", "answer", "contexts", "ground_truth"]])
    result = evaluate(ragas_dataset, metrics=[
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
        answer_similarity
    ])
    return result.to_pandas()


def merge_and_store_results(
    df_rouge: pd.DataFrame,
    df_ragas: pd.DataFrame,
    results_store_name: str
) -> pd.DataFrame:
    """
    Merges ROUGE and RAGAS evaluation results on their shared question fields and saves the
    combined result to a timestamped CSV file.

    This function performs an inner join between the RAGAS evaluation DataFrame and the ROUGE
    scores DataFrame using `user_input` from RAGAS and `question` from ROUGE. The merged
    DataFrame is saved in the specified directory with a filename that includes the current
    date and time.

    Args:
        df_rouge (pd.DataFrame): DataFrame containing ROUGE scores with a 'question' column.
        df_ragas (pd.DataFrame): DataFrame containing RAGAS metrics with a 'user_input' column.
        results_store_name (str): Base file path for saving the merged results (e.g., "./results/output.csv").

    Returns:
        pd.DataFrame: The merged DataFrame containing both RAGAS and ROUGE metrics.
    """    
    df_rouge = df_rouge[["question", "rouge1", "rouge2", "rougeL"]]
    merged_df = pd.merge(df_ragas, df_rouge, how='inner', left_on='user_input', right_on='question')
    merged_df.drop(columns=["question"], inplace=True)

    directory, filename = os.path.split(results_store_name)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = os.path.join(directory, f"{name}_{timestamp}{ext}")
    os.makedirs(directory, exist_ok=True)
    merged_df.to_csv(output_filename, index=False)

    return merged_df

def print_summary_metrics(df: pd.DataFrame) -> None:
    """
    Computes and prints the average performance scores from a DataFrame containing RAGAS and ROUGE metrics.

    This function calculates the mean values of key evaluation metrics such as answer relevancy,
    context precision, recall, faithfulness, semantic similarity, and ROUGE scores. The results
    are printed in a readable format for quick assessment of overall system performance.

    Args:
        df (pd.DataFrame): DataFrame containing evaluation metrics with the following columns:
            - 'answer_relevancy'
            - 'context_precision'
            - 'context_recall'
            - 'faithfulness'
            - 'semantic_similarity'
            - 'rouge1'
            - 'rouge2'
            - 'rougeL'

    Returns:
        None
    """    
    summary = {
        "answer_relevancy": df["answer_relevancy"].mean(),
        "context_precision": df["context_precision"].mean(),
        "context_recall": df["context_recall"].mean(),
        "faithfulness": df["faithfulness"].mean(),
        "semantic_similarity": df["semantic_similarity"].mean(),
        "ROUGE-1": df["rouge1"].mean(),
        "ROUGE-2": df["rouge2"].mean(),
        "ROUGE-L": df["rougeL"].mean(),
    }
    logger.info("\nOverall Performance Scores:")
    for key, value in summary.items():
        logger.info(f"{key}: {value:.4f}")


def run_rag_pipeline(
    absolute_path: str,
    retriever,
    model_name: str,
    temperature: float,
    results_store_name: str
) -> None:
    """
    Executes a Retrieval-Augmented Generation (RAG) pipeline using an LLM and FAISS retriever,
    performs both ROUGE and RAGAS evaluations, and stores the merged results.

    This function loads questions from a CSV file, uses a retrieval-based QA chain to generate
    answers, computes ROUGE and RAGAS metrics, saves the evaluated results to a timestamped CSV,
    and prints the overall performance summary.

    Args:
        absolute_path (str): Path to the input CSV file containing 'Question' and 'Answer' columns.
        retriever: LangChain-compatible retriever (e.g., FAISS retriever).
        model_name (str): OpenAI model name (e.g., "gpt-3.5-turbo", "gpt-4").
        temperature (float): Temperature setting for the LLM to control response variability.
        results_store_name (str): Base path and filename where evaluation results will be saved.

    Returns:
        None
    """    
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
    questions_df = load_questions(absolute_path)
    df_rouge = generate_results(qa_chain, questions_df)
    df_ragas = evaluate_ragas_scores(df_rouge)
    merged_df = merge_and_store_results(df_rouge, df_ragas, results_store_name)
    print_summary_metrics(merged_df)


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
        logger.info("Invalid input. Exiting.")    

if __name__ == "__main__":
    main()