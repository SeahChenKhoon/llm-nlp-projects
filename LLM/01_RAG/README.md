pip install uv
C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\python.exe -m venv .venv
.\\.venv\\Scripts\\activate
pip install uv
uv pip install -r requirements.txt
uv run rag_document_retrieval.py 

# 🔍 Project Overview: Retrieval-Augmented Generation Evaluation Pipeline
This project implements a Retrieval-Augmented Generation (RAG) evaluation framework that leverages large language models (LLMs), document embeddings, and robust evaluation metrics to assess the performance of generated responses against human-annotated answers.

# 🎯 Objectives
<li>Generate answers to questions using retrieved knowledge from a custom document store.</li>
<li>Evaluate answer quality using both ROUGE scores (for surface-level overlap) and RAGAS metrics (for deeper factual consistency and semantic relevance).</li>
<li>Support both batch mode (CSV-based) and interactive mode (live querying).</li>



# ⚙️ Key Features
## 1. Document Ingestion & Indexing
<li>Loads PDF files from a user-specified directory.</li>
<li>Splits documents into overlapping chunks using RecursiveCharacterTextSplitter.</li>
<li>Embeds chunks using OpenAIEmbeddings.</li>
<li>Stores vectorized chunks in a FAISS index for efficient semantic retrieval.</li>

## 2. Question Answering Pipeline
<li>Supports batch-mode inference using questions from a CSV file.</li>
<li>Uses LangChain's RetrievalQA to fetch relevant documents and generate answers via OpenAI’s ChatOpenAI models.</li>
<li>Prompting enforces short, concise answers (max 20 words).</li>

## 3. Evaluation Metrics
<li>ROUGE (rouge1, rouge2, rougeL): Measures lexical overlap between generated and human answers.</li>
  <li>RAGAS metrics:
    <ul>
      <li>answer_relevancy</li>
      <li>context_precision</li>
      <li>context_recall</li>
      <li>faithfulness</li>
      <li>semantic_similarity</li>
    </ul>
  </li>

## 4. Result Management
<li> Results are merged (ROUGE + RAGAS) and saved to a timestamped CSV file.</li> 
<li> Summary of averaged metrics is printed to the console for quick performance analysis.</li>

## 5. Interactive Mode
<li>Users can ask ad-hoc questions via terminal.</li>
<li>Model responses are generated in real-time based on document retrieval.</li>


# 📁 Project Structure Highlights

| Component | Description |
|---------|-------------|
| .env | Stores API keys, model name, paths, chunk config, etc. |
| load_pdfs() | Loads and parses PDFs into Document objects.
process_documents()	| Splits documents, embeds them, builds FAISS index. |
| run_rag_pipeline() | End-to-end batch evaluation using CSV questions and reference answers. |
| generate_results() | Runs LLM over each question and scores it with ROUGE. |
| evaluate_ragas_scores() | Applies RAGAS metrics for deep evaluation. |
| merge_and_store_results() | 	Combines all scores and exports CSV. |
| run_interactive_mode() | 	CLI interface for exploratory QA.
main()	| Orchestrates the app based on user mode selection. |


# 🧱 Dependencies
<li> langchain, langchain_community, langchain_openai </li>
<li> openai, pandas, datasets, ragas </li>
<li> rouge_score, faiss-cpu </li>
<li> python-dotenv (for managing .env configs) </li>

# 🚀 Example Use Case
A user working with a collection of internal PDFs can:
## 1. Drop files into a folder.
## 2. Specify chunking parameters and API keys in a .env file.
## 3. Run the pipeline to:
<li> Generate answers to a set of pre-written questions.</li>
<li> Evaluate how relevant, faithful, and semantically similar those answers are.</li>
<li> Review performance in an output CSV.</li>


# 📦 Deployment Environment Setup

## 🔧 1. Install Python 3.11

<li> Download and install Python 3.11.0 from [python.org](python.org). </li>

## 🔧 2. Setup Virtual Environment & Install Dependencies

```
pip install uv
{folder}\Python311\python.exe -m venv .venv
.\.venv\Scripts\activate
pip install uv
uv pip install -r requirements.txt
uv run rag_document_retrieval.py
```

### 📄 requirements.txt
```
langchain_community==0.3.20
langchain_openai==0.3.9
pandas==2.2.3
rouge_score==0.1.2
faiss-cpu==1.10.0
pypdf==5.4.0
ragas==0.2.14
```
