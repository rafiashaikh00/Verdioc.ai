# Veridoc — Conversational PDF RAG Assistant

Veridoc is a conversational **Retrieval-Augmented Generation (RAG)** application that enables users to interact with PDF documents through natural-language questions.

Instead of sending an entire document to the language model, Veridoc retrieves relevant document chunks from a vector database, compresses the retrieved context, and uses that context together with conversation history to generate grounded answers.

The application is designed around a **local AI stack**, using Ollama for LLM inference and Hugging Face embeddings for semantic document retrieval.

---

## Overview

Veridoc addresses a common document-questioning problem: extracting useful information from lengthy PDFs without requiring the user to manually search through the document.

The application provides:

* Conversational question answering over uploaded PDFs
* Context-aware follow-up questions
* Semantic retrieval of relevant document chunks
* MMR-based retrieval
* Contextual compression of retrieved content
* Source references from retrieved document chunks
* Page-level references
* Local LLM inference through Ollama

---

## Architecture

```text
                     PDF Upload
                         │
                         ▼
                   PyPDFLoader
                         │
                         ▼
              Recursive Text Splitter
                         │
                         ▼
              Hugging Face Embeddings
                         │
                         ▼
                  Chroma Vector Store
                         │
                         ▼
                  MMR Retriever
                         │
                         ▼
             Contextual Compression
                         │
                         ▼
              Retrieved PDF Context
                         │
             ┌───────────┴───────────┐
             │                       │
       Chat History              User Query
             │                       │
             └───────────┬───────────┘
                         ▼
                  Chat Prompt
                         │
                         ▼
                  Ollama LLM
                         │
                         ▼
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
           Answer    References     Pages
```

---

## Key Technical Components

### 1. PDF Processing

Uploaded PDFs are temporarily stored and processed using `PyPDFLoader`.

The extracted document content is then divided into smaller chunks using `RecursiveCharacterTextSplitter`.

Current configuration:

* Chunk size: `200`
* Chunk overlap: `50`

Chunking allows the retrieval system to search for relevant sections of the document instead of processing the entire PDF for every question.

---

### 2. Semantic Embeddings

Veridoc uses:

`sentence-transformers/all-MiniLM-L6-v2`

to convert document chunks into numerical vector representations.

These embeddings allow semantically relevant content to be retrieved even when the user's question does not contain the exact wording used in the PDF.

---

### 3. Vector Database

The generated embeddings are stored in **Chroma**.

When a user submits a question, Chroma is queried through a LangChain retriever to identify the most relevant document chunks.

---

### 4. MMR Retrieval

Veridoc uses **Maximum Marginal Relevance (MMR)** retrieval.

Current configuration:

```python
search_type="mmr"

search_kwargs={
    "fetch_k": 10,
    "k": 3,
    "lambda_mult": 1
}
```

The retriever first considers a larger candidate pool and then selects the most relevant chunks for the question.

---

### 5. Contextual Compression

Retrieved chunks are passed through LangChain's `ContextualCompressionRetriever`.

The compression layer uses:

`LLMChainExtractor`

to extract the parts of retrieved documents that are most relevant to the user's question.

This helps reduce unnecessary information before the final generation step.

---

### 6. Conversational Context

Veridoc maintains the current conversation using Streamlit session state.

Previous user questions and AI responses are stored as:

* `HumanMessage`
* `AIMessage`

This conversation history is passed to the prompt through `MessagesPlaceholder`, allowing the model to understand follow-up questions.

For example:

```text
User:
What is the main topic of this PDF?

AI:
...

User:
What are its main characteristics?

AI:
...
```

The second question can be interpreted using the previous conversation.

---

## Grounded Answer Generation

The system prompt instructs the model to:

* Answer using the retrieved PDF context
* Use its own wording
* Avoid unnecessary copying of the document
* Avoid outside knowledge
* Avoid hallucinating information
* Use conversation history for follow-up questions
* Clearly state when the requested information is unavailable in the provided PDF

The final answer is generated using a local **Llama 3.2 1B** model through Ollama.

---

## Source References

Veridoc separates the generated answer from the retrieved source material.

For every response, the interface can display:

**Answer**

The model-generated response.

**Reference**

The retrieved PDF chunks used as supporting context.

**Page**

The corresponding PDF page information extracted from document metadata.

This provides users with additional visibility into the retrieved evidence behind an answer.

---

## Technology Stack

| Technology                     | Purpose                       |
| ------------------------------ | ----------------------------- |
| Python                         | Application development       |
| Streamlit                      | Interactive web interface     |
| LangChain                      | RAG pipeline orchestration    |
| PyPDFLoader                    | PDF document loading          |
| RecursiveCharacterTextSplitter | Document chunking             |
| Hugging Face Embeddings        | Semantic embeddings           |
| Chroma                         | Vector storage and retrieval  |
| MMR                            | Diverse/relevant retrieval    |
| ContextualCompressionRetriever | Retrieved-context compression |
| Ollama                         | Local LLM inference           |
| Llama 3.2 1B                   | Answer generation             |

---

## Project Structure

```text
Veridoc/
│
├── app.py
├── requirements.txt
├── README.md
└── ...
```

---

## Running Locally

### Prerequisites

Make sure the following are installed:

* Python
* Ollama
* Llama 3.2 1B model
* Required Python dependencies

### Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Make sure the required Ollama model is available locally.

Then start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Current Limitations

The current version intentionally focuses on a simple and understandable PDF RAG workflow.

Current limitations include:

* PDF is the supported document type
* Retrieval quality depends on chunking and embedding quality
* Multiple retrieved chunks may originate from the same PDF page
* The current implementation uses a lightweight local LLM
* Advanced retrieval techniques such as hybrid search and dedicated reranking are not currently implemented

These areas provide opportunities for future iterations of the system.

---

## Future Improvements

Potential future improvements include:

* Hybrid keyword + semantic retrieval
* Dedicated reranking models
* Retrieval evaluation
* Improved citation handling
* Better duplicate-source handling
* Support for additional document formats
* More advanced conversational memory
* Evaluation datasets and retrieval metrics

---

## Project Goal

Veridoc was built to explore the practical engineering of **Retrieval-Augmented Generation systems**, from document ingestion and vector indexing to retrieval, contextual compression, prompt construction, conversational memory, and grounded response generation.

The project emphasizes understanding the complete RAG pipeline rather than relying on a black-box implementation.

---

## Author

**Rafia Shaikh**

Computer Science Student | AI & Generative AI

ssistant that lets users chat with their PDF documents using local LLMs, semantic retrieval, contextual compression, and source references with page numbers.
