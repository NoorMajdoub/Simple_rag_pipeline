
# LangChain RAG Project

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using [LangChain](https://github.com/langchain-ai/langchain), [Chroma DB](https://github.com/chroma-core/chroma), and HuggingFace embeddings. The system allows you to load text documents, embed them, store them in a vector database, and later query them to generate answers based on relevant document chunks.

---

##  Project Structure

```
.
├── chroma/                      # Directory where Chroma DB persists
├── creating_the_database.py     # Script to create the vector database from documents
├── quering_the_data.py          # Script to query the vector DB and generate responses
├── data.txt           # Input text document(s)
└── README.md                    # Project documentation
```

---

##  Setup

1. **Install Dependencies**

```bash
pip install langchain langchain-community chromadb sentence-transformers langchain-openai
```

2. **Set OpenAI API Key**

For `ChatOpenAI`, export your OpenAI API key:

```bash
export OPENAI_API_KEY=your_openai_key_here
```

---

## How It Works

### Step 1: Create the Database

Run the script to load your `.txt` files, split them into chunks, embed them using `all-MiniLM-L6-v2`, and store them in Chroma.

```bash
python creating_the_database.py
```

### Step 2: Query the Database

Use natural language queries to retrieve relevant document chunks and generate an answer using OpenAI's chat model.

```bash
python quering_the_data.py
```

---

## 📝 Prompt Template

The following template is used to formulate the prompt passed to the language model:

```
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
```

