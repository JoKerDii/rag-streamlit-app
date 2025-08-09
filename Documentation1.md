# Documentation I

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines:
- **Retrieval**: Finding relevant information from a knowledge base
- **Generation**: Using an LLM to generate answers based on retrieved information

Think of it like having a research assistant who:
1. Searches through documents to find relevant information
2. Uses that information to write a comprehensive answer

## System Architecture Overview

This codebase implements **4 different RAG approaches**:

1. **Simple RAG** - Basic vector search + LLM generation
2. **RAG with Neo4j** - Knowledge graph-based RAG
3. **Adaptive RAG** - Smart routing between different data sources
4. **RAG with Ranking** - Enhanced retrieval with document reranking

## Core Libraries and Their Purposes

### UI and Application Framework
```python
import streamlit as st  # Web app framework
from streamlit_option_menu import option_menu  # Navigation menu
```

### Document Processing
```python
from langchain_community.document_loaders import DirectoryLoader  # Load documents from folders
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split documents into chunks
import tempfile  # Create temporary files
from pathlib import Path  # Handle file paths
```

### Vector Storage and Embeddings
```python
from langchain_community.vectorstores import FAISS  # Vector database
from langchain_huggingface import HuggingFaceEmbeddings  # Convert text to vectors
```

### LLM and Chains
```python
from langchain.chains import ConversationalRetrievalChain  # RAG chain
from langchain_groq import ChatGroq  # LLM provider
```

### Knowledge Graph (Neo4j)
```python
from py2neo import Graph  # Neo4j database connection
from langchain_community.graphs import Neo4jGraph  # LangChain Neo4j integration
```

### Environment and Configuration
```python
from dotenv import load_dotenv  # Load environment variables
import os  # Operating system interface
```

## Detailed Code Walkthrough

### 1. Application Structure (Streamlit_RAG.py)

#### Main Navigation Function
```python
def streamlit_ui():
    with st.sidebar:
        choice = option_menu('Navigation',["Home",'Simple RAG','RAG with Neo4J','AdaptiveRAG','RAG_Ranking'])
```

This creates a sidebar menu that lets users choose between different RAG implementations.

#### Simple RAG Implementation

Let's understand the `RAG()` function step by step:

```python
def RAG(docs):
    # Step 1: Save uploaded files temporarily
    for source_docs in docs:
        with tempfile.NamedTemporaryFile(delete=False,dir=TMP_DIR.as_posix(),suffix='.pdf') as temp_file:
            temp_file.write(source_docs.read())
```

**What's happening here?**
- Users upload PDF files through Streamlit
- We save these files temporarily so we can process them
- `tempfile.NamedTemporaryFile` creates a temporary file that won't be deleted immediately

```python
    # Step 2: Load documents from the temporary directory
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf', show_progress=True)
    documents = loader.load()
```

**What's happening?**
- `DirectoryLoader` scans the temp directory for PDF files
- `glob='**/*.pdf'` means "find all PDF files in any subdirectory"
- Returns a list of Document objects

```python
    # Step 3: Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(documents)
```

**Why split documents?**
- Large documents are too big for LLMs to process effectively
- We break them into 1000-character chunks
- `chunk_overlap=0` means no overlap between chunks (could be 100-200 for better context)

```python
    # Step 4: Create embeddings and vector store
    embedding = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2',
                                     model_kwargs={'device':'cpu'})
    db = FAISS.from_documents(text,embedding)
    db.save_local(DB_FAISS_PATH)
```

**What are embeddings?**
- Convert text into numerical vectors (arrays of numbers)
- Similar texts have similar vectors
- `sentence-transformers/all-MiniLM-L6-v2` is a pre-trained model that's good at this
- FAISS is a fast vector search library

```python
    # Step 5: Create the RAG chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        db.as_retriever(search_kwargs={'k':20}),
        return_source_documents=True
    )
```

**What's a ConversationalRetrievalChain?**
- Combines retrieval (finding relevant documents) with generation (creating answers)
- `search_kwargs={'k':20}` means "find the 20 most similar document chunks"
- `return_source_documents=True` lets us show users where answers came from

### 2. Knowledge Graph RAG (KnowledgeGrpah_Neo4j.py)

This is more advanced - it creates a knowledge graph from documents.

#### What's a Knowledge Graph?
Instead of just storing text chunks, we extract:
- **Entities** (people, organizations, concepts)
- **Relationships** (connections between entities)

Example: "John works at OpenAI" becomes:
- Entity: John (Person)
- Entity: OpenAI (Organization)  
- Relationship: John -[WORKS_AT]-> OpenAI

#### Key Components:

```python
class RAG_Graph:
    def __init__(self):
        os.environ["NEO4J_URI"] = "bolt://localhost:7687"
        self.graph = Neo4jGraph()
        self.llm = ChatGroq(...)
```

**Neo4j Setup:**
- Neo4j is a graph database
- `bolt://localhost:7687` is the default connection
- Requires Neo4j to be running locally

```python
def create_graph(self,docs,TMP_DIR):
    # Load and split documents
    loader = DirectoryLoader(TMP_DIR.as_posix(),glob='**/*docx',show_progress=True)
    texts = text_splitter.split_documents(document)
    
    # Extract graph structure using LLM
    llm_transformer = LLMGraphTransformer(llm=self.llm)
    graph_documents = llm_transformer.convert_to_graph_documents(texts)
    
    # Store in Neo4j
    self.graph.add_graph_documents(graph_documents)
```

**How does LLMGraphTransformer work?**
- Sends text to the LLM with instructions to extract entities and relationships
- LLM identifies important concepts and how they connect
- Creates a structured graph representation

### 3. Adaptive RAG (Adaptive_RAG.py)

This implements "smart" RAG that decides whether to:
- Search your documents (vectorstore)
- Search the web
- Rewrite questions for better results

#### Query Router
```python
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

def query_analyzer(question):
    structured_llm_router = llm.with_structured_output(RouteQuery)
    # ... prompt setup
    return question_router.invoke({"question": question})
```

**How it works:**
- LLM analyzes the question
- Decides if it's about your documents or needs web search
- Routes accordingly

#### Document Grading
```python
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
```

**Quality Control:**
- After retrieving documents, LLM grades their relevance
- Only uses relevant documents for answer generation
- Improves answer quality

#### State Graph (LangGraph)
```python
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retriever: VectorStoreRetriever
```

**LangGraph creates a workflow:**
1. Route question
2. Retrieve documents OR web search
3. Grade document relevance
4. Generate answer
5. Check for hallucinations
6. If answer is bad, try again with modified query

## How to Write This Yourself

### Step 1: Start Simple
```python
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Basic RAG implementation
def simple_rag():
    # 1. Load document
    loader = TextLoader("your_document.txt")
    documents = loader.load()
    
    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # 4. Create QA chain
    llm = ChatGroq(api_key="your_api_key", model_name="llama3-70b-8192")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa_chain
```

### Step 2: Add Streamlit Interface
```python
def main():
    st.title("My RAG Chatbot")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a document", type=['txt', 'pdf'])
    
    if uploaded_file:
        # Process file and create QA chain
        qa_chain = simple_rag()
        
        # Chat interface
        if prompt := st.chat_input("Ask a question"):
            response = qa_chain.run(prompt)
            st.write(response)

if __name__ == "__main__":
    main()
```

### Step 3: Add More Features Gradually
1. **Multiple file types** (PDF, DOCX)
2. **Better text splitting** strategies
3. **Document ranking** with Cohere
4. **Knowledge graphs** with Neo4j
5. **Adaptive routing** with LangGraph

## Key Concepts to Master

### 1. Vector Embeddings
- Text → Numbers that capture meaning
- Similar meanings = similar vectors
- Enable semantic search

### 2. Chunking Strategies
- **Fixed size**: Simple but may break context
- **Recursive**: Tries to preserve sentence/paragraph boundaries
- **Semantic**: Splits based on topic changes

### 3. Retrieval Methods
- **Similarity search**: Find most similar vectors
- **Hybrid search**: Combine keyword + semantic search
- **Reranking**: Use another model to reorder results

### 4. Chain Types
- **Stuff**: Put all context in one prompt
- **Map-reduce**: Summarize chunks, then combine
- **Refine**: Iteratively improve answer

## Environment Setup

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
TAVILY_API_KEY=your_tavily_api_key
NEO4J_PASSWORD=your_neo4j_password
```

Install dependencies:
```bash
pip install streamlit langchain langchain-community langchain-groq
pip install faiss-cpu sentence-transformers
pip install neo4j py2neo  # For knowledge graph
pip install langgraph  # For adaptive RAG
```

## Practice Exercises

1. **Build Basic RAG**: Start with a simple text file and basic QA
2. **Add PDF Support**: Handle PDF document uploads
3. **Improve Chunking**: Experiment with different chunk sizes
4. **Add Source Citations**: Show which document chunks were used
5. **Implement Ranking**: Add Cohere reranking
6. **Try Knowledge Graphs**: Set up Neo4j and extract entities

## Common Patterns

### Error Handling
```python
try:
    # RAG operations
    result = qa_chain.run(question)
except Exception as e:
    st.error(f"Error: {e}")
    return "Sorry, I couldn't process your question."
```

### Session State Management
```python
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add to chat history
st.session_state.messages.append({"role": "user", "content": prompt})
```

### Configuration Management
```python
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

This tutorial should give you a solid foundation to understand and eventually recreate this RAG system. Start with the simple version and gradually add complexity as you master each concept!