# Documentation II


This is a sophisticated **Retrieval-Augmented Generation (RAG)** application built with Streamlit that demonstrates multiple RAG techniques. Here's a detailed breakdown:

### **Overall Architecture**

The application is structured as a **multi-page Streamlit app** with 5 different RAG implementations:

1. **Simple RAG** - Basic vector-based retrieval
2. **RAG with Neo4j** - Knowledge graph-based retrieval
3. **Adaptive RAG** - Self-corrective RAG with LangGraph
4. **RAG with Ranking** - Enhanced retrieval with Cohere reranking
5. **Home** - Landing page

### **Key Components**

#### **1. Dependencies and Setup**
```python
# Core RAG libraries
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

# LLM integration
from langchain_groq import ChatGroq  # Using Groq's LLM

# Knowledge graph
from py2neo import Graph
import networkx as nx

# UI components
import streamlit as st
from streamlit_option_menu import option_menu
```

#### **2. Main UI Structure**
The `streamlit_ui()` function creates the navigation sidebar and routes to different RAG implementations:

```python
def streamlit_ui():
    with st.sidebar:
        choice = option_menu('Navigation',["Home",'Simple RAG','RAG with Neo4J','AdaptiveRAG','RAG_Ranking'])
    
    # Route to appropriate RAG implementation based on choice
```

### **RAG Implementation Details**

#### **1. Simple RAG (`RAG()` function)**

This is the **most basic RAG implementation**:

```python
def RAG(docs):
    # 1. Document Processing
    for source_docs in docs:
        with tempfile.NamedTemporaryFile(delete=False,dir=TMP_DIR.as_posix(),suffix='.pdf') as temp_file:
            temp_file.write(source_docs.read())
    
    # 2. Load documents
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf', show_progress=True)
    documents = loader.load()

    # 3. Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(documents)

    # 4. Vector embeddings and storage
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(text, embedding)
    db.save_local(DB_FAISS_PATH)

    # 5. Create conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        db.as_retriever(search_kwargs={'k':20}),
        return_source_documents=True
    )
```

**How it works:**
- Uploads PDF documents
- Splits them into chunks (1000 characters each)
- Creates embeddings using HuggingFace's sentence transformer
- Stores vectors in FAISS database
- Uses ConversationalRetrievalChain for Q&A

#### **2. RAG with Neo4j (`RAG_Neo4j()` function)**

This implementation uses **Neo4j knowledge graph** for enhanced retrieval:

```python
def RAG_Neo4j():
    rag_graph = RAG_Graph()
    
    # Check if Neo4j is available
    if not rag_graph.neo4j_available:
        st.error("⚠️ Neo4j database is not available...")
        return
    
    # Two options: upload documents or just view graph
    choice = option_menu('Options',["Upload document",'Graph(Skip document upload)'])
```

**Key features:**
- Creates knowledge graphs from documents using LLM
- Stores entities and relationships in Neo4j
- Uses hybrid search (vector + graph) for retrieval
- Visualizes the knowledge graph

#### **3. Adaptive RAG (`Adaptive_RAG()` function)**

This is the **most advanced implementation** using LangGraph for self-corrective RAG:

```python
def Adaptive_RAG():
    source_docs = st.file_uploader(label="Upload document", type=['docx'],accept_multiple_files=True)
    
    if prompt2 := st.chat_input("Ask question to document assistant"):
        # Retrieve documents
        get_vectore_retriever(source_docs,"web")
        # Generate answer using adaptive graph
        generated_answer = build_graph(prompt2)
```

**Advanced features:**
- **Query analysis** - Determines best data source (vectorstore vs web search)
- **Document grading** - Evaluates relevance of retrieved documents
- **Self-correction** - Rewrites queries if initial results are poor
- **Hallucination detection** - Checks if generated answers are factual

#### **4. RAG with Ranking (`RAG_with_ranking()` function)**

This implementation adds **document reranking** using Cohere:

```python
def RAG_with_ranking():
    # Standard document processing...
    
    # Create retriever
    retriever = db.as_retriever(search_kwargs={"k":20})
    
    # Add reranking
    compressor = CohereRerank()
    compressor_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    
    # Create QA chain with reranked results
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compressor_retriever
    )
```

**Benefits:**
- Reranks retrieved documents for better relevance
- Improves answer quality by prioritizing most relevant chunks

### **Key Workflows**

#### **Document Processing Pipeline:**
1. **Upload** → User uploads documents (PDF/DOCX)
2. **Load** → Documents are loaded using appropriate loaders
3. **Split** → Text is split into manageable chunks
4. **Embed** → Chunks are converted to vector embeddings
5. **Store** → Vectors are stored in database (FAISS/Neo4j/Chroma)
6. **Retrieve** → Relevant chunks are retrieved for queries
7. **Generate** → LLM generates answers based on retrieved context

#### **Chat Interface:**
- Uses Streamlit's chat interface for conversation
- Maintains chat history in session state
- Displays both user questions and AI responses
- Shows source documents for transparency

### **Technical Highlights**

1. **Multiple Vector Stores**: FAISS, Neo4j, Chroma
2. **Hybrid Search**: Combines vector similarity with graph relationships
3. **Self-Corrective**: Adaptive RAG can rewrite queries and re-retrieve
4. **Document Reranking**: Uses Cohere for better relevance scoring
5. **Knowledge Graphs**: Extracts entities and relationships
6. **Error Handling**: Graceful fallbacks when services are unavailable

### **Usage Flow**

1. **Start the app**: `streamlit run Streamlit_RAG.py`
2. **Choose RAG type**: Select from sidebar navigation
3. **Upload documents**: Upload PDF/DOCX files
4. **Ask questions**: Use the chat interface to query documents
5. **View results**: Get AI-generated answers with source citations

This application demonstrates **state-of-the-art RAG techniques** and serves as an excellent learning resource for understanding different approaches to document-based question answering!