import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from streamlit_option_menu import option_menu
#Let's integrate langsmith
from dotenv import load_dotenv, find_dotenv
from langsmith import Client
#Import related to KnowledgeGraph
from py2neo import Graph
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

from neo4j_kg import RAG_Graph
from adaptive_rag import build_graph, get_vectore_retriever
from langchain_groq import ChatGroq
#Reranking documents 
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

#load langsmith API key
load_dotenv(find_dotenv())
os.environ["GROQ_API_KEY"]=str(os.getenv("GROQ_API_KEY"))
os.environ["COHERE_API_KEY"]=str(os.getenv("COHERE_API_KEY"))



#Initialize the Client

#Create temporary folder location for document storage
TMP_DIR = Path(__file__).resolve().parent.joinpath('data','tmp')
llm = ChatGroq(temperature = 0.5,groq_api_key=os.environ["GROQ_API_KEY"],model_name="llama3-70b-8192")


header = st.container()

def streamlit_ui():

    with st.sidebar:
        choice = option_menu('Navigation',["Home",'Simple RAG','RAG with Neo4J','AdaptiveRAG','RAG_Ranking'])

    if choice == 'Home':
        st.title("RAG Application with different implementations")

    elif choice == 'Simple RAG':
        with header:
            st.title('Simple RAG with vector')  
            st.write("""This is a simple RAG process where user will upload a document then the document
                     will go through RecursiveCharacterSplitter and embedd in FAISS DB""")
            
            source_docs = st.file_uploader(label ="Upload a document", type=['pdf'], accept_multiple_files=True)
            if not source_docs:
                st.warning('Please upload a document')
            else:
                RAG(source_docs)
    
    elif choice == 'RAG with Neo4J':
        with header:
            st.title('RAG with Neo4J')
            st.write("""This is RAG with Neo4J knowledge graph. After uploading your document, click on Load Graph.
                     Knowledge graph will display in chatbot. Responses of user queries will be fetched using hybrid search approach. (embedding retrieval + graph search)""")
            RAG_Neo4j()

    elif choice == 'AdaptiveRAG':
        with header:
            st.title('Adaptive RAG with Langgraph')
            st.write("""Adaptive RAG is a strategy for RAG that unites (1) query analysis with (2) active / self-corrective RAG.""")
            Adaptive_RAG()

    elif choice == 'RAG_Ranking':
        with header:
            st.title('RAG with Cohere Ranking')
            st.write("""Improved RAG retrieval process with cohere ranking""")
            RAG_with_ranking()
    

            
def RAG(docs):
    #load the document
    for source_docs in docs:
        with tempfile.NamedTemporaryFile(delete=False,dir=TMP_DIR.as_posix(),suffix='.pdf') as temp_file:
            temp_file.write(source_docs.read())

    
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf', show_progress=True)
    documents = loader.load()

    #Split the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(documents)

    #Vector and embeddings
    DB_FAISS_PATH = 'vectorestore/faiss'
    embedding = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2',
                                         model_kwargs={'device':'cpu'})
    db = FAISS.from_documents(text,embedding)
    db.save_local(DB_FAISS_PATH)

    #Setup LLM, Fetch base url from LM Studio
    #llm = ChatOpenAI(base_url="http://localhost:1234/v1",api_key='lm-studio')
    llm = ChatGroq(temperature = 0.5,groq_api_key=os.environ["GROQ_API_KEY"],model_name="llama3-70b-8192")

    #Build a conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        db.as_retriever(search_kwargs={'k':20}),
        return_source_documents=False
    )
    chat_history = []
    #Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages =[]
    
    
    #React to user input
    if prompt := st.chat_input("Ask question to document assistant"):
        #Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        #Add user message to chat history
        st.session_state.messages.append({"role":"user","context":prompt})

        response = f"Echo: {prompt}"
        #Display assistant response in chat message container
        response = qa_chain({'question':prompt,'chat_history':chat_history})

        with st.chat_message("assistant"):
            st.markdown(response['answer'])
            # st.markdown(response['source_documents'])

        st.session_state.messages.append({'role':"assistant", "content":response})
        chat_history.append({prompt,response['answer']})

def RAG_Neo4j():
    rag_graph = RAG_Graph()
    
    if not rag_graph.neo4j_available:
        st.error("⚠️ Neo4j database is not available or APOC plugin is missing.")
        st.info("To use Neo4j features, you need to:")
        st.markdown("""
        1. **Install Neo4j Desktop or Neo4j Community Edition**
        2. **Install APOC plugin** (required for LangChain integration):
           - Download APOC plugin from: https://github.com/neo4j/apoc/releases
           - Place the APOC jar file in Neo4j's plugins directory
           - Add `apoc.*` to the allowed procedures in neo4j.conf
        3. **Start the Neo4j database**
        4. **Create a database** with username 'neo4j' and your password
        5. **Or update the connection settings** in KnowledgeGrpah_Neo4j.py
        """)
        st.markdown("**Alternative**: You can use other RAG features (Simple RAG, Adaptive RAG, RAG with Ranking) that don't require Neo4j.")
        return
    
    choice = option_menu('Options',["Upload document",'Graph(Skip document upload)'])
    flag = 0

    if choice == 'Upload document':
        flag = 1
        source_docs = st.file_uploader(label="Upload document", type=['docx'],accept_multiple_files=True)
        if not source_docs:
            st.warning("Please upload a document")
        else:
            try:
                rag_graph.create_graph(source_docs,TMP_DIR)
                st.success("Document processed successfully!")
            except Exception as e:
                st.error(f"Error processing document: {e}")
                st.info("Please make sure the document is valid and Neo4j is properly configured.")
    else:
        show_graph()

    
    st.session_state.messages1 = []
    #Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages1 =[]
    
    #Display chat messages from history on app rerun
    for message in st.session_state.messages1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    #React to user input
    if flag == 0:
        if prompt1 := st.chat_input("Ask question to document assistant"):
            #Display user message in chat message container
            st.chat_message("user").markdown(prompt1)
            #Add user message to chat history
            st.session_state.messages1.append({"role":"user","context":prompt1})

            try:
                #Display assistant response in chat message container
                response1 = rag_graph.ask_question_chain(prompt1)
                #response1 = rag_graph.retriever1(prompt1)

                with st.chat_message("assistant"):
                    st.markdown(response1)

                st.session_state.messages1.append({'role':"assistant", "content":response1})
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"Error processing your question: {e}")
                    st.info("Please make sure you have uploaded documents and Neo4j is properly configured.")

def Adaptive_RAG():

    source_docs = st.file_uploader(label="Upload document", type=['docx'],accept_multiple_files=True)

    chat_history2 = []
    st.session_state.messages2 = []
    #Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages2 =[]
    
    #Display chat messages from history on app rerun
    for message in st.session_state.messages2:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    #React to user input
    if prompt2 := st.chat_input("Ask question to document assistant"):
        #Display user message in chat message container
        st.chat_message("user").markdown(prompt2)
        #Add user message to chat history
        st.session_state.messages2.append({"role":"user","context":prompt2})

        response2 = f"Echo: {prompt2}"
        #Display assistant response in chat message container
        #response = qa_chain({'question':prompt,'chat_history':chat_history})
        # Retrieve 
        get_vectore_retriever(source_docs,"web")
        # Generate answer
        generated_answer =build_graph(prompt2)

        st.session_state.messages2.append({'role':"assistant", "content":response2})
        st.session_state.messages2.append({'role':"assistant", "content":generated_answer})

        #chat_history2.append({prompt2,generated_answer1})

def RAG_with_ranking():
    #upload document
    docs = st.file_uploader(label= "Upload document", type=['pdf'],accept_multiple_files=True)
   
    
    if not docs:
        st.warning("Please upload a document")
    
    for source_docs in docs:
        with tempfile.NamedTemporaryFile(delete=False,dir=TMP_DIR.as_posix(),suffix='.pdf') as temp_file:
            temp_file.write(source_docs.read())
 
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf', show_progress=True)
    documents = loader.load()

    #Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(documents)

    #Vector and embeddings
    DB_FAISS_PATH = 'vectorestore_raking/faiss'
    embedding = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2',
                                         model_kwargs={'device':'cpu'})
    db = FAISS.from_documents(text,embedding)
    db.save_local(DB_FAISS_PATH)
    retriever = db.as_retriever(search_kwargs={"k":20})

    # Fix: Specify the correct Cohere rerank model
    try:
        compressor = CohereRerank(model="rerank-english-v3.0")
    except Exception as e:
        st.warning(f"Cohere rerank model not available: {e}")
        st.info("Falling back to basic retriever without reranking")
        # Fallback to basic retriever without reranking
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        if prompt := st.chat_input("Ask a question"):
            response = qa_chain({"query": prompt})
            st.write(response)
        return
    compressor_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = compressor_retriever
    )

    if prompt := st.chat_input("Ask a question"):
        response = qa_chain({"query": prompt})
        st.write(response)
        try:
            compressor_docs = compressor_retriever.get_relevant_documents(prompt)
            st.write("Retrieved documents:")
            st.write(compressor_docs)
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")


def show_graph():
    st.title("Neo4j Graph Visualization")

    #user input for Neo4J credential
    uri = st.text_input("Neo4j URI", "bolt://localhost:7687")
    user = st.text_input("Neo4j username", "neo4j")
    password = st.text_input("Neo4j password", type="password")

    #Create a load graph button
    if st.button("Load Graph"):
        try:
            data = get_graph_data(uri,user,password)
            if not data:
                st.warning("No data found in Neo4j database. Please make sure the database contains nodes and relationships.")
                return
            G = create_networkx_graph(data)
            visualize_graph(G)

            HtmlFile = open("graph.html", "r", encoding="utf-8")
            source_code = HtmlFile.read()
            components.html(source_code,height=600, scrolling=True)
        except Exception as e:
            st.error(f"Error connecting to Neo4j: {e}")
            st.info("Please make sure Neo4j is running, APOC plugin is installed, and the credentials are correct.")

def get_graph_data(uri,user,password):
    try:
        graph = Graph(uri,auth=(user,password))
        query = """
        MATCH (n)-[r]->(m)
        RETURN n,r,m
        LIMIT 100
        """

        data = graph.run(query).data()
        return data
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return None

def create_networkx_graph(data):
    if not data:
        return nx.DiGraph()
    
    G = nx.DiGraph()
    for record in data:
        try:
            n = record['n']
            m = record['m']
            r = record['r']
            
            # Handle cases where nodes might not have 'id' or 'name' properties
            n_id = n.get('id', str(n))
            m_id = m.get('id', str(m))
            n_name = n.get('name', str(n))
            m_name = m.get('name', str(m))
            
            G.add_node(n_id, label=n_name)
            G.add_node(m_id, label=m_name)
            G.add_edge(n_id, m_id, label=r.get('type', 'RELATES_TO'))
        except Exception as e:
            print(f"Error processing record: {e}")
            continue
    return G

def visualize_graph(G):
    try:
        net = Network(notebook=True)
        net.from_nx(G)
        net.show("graph.html")
    except Exception as e:
        st.error(f"Error creating graph visualization: {e}")
        # Create a simple HTML file with an error message
        with open("graph.html", "w") as f:
            f.write("<html><body><h2>Graph Visualization Error</h2><p>Unable to create graph visualization.</p></body></html>")


streamlit_ui()




