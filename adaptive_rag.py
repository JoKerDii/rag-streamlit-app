from autogen import AssistantAgent, UserProxyAgent

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import langchain_core
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader,WebBaseLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core import vectorstores
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Literal
from pprint import pprint
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START

load_dotenv(find_dotenv())

#Create temporary folder location for document storage
TMP_DIR = Path(__file__).resolve().parent.parent.joinpath('data','tmp')

retriever = 0;

#os.environ["TRAVILY_API"] = str(os.getenv("TRAVILY_API"))
os.environ["TAVILY_API_KEY"] = str(os.getenv("TRAVILY_API"))

llm = ChatGroq(temperature = 0.5,groq_api_key=os.getenv("GROQ_API_KEY"),model_name="llama3-70b-8192")


################## Vectore store ###############################

def  get_vectore_retriever(docs,choice):
    global retriever
    
    if(choice=="document"):
        for source_docs in docs:
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(),suffix='.docx') as temp_file:
                    temp_file.write(source_docs.read())
            
        loader = DirectoryLoader(TMP_DIR.as_posix(),glob='**/*docx',show_progress=True)
        document = loader.load()

    elif choice == "web":
        urls = [
        "https://www.nseindia.com/",
        "https://www.moneycontrol.com/promo/mc_interstitial_dfp.php?size=1280x540",
        "https://economictimes.indiatimes.com/?from=mdr",
        "https://www.investing.com/",
        "https://finance.yahoo.com/",
        "https://www.zeebiz.com/",
        "https://www.screener.in/"
        ]

        # Load
        docs = [WebBaseLoader(url).load() for url in urls]
        document = [item for sublist in docs for item in sublist]


    # Split
    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(document)

    embedding = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2',
                                            model_kwargs={'device':'cpu'})

    # Add to vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embedding,
    )
    retriever = vectorstore.as_retriever()
    print(type(retriever))
    return retriever



# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

def query_analyzer(question):
    # LLM with function call
    structured_llm_router = llm.with_structured_output(RouteQuery)

    # Prompt
    system = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. Otherwise, use web-search."""
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    question_router = route_prompt | structured_llm_router
    return question_router.invoke({"question": question})
    #return question_router

################## Retrieval Grader ##################

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def retriever_grader(retriever,question):
    # LLM with function call
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    docs = retriever.get_relevant_documents(question)
    doc_txt = docs[1].page_content
    return (retrieval_grader.invoke({"question": question, "document": doc_txt}))


################## Generate ################
def generate_answer(question,docs):
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": docs, "question": question})
    return(generation)


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    retriever = state["retriever"]
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        ''' score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        ) '''
        score = retriever_grader(retriever,question)
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            #st.write("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            #st.write("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}



def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}



def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    web_search_tool = TavilySearchResults(k=3)
    docs = web_search_tool.invoke({"query": question})
    #docs = web_search().invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {"documents": web_results, "question": question}


### Edges ###


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    #source = query_analyzer().invoke({"question": question})
    source = query_analyzer(question)
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        #st.write("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        #st.write("---DECISION: GENERATE---")
        return "generate"



def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    ''' score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    ) '''
    score = hallucination_grader(documents,generation)
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
       # st.write("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader(question,generation)
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            #st.write("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            #st.write("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


### Answer Grader

# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

def answer_grader(question,generation):
    # LLM with function call
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Prompt
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader
    return (answer_grader.invoke({"question": question, "generation": generation}))

#################### Hallucination Grader #####################

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

def hallucination_grader(docs,generation):
    # LLM with function call
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader
    return hallucination_grader.invoke({"documents": docs, "generation": generation})
     

############# Question Re-writer ################
def question_rewriter(question):
    # Prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    question_rewriter.invoke({"question": question})


############ Web search ################
def web_search():
    web_search_tool = TavilySearchResults(k=3)
    return web_search_tool


############## Graph State ##################

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    retriever: langchain_core.vectorstores.base.VectorStoreRetriever


##################### Graph Flow ############################

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    #st.write("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    #documents = retriever.invoke(question)
    #retriever = get_vectore_retriever()
    documents = retriever.invoke(question)
    return {"retriever": retriever, "documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    #st.write("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    
    #generation = generate_answer().invoke({"context": documents, "question": question})
    generation = generate_answer(documents,question)
    return {"documents": documents, "question": question, "generation": generation}

def build_graph(prompt2):
    workflow = StateGraph(GraphState)
    #st.write("---ROUTE QUESTION---")
    # Define the nodes
    #workflow.add_node("route_question", route_question)
    workflow.add_node("web_search", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query

    # Build graph
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile
    app = workflow.compile()

    # Run
    inputs = {
        "question": f"{prompt2}"
    }
    results = []
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            st.text(f"Node '{key}':")
            node_info = f"Node '{key}':"
            
            results.append(node_info)
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")
        st.text("--->")

    # Final generation
    pprint(value["generation"])
    generated_result = value["generation"]
    st.markdown(generated_result)
    return results, generated_result