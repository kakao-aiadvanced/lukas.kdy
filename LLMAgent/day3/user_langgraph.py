from pprint import pprint
from typing import List

from langchain_core.documents import Document
from typing_extensions import TypedDict

from web_retriever import retriever
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, START, END

from llm_chain import rag_chain,retrieval_grade_chain, answer_generation_chain, hallucination_grader_chain, answer_grade_chain

from tavily import TavilyClient
tavily = TavilyClient(api_key='****')

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        web_search_count: number of web searches
        hallucination_check_count: number of hallucination checks
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    web_search_count: int = 0
    hallucination: str
    hallucination_check_count: int = 0
    documents: List[Document]

### Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    print(question)
    print(documents)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = answer_generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grade_chain.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    if web_search == "Yes":
        return {"documents": filtered_docs, "question": question, "web_search": web_search, "generation":"failed: not relevant"}
    else:
        return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    print(state)
    question = state["question"]
    documents = None
    if "documents" in state:
      documents = state["documents"]

    # Web search
    docs = tavily.search(query=question)['results']

    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results, metadata={"url": docs[0]["url"], "title": docs[0]["title"]})
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question, "web_search_count": 1}

def hallucination_check(state):
    """
    Check hallucination

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, hallucination, that contains hallucination result
    """

    print("---HALLUCINATION CHECK---")

    documents = state["documents"]
    generation = state["generation"]
    hallucination_check_count = state["hallucination_check_count"]

    score = hallucination_grader_chain.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "no":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return {"hallucination": "No", "generation": generation, "hallucination_check_count": hallucination_check_count + 1, "documents": documents}

    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS")
        return {"hallucination": "Yes", "generation": "failed: hallucination", "hallucination_check_count": hallucination_check_count + 1, "documents": documents}


### Edges

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "No":
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    elif web_search == "Yes" and state["web_search_count"] == 0:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        return "not relevant"

def decide_to_hallucination_check(state):
    """
    Determines whether to re-generate an answer, or finish

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---DECISION HALLUCINATION CHECK---")
    hallucination_check_count = state["hallucination_check_count"]
    hallucination = state["hallucination"]
    if hallucination_check_count > 1:
        return "over limit"
    elif hallucination == "Yes":
        return "generate"
    else:
        return "not hallucination"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("hallucination_check", hallucination_check)  # hallucination check

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"websearch": "websearch", "generate": "generate", "not relevant": END},
)
workflow.add_edge("websearch", "grade_documents")
workflow.add_edge("generate", "hallucination_check")
workflow.add_conditional_edges(
    "hallucination_check",
    decide_to_hallucination_check,
    {
        "generate": "generate",
        "over limit": END,
        "not hallucination": END,
    },
)
