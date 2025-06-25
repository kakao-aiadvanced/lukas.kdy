from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
import os

os.environ['OPENAI_API_KEY'] = '****'
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "****"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = rag_prompt | llm | StrOutputParser()

retrieval_grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """),
        ("human", "question: {question}\n\n document: {document} "),
    ]
)
retrieval_grade_chain = retrieval_grade_prompt | llm | JsonOutputParser()

answer_generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise"""),
        ("human", "question: {question}\n\n context: {context} "),
    ]
)
answer_generation_chain = answer_generation_prompt | llm | StrOutputParser()

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a grader assessing whether
    an answer is hallucinated. Give a binary 'yes' or 'no' score to indicate
    whether the answer is hallucinated. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation."""),
        ("human", "documents: {documents}\n\n answer: {generation} "),
    ]
)
hallucination_grader_chain = hallucination_prompt | llm | JsonOutputParser()

answer_grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a grader assessing whether an
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
        ("human", "question: {question}\n\n answer: {generation} "),
    ]
)
answer_grade_chain = answer_grade_prompt | llm | JsonOutputParser()