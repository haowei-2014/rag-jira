import os, getpass
import streamlit as st
from langchain_openai import ChatOpenAI
import getpass
import os
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.vectorstores import Chroma
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict
from langchain_core.prompts.chat import ChatPromptTemplate


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


llm = ChatOpenAI(model="gpt-4o")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

#vector_store = InMemoryVectorStore(embeddings)

import pandas as pd
from langchain_core.documents.base import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings  # or any embedding class

# Load CSV
df = pd.read_csv("/Users/haowei/Documents/interview/AlephAlpha/case-study/aa-case-study-ai-solutions-engineer-main/data/old_tickets/ticket_dump_1.csv")

# Convert rows into Document objects
documents = []
for idx, row in df.iterrows():
    content = f"Issue: {row['Issue']}\n\nDescription: {row['Description']}\n\nCategory: {row['Category']}"
    documents.append(
        Document(page_content=content, 
        metadata={"Ticket ID": row["Ticket ID"], "Resolution": row['Resolution'], "Agent Name": row['Agent Name'], "Resolved": row['Resolved']}))

vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)

# Initialize embedding + vector store
# embeddings = OpenAIEmbeddings()
# vector_store = Chroma.from_documents(documents, embeddings)

# Or, if using an existing vector store:
# vector_store.add_documents(documents)

# Index chunks
# _ = vector_store.add_documents(documents=all_splits)
vector_store.add_documents(documents)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
# prompt = hub.pull("rlm/rag-prompt")
# print("type of prompt:")
# print(type(prompt))
# print(prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an assistant for answering jira questions. You are given a retrieved context which includes an existing similar question, its description and its resolution. Use the retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:")
    ]
)
print(prompt)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    print("----- retrieve")
    # retrieved_docs, scores = vector_store.similarity_search_with_relevance_scores(
    #     state["question"], k=1, score_threshold=0.1)

    k = 1
    score_threshold = 0.0

    results_with_scores = vector_store.similarity_search_with_relevance_scores(
        state["question"],
        k=k,
        score_threshold=score_threshold
    )

    print(f"Results for query: '{state['question']}' (k={k}, threshold={score_threshold})")
    if not results_with_scores:
        print("→ No documents met the threshold.")
    else:
        for idx, (doc, score) in enumerate(results_with_scores, start=1):
            content = doc.page_content
            print(f"{idx}. Score: {score:.2f} — Content: {content}")

    print("----- retrieve2")

    retrieved_docs = [doc for doc, _ in results_with_scores]
    scores = [score for _, score in results_with_scores]

    print(type(retrieved_docs))
    print(type(retrieved_docs[0]))
    print(retrieved_docs[0].page_content)
    print(retrieved_docs[0].metadata)
    print(scores[0])

    return {"context": retrieved_docs}


def generate(state: State):
    print("----- generate")
    page_content = state["context"][0].page_content
    resolution = state["context"][0].metadata["Resolution"]
    context = page_content + "\n\n" + resolution
    print(type(context))
    print(prompt)
    messages = prompt.invoke({"question": state["question"], "context": context})
    print(type(messages))
    response = llm.invoke(messages)
    print(type(response))
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Build graph
# builder = StateGraph(State)
# builder.add_node("retrieve", retrieve)
# builder.add_node("generate", generate)
# builder.add_edge(START, "retrieve")
# builder.add_edge("retrieve", "generate")
# builder.add_edge("generate", END)
# graph = builder.compile()


# (Section 6: Streamlit Dashboard) ----------
st.title("A RAG system to answer your jira questions")

question = st.text_input("Enter your question:")
if st.button("Submit") and question:
    state: State = {
        "question": question,
        "context": [],
        "answer": "",
    }
    
    with st.spinner("Processing your question..."):
        state = graph.invoke(state)
    
    st.write("**Retrieved context:**")
    st.write(state["context"])
    st.write("**LLM suggestion:**")
    st.write(state["answer"])
