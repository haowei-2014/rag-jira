import os, getpass
import streamlit as st
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts.chat import ChatPromptTemplate
import pandas as pd
import bs4
import chromadb
from typing import Dict, Union
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import argparse

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    score: float

class JiraAgent:
    def __init__(self, input_file: str):

        # Load LLM
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="Nexusflow/Athene-V2-Chat",
            temperature=0,
            max_new_tokens=512
        )
        self.llm = ChatHuggingFace(llm=llm_endpoint)

        # Load Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"})

        # Load CSV
        df = pd.read_csv(input_file)

        # Convert rows into Document objects
        documents = []
        for _, row in df.iterrows():
            content = f"Issue: {row['Issue']}\n\nDescription: {row['Description']}\n\nCategory: {row['Category']}"
            documents.append(
                Document(page_content=content, 
                metadata={"Ticket ID": row["Ticket ID"], 
                            "Issue": row["Issue"],
                            "Category": row["Category"],
                            "Resolution": row['Resolution'], 
                            "Date": row["Date"],
                            "Agent Name": row['Agent Name'], 
                            "Resolved": row['Resolved'],
                            "Description": row["Description"]}))

        self.vector_store = Chroma.from_documents(documents=documents, embedding=self.embeddings)
        self.vector_store.add_documents(documents)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant for answering jira questions. You are given a retrieved context which includes an existing similar question, its description and its resolution. Use the retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:")
            ]
        )

        # retrieval arguments
        self.k = 1
        self.score_threshold = 0.15

        # Compile graph
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()

    def retrieve(self, state: State) -> Dict[str, Union[List[Document], float]]:
        """
        Define retrieval steps
        """
        results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
            state["question"],
            k=self.k,
            score_threshold=self.score_threshold
        )

        if not results_with_scores:
            return {"context": [], "score": 0.0}

        retrieved_docs = [doc for doc, _ in results_with_scores]
        scores = [score for _, score in results_with_scores]
        return {"context": retrieved_docs, "score": scores[0]}

    def generate(self, state: State) -> Dict[str, str]:
        """
        Define generation steps
        """

        if not state["context"]:
            return {"answer": "No relevant docs were retrieved using the relevance score threshold 0.15"}

        page_content = state["context"][0].page_content
        resolution = state["context"][0].metadata["Resolution"]
        context = page_content + "\n\n" + resolution
        resolved = state["context"][0].metadata["Resolved"]

        if resolved:
            messages = self.prompt.invoke({"question": state["question"], "context": context})
            response = self.llm.invoke(messages)
            return {"answer": response.content}
        else:
            response = "The retrieved issue has not been resolved yet."
            return {"answer": response}

def main(input_file: str):

    jira_agent = JiraAgent(input_file)

    # Streamlit Dashboard
    st.title("A RAG system to answer your IT issue")

    issue = st.text_input("Enter your issue:")
    description = st.text_input("Enter your issue's description:")
    category = st.text_input("Enter your issue's category:")
    if st.button("Submit") and issue:
        state: State = {
            "question": issue,
            "context": [],
            "answer": "",
            "score": 0.0
        }
        
        with st.spinner("Processing your question..."):
            state = jira_agent.graph.invoke(state)
        
        if state["context"]:
            st.write("**Retrieved context:**")
            st.write(f"Score: {state['score']}")
            st.write(state["context"][0].metadata)
        st.write("**LLM suggestion:**")
        st.write(state["answer"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A RAG system to answer your jira questions"
    )
    parser.add_argument("--input_file", type=str, default="ticket_dump_1.csv")
    args = parser.parse_args()
    
    main(args.input_file)
