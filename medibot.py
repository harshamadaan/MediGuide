import os
from langchain_groq import ChatGroq
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()


def require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Missing {name}. Create a .env file in the project root with {name}=your_key_here "
            "or export it in your terminal before running the app."
        )
    return value


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource

def get_vectorstore():
    if not os.path.exists(DB_FAISS_PATH):
        from create_memory_for_llm import create_vector_db
        create_vector_db()
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db=FAISS.load_local(DB_FAISS_PATH,embedding_model, allow_dangerous_deserialization=True)
    return db


def main():
    st.title("Ask Medibot")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt=st.chat_input("Ask your question here:")

    if(prompt):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role":"user","content":prompt})
        HUGGING_FACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try:
            with st.spinner("Processing your request..."):
                vectorstore=get_vectorstore()
                if vectorstore is None:
                    st.error("Vectorstore is not loaded. Please check the logs for errors.")
                GROQ_MODEL_NAME="openai/gpt-oss-20b"
                GROQ_API_KEY = require_env_var("GROQ_API_KEY")
                llm=ChatGroq(
                    model=GROQ_MODEL_NAME,
                    temperature=0.5,
                    api_key=GROQ_API_KEY
                )
                
                retrival_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")

                combine_docs_chain=create_stuff_documents_chain(llm=llm, prompt=retrival_qa_chat_prompt)

                rag_chain=create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k":3}), combine_docs_chain)

                response=rag_chain.invoke({"input":prompt})

                print("RESULT:", response["answer"])
                for doc in response["context"]:
                    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")

            result=response["answer"]
            st.chat_message("assistant").markdown(result)

            st.session_state.messages.append({"role":"assistant","content":result})

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()