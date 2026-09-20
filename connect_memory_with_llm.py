import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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

#1 setup groq llm
GROQ_API_KEY = require_env_var("GROQ_API_KEY")
GROQ_MODEL_NAME="openai/gpt-oss-20b"  # change to any groq model which support that time.

llm=ChatGroq(
    model=GROQ_MODEL_NAME,
    temperature=0.5,
    api_key=GROQ_API_KEY
)

# connect llm with faiss and create chain like usko kha sa data lena h


DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH,embedding_model, allow_dangerous_deserialization=True)

retrival_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain=create_stuff_documents_chain(llm=llm, prompt=retrival_qa_chat_prompt)

rag_chain=create_retrieval_chain(db.as_retriever(search_kwargs={"k":3}), combine_docs_chain)

#invoke with single query
user_query=input("Enter your query: ")
response=rag_chain.invoke({"input":user_query})

print("RESULT:", response["answer"])
for doc in response["context"]:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")