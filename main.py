# import os
# import langchain
# from dotenv import load_dotenv
# import pickle
# import time
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# import streamlit as st

# load_dotenv()

# api_key = os.getenv("GROQ_API_KEY")

# llm = ChatGroq(groq_api_key=api_key,model_name="llama3-8b-8192")

# file_path = "real_data.pkl"

# st.title("News Research Tool")

# st.sidebar.title("News Article URLs")

# main_placeholder = st.empty()

# Urls=[]
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     Urls.append(url)

# process_url_clicked = st.sidebar.button("Process URLs")

# if process_url_clicked:

#     #loading data
#     loader = UnstructuredURLLoader(urls=Urls)
#     main_placeholder.text("Data is getting load...")
#     data = loader.load()

#     #splitting data
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators = ["\n\n","\n","." , ","],
#         chunk_size=200
#     )
#     main_placeholder.text("Data is getting splitted...")
#     docs = text_splitter.split_documents(data)

#     #creating embeddings
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     main_placeholder.text("Creating Embeddings of data and saving it...")
#     vectorstoreindex = FAISS.from_documents(docs,embeddings)

#     with open(file_path,"wb") as f:
#         pickle.dump(vectorstoreindex,f)

# query = main_placeholder.text_input("Question: ")
# if query:
#     if os.path.exists(file_path):
#         with open(file_path,"rb") as f:
#             vectorstore = pickle.load(f)
#             chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())
#             result = chain.invoke({"question":query},return_only_outputs=True)

#             #{"answer":"","sources":[]}
#             st.header("Answer")
#             st.subheader(result["answer"])

import os
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# ✅ New imports replacing RetrievalQAWithSourcesChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------
# Load environment variables
# ---------------------------------------------
load_dotenv()

st.title("📰 News Research & Analysis Tool")
st.sidebar.title("📎 Add News Article URLs")

# ---------------------------------------------
# Provider & model selection
# ---------------------------------------------
provider = st.sidebar.radio("Model provider", ["Groq (OSS)", "Google Gemini"], index=0)

llm = None
selected_model = None

if provider == "Groq (OSS)":
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY not found in environment variables.")
        st.stop()

    groq_models = [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "gemma-7b-it",
        "openai/gpt-oss-120b",
        "Custom (enter below)",
    ]
    selected_model = st.sidebar.selectbox("🧠 Groq model", groq_models, index=1)
    if selected_model == "Custom (enter below)":
        selected_model = st.sidebar.text_input("Custom Groq model id", value="llama3-70b-8192").strip()
    llm = ChatGroq(model=selected_model, groq_api_key=groq_api_key, temperature=0.2)
else:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("⚠️ GEMINI_API_KEY not found in environment variables.")
        st.stop()

    gemini_models = [
        "gemini-1.0-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ]
    selected_model = st.sidebar.selectbox("🧠 Gemini model", gemini_models, index=0)
    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        google_api_key=gemini_api_key,
        temperature=0.2,
    )

file_path = "real_data.pkl"

main_placeholder = st.empty()

# ---------------------------------------------
# Input URLs
# ---------------------------------------------
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")

# ---------------------------------------------
# Processing URLs
# ---------------------------------------------
if process_url_clicked:
    if not urls:
        st.warning("Please enter at least one valid URL.")
    else:
        main_placeholder.text("🔄 Loading data from URLs...")
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=500,
            chunk_overlap=50
        )
        main_placeholder.text("✂️ Splitting text into chunks...")
        docs = text_splitter.split_documents(data)

        main_placeholder.text("🧠 Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        main_placeholder.text("📦 Building FAISS vector index...")
        vectorstore_index = FAISS.from_documents(docs, embeddings)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_index, f)

        st.success("✅ Data processed and saved successfully!")

# ---------------------------------------------
# Query Section
# ---------------------------------------------
query = st.text_input("💬 Ask a question based on the news content:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # ---------------------------------------------
        # New LangChain 0.3.x compatible chain
        # ---------------------------------------------
        prompt = ChatPromptTemplate.from_template(
            """You are a financial analyst AI. Use ONLY the following context to answer the question accurately and clearly.

Context:
{context}

Question: {input}

Requirements:
- If the context is insufficient, say so and avoid guessing.
- After the answer, include a 'Sources' section listing the unique source URLs you used (see the 'Source:' lines in the context)."""
        )

        # Include source metadata inline for each chunk so the LLM can cite them.
        doc_prompt = PromptTemplate.from_template("Source: {source}\n\n{page_content}")

        document_chain = create_stuff_documents_chain(
            llm,
            prompt,
            document_prompt=doc_prompt,
        )
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        st.info("⏳ Generating answer...")
        try:
            result = retrieval_chain.invoke({"input": query})
        except Exception as e:
            st.error(
                "Model call failed. Try switching to another model from the sidebar.\n"
                "For Gemini, avoid '-latest' aliases as they may not be supported by the current SDK.\n\n"
                f"Provider: {provider}\nSelected model: {selected_model}\nDetails: {str(e)}"
            )
            st.stop()

        st.subheader("🧾 Answer:")
        st.write(result.get("answer", "No answer found."))

        # Display sources (deduplicated) from retrieved context
        sources = []
        for doc in result.get("context", []):
            src = doc.metadata.get("source", "Unknown source")
            if src and src not in sources:
                sources.append(src)
        if sources:
            st.subheader("🔗 Sources:")
            for i, src in enumerate(sources, 1):
                st.write(f"{i}. {src}")

        # Show retrieval scores and snippets for transparency
        try:
            top_docs = vectorstore.similarity_search_with_score(query, k=3)
            with st.expander("ℹ️ Retrieval details (scores & snippets)"):
                for i, (doc, score) in enumerate(top_docs, 1):
                    st.write(f"{i}. {doc.metadata.get('source', 'Unknown source')} (score: {score:.4f})")
                    st.caption((doc.page_content[:300] + "...") if len(doc.page_content) > 300 else doc.page_content)
        except Exception:
            # Some vectorstores may not support similarity_search_with_score; ignore if not available
            pass
    else:
        st.warning("Please process URLs first before asking questions.")
