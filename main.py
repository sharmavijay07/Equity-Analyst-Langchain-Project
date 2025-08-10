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
import langchain
from dotenv import load_dotenv
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.2
)

file_path = "real_data.pkl"

st.title("News Research Tool")

st.sidebar.title("News Article URLs")
main_placeholder = st.empty()

# Input URLs
Urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    Urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked:
    # Loading data
    loader = UnstructuredURLLoader(urls=Urls)
    main_placeholder.text("Data is getting loaded...")
    data = loader.load()

    # Splitting data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=200
    )
    main_placeholder.text("Data is getting split...")
    docs = text_splitter.split_documents(data)

    # Creating embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    main_placeholder.text("Creating embeddings of data and saving it...")
    vectorstoreindex = FAISS.from_documents(docs, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstoreindex, f)

# Querying
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            result = chain.invoke({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources","")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split('\n')
                for source in sources_list:
                    st.write(source)
