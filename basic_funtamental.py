import os 
import pickle
import time
import langchain
import langchain_community
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=api_key,model="llama3-8b-8192")


loaders = UnstructuredURLLoader(
    urls=["https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html",
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html",
    "https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html"
    ]
)

data = loaders.load()

text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=1000,
   chunk_overlap=200
)

docs = text_splitter.split_documents(data)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorIndex_huggingface = FAISS.from_documents(docs,embeddings)

file_path = "vector_index.pkl"
with open(file_path,"wb") as f:
    pickle.dump(vectorIndex_huggingface,f)

if os.path.exists(file_path):
    with open(file_path,"rb") as f:
        vectorIndex = pickle.load(f)

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())

query = "What is the price of Tiago iCNG"

langchain.debug = True
chain({"question":query},return_only_outputs=True)
