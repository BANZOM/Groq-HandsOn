from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import Template
import time
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GroqService:
    def __init__(self, model_name="Llama3-8b-8192", path_to_pdfs=None):
        logging.info("Initializing GroqService")
        self.model_name = model_name

        if path_to_pdfs not in [None, ""]:
            logging.info(f"Path to PDFs: {path_to_pdfs}")
            self.path_to_pdfs = path_to_pdfs
        else:
            logging.error("PDFs path not provided")
            raise ValueError("Please provide path to PDFs")
        
        self._llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=self.model_name)
    
    def embed_and_load(self):
        logging.info("Embedding and loading PDFs")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
        loader = PyPDFDirectoryLoader(self.path_to_pdfs)
        docs = loader.load()
        logging.info("Documents loaded")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splitted_docs = text_splitter.split_documents(docs)
        logging.info("Documents splitted")
        vectors = FAISS.from_documents(splitted_docs, embeddings)
        return vectors

    def template(self):
        logging.info("Creating template")
        return Template.template()
    
    def run(self, query):
        logging.info("creating chain and invoking it")
        document_chain = create_stuff_documents_chain(llm=self._llm, prompt=self.template())
        retriver = self.embed_and_load().as_retriever()
        retrieval_chain = create_retrieval_chain(retriever=retriver, document_chain=document_chain)
        start_time = time.process_time()
        logging.info("Invoking chain")
        response = retrieval_chain.invoke({"input": query})
        end_time = time.process_time()
        return response, end_time - start_time


if __name__ == '__main__':
    obj = GroqService(path_to_pdfs="../../Documents")
    response, time_taken = obj.run("What is the capital of India?")
    print(response["answer"])
    print(time_taken)