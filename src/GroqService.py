from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import Template
import Embed
import time
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GroqService:
    def __init__(self, model_name="mixtral-8x7b-32768", path_to_pdfs=None):
        logging.info("Initializing GroqService")
        self.model_name = model_name

        if path_to_pdfs not in [None, ""]:
            logging.info(f"Path to PDFs: {path_to_pdfs}")
            self.path_to_pdfs = path_to_pdfs
        else:
            logging.error("PDFs path not provided")
            raise ValueError("Please provide path to PDFs")
        
        self._llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=self.model_name)
        self._is_local = False
        self._embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    def embed_and_load(self):
        if self._is_local:
            logging.info("Loading vectors from local")
            return FAISS.load_local("./faiss_db", self._embeddings, allow_dangerous_deserialization=True)
        
        Embed.load_pdf_and_embed(self.path_to_pdfs)
        self._is_local = True
        return self.embed_and_load()

    def template(self):
        logging.info("Creating template")
        return Template.template()
    
    def run(self, query):
        logging.info("creating chain and invoking it")
        retriver = self.embed_and_load().as_retriever()
        logging.info("Retriever created")
        document_chain = create_stuff_documents_chain(self._llm, self.template())
        logging.info("Document chain created")
        retrieval_chain = create_retrieval_chain(retriver,document_chain)
        logging.info("Retrieval chain created")
        start_time = time.process_time()
        logging.info("Invoking chain")
        response = retrieval_chain.invoke({"input": query})
        end_time = time.process_time()
        return response, end_time - start_time


if __name__ == '__main__':
    obj = GroqService(path_to_pdfs=os.getenv("BASE_PATH") + "/src/Documents")
    while True:
        query = input("Enter your query: ")
        response, time_taken = obj.run(query)
        print(response["answer"])
        print(time_taken)