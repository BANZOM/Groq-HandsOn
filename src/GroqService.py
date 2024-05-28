from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

class GroqService:
    def __init__(self, model_name="Llama3-8b-8192", path_to_pdfs=None):
        self.model_name = model_name

        if path_to_pdfs not in [None, ""]:
            self.path_to_pdfs = path_to_pdfs
        else:
            raise Exception("Please provide path to PDFs")
        
        self._llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=self.model_name)

    def embed_and_load(self):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        loader = PyPDFDirectoryLoader(self.path_to_pdfs)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splitted_docs = text_splitter.split_documents(docs)
        vectors = FAISS.from_documents(splitted_docs, embeddings)
        return vectors

    def template(self):
        prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Questions:{input}

        """
        )
        return prompt

if __name__ == '__main__':
    pass