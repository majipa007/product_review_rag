import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import embeddings
import threading
import sys
import time
import shutil
import re


def clean_text(text):
    # Replace non-breaking spaces with regular spaces
    text = text.replace('\xa0', ' ')
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class RAG:
    def __init__(self):
        self.loading = None
        self.path = "dream.pdf"
        self.emodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        self.vectordb = None
        self.model = AutoModelForCausalLM.from_pretrained(
            "my_model",
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("my_model")

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,

        )
        self.generation_args = {
            "max_new_tokens": 100,
            "return_full_text": False,
            "temperature": 0.7,
            "do_sample": True
        }
        self.vector_database()

    def spinning_loader(self):
        while self.loading:
            for char in "|/-|\\":
                sys.stdout.write(f"\r {char}")
                sys.stdout.flush()
                time.sleep(0.1)

    def document_loader(self):
        loader = PyPDFLoader(self.path)
        pages = loader.load()
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,  # Reduced from 1000
            chunk_overlap=50,  # Reduced from 150
            length_function=len
        )
        docs = text_splitter.split_documents(pages)
        return docs

    def vector_database(self):
        persist_directory = os.path.join(os.getcwd(), 'docs', 'chroma')
        os.makedirs(persist_directory, exist_ok=True)
        shutil.rmtree(persist_directory, ignore_errors=True)
        embedding = embeddings.SentenceTransformerEmbeddings(self.emodel)
        self.vectordb = Chroma.from_documents(
            documents=self.document_loader(),
            embedding=embedding,
            persist_directory=persist_directory
        )

    def retrieving(self, question):
        docs_ss = self.vectordb.similarity_search(question, k=5)
        return docs_ss

    def basic_integration(self, query):
        docs_ss = self.retrieving(query)
        context = "\n\n".join([doc.page_content for doc in docs_ss])
        context = clean_text(context)
        return f"Query: {query}\n\n Context: {context}"

    def run(self):
        message = [
            {"role": "system",
             "content": "You are a professional Psychologist and you will help people with their problem"},
        ]
        text = None
        os.system('clear')
        while text != 'bye':
            print("\rYOU:")
            text = input("")
            text_rag = self.basic_integration(text)
            message.append({"role": "user", "content": text_rag})
            print("BOT:")
            # Start the loading animation
            self.loading = True
            loader_thread = threading.Thread(target=self.spinning_loader)
            loader_thread.start()
            try:
                # Your long-running code here
                output = self.pipe(message, **self.generation_args)
                ans = output[0]['generated_text']
                print("\r-" + ans)
                message.pop()
                message.append({"role": "user", "content": text})
                message.append({"role": "assistant", "content": ans})
                torch.cuda.empty_cache()
            finally:
                # Stop the loading animation
                self.loading = False
                loader_thread.join()


x = RAG()
x.run()
