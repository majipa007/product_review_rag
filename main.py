import torch
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import os
import threading
import sys
import time
import shutil
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


class RAG:
    def __init__(self):
        self.loading = None
        self.path = "output.txt"
        self.emodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        self.vectordb = None
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "Majipa/cars_base",
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,

        )
        self.generation_args = {
            "max_new_tokens": 1000,
            "return_full_text": False,
            "temperature": 0.5,
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
        with open(self.path, 'r', encoding='utf-8') as file:
            docs = file.read()
        split_data = docs.split("--------------------------------------------------")
        split_data = [review.strip() for review in split_data if review.strip()]
        documents = [Document(page_content=review) for review in split_data]
        return documents

    def vector_database(self):
        persist_directory = os.path.join(os.getcwd(), 'docs', 'chroma')
        os.makedirs(persist_directory, exist_ok=True)
        shutil.rmtree(persist_directory, ignore_errors=True)
        embedding = SentenceTransformerEmbeddings(self.emodel)
        self.vectordb = Chroma.from_documents(
            documents=self.document_loader(),
            embedding=embedding,
            persist_directory=persist_directory
        )

    def retrieving(self, question):
        docs_ss = self.vectordb.similarity_search(question, k=5)
        return docs_ss

    def semantic_search(self, query: str, top_k: int = 10):
        # Perform the search
        docs_mmr = self.vectordb.max_marginal_relevance_search(query, k=top_k)

        # Format the results
        formatted_results = []
        for doc in docs_mmr:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
            })

        return formatted_results

    def run(self):
        message = [
            {"role": "system",
             "content": "You are a helpful Car Improvement analyst that works on the basis of provided "
                        "Reviews and gives described information."},
        ]
        text = None
        os.system('clear')
        while text != 'bye':
            print("\rYOU:")
            text = input("")
            text_rag = self.semantic_search(text)
            message.append({"role": "user", "content": f"context: {text_rag}, question: {text}"})
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
