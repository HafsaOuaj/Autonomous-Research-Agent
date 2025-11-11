import os
import json 
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

CHROMA_PATH = "./vector_store"

class DocumentAnalyzerAgent:
    def __init__(self,model_name:str="mistral"):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        self.llm = Ollama(model=model_name)

    def fetch_relevent_docs(self,query:str,top_k:int=3):
        print(f"📚 Retrieving top {top_k} documents for: {query}")
        results = self.db.similarity_search(query, k=top_k)
        return results
    
    def analyze_documents(self,docs):
        print("🧠 Analyzing retrieved documents...")
        combined_text = "\n\n".join([doc.page_content for doc in docs])

        prompt = f""""
        You are an Ai assistant.
        Analyze the following text and extract structured research information.

        Text:
        {combined_text}

        Return the answer as a JSON with these fields

                {{
            "main_topic": "",
            "key_points": [],
            "methods": [],
            "findings": [],
            "limitations": []
        }}

        """

        response = self.llm.invoke(prompt)

        try:
            data = json.load(response)
        except:
            data = {"raw summary":response}

        return data
    

if __name__ == "__main__":
    query = input("Enter the topic to analyze: ")
    agent = DocumentAnalyzerAgent()
    docs = agent.fetch_relevent_docs(query)
    summary = agent.analyze_documents(docs)
    print("\n📊 Structured Analysis:\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


        

        

