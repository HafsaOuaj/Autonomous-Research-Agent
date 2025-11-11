import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

CHROMA_PATH ="./vector_store"

class ResearchRetrieverAgent:
    def __init__(self,topic:str,n_results:int=5):
        self.topic = topic
        self.n_results =n_results 
        self.search_tool = DuckDuckGoSearchRun()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = None


    def retrieve_documents(self):
        print(f"🔍 Searching DuckDuckGo for: {self.topic}")
        results = self.search_tool.run(self.topic)
        # DuckDuckGo returns a string summary; wrap in Document
        return [Document(page_content=results, metadata={"source": "duckduckgo"})]
    
    def process_and_store(self,docs):
        print("📄 Splitting and embedding documents...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        all_splits = []
        for doc in tqdm(docs):
            splits = splitter.split_documents([doc])
            all_splits.extend(splits)

        self.db = Chroma.from_documents(all_splits,self.embeddings,persist_directory=CHROMA_PATH)
        self.db.persist()
        print("✅ Documents embedded and stored locally")

    def query_local_db(self,query:str,top_k:int=3):
        if not self.db:
            self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        results = self.db.similarity_search(query,k=top_k)
        return results
    

if __name__ == "__main__":
    topic = input("Enter your research topic: ")
    agent = ResearchRetrieverAgent(topic)
    docs = agent.retrieve_documents()
    agent.process_and_store(docs)
    print(docs)

    q = input("\nAsk something about the topic: ")
    answers = agent.query_local_db(q)
    for doc in answers:
        print("\n🔹", doc.page_content[:400], "...")
