from retriever_agent import ResearchRetrieverAgent
from analyzer_agent import DocumentAnalyzerAgent
import json
import logging
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



class SupervisorAgeent:
    def __init__(self,model_name:str="phi"):
        self.model_name = model_name

    def run_research(self,topic:str):
        print("\n🟦 STEP 1 — Retrieving information\n")
        retriever = ResearchRetrieverAgent(topic)
        docs = retriever.retrieve_documents()
        retriever.process_and_store(docs)

        print("\n🟩 STEP 2 — Analyzing retrieved documents\n")
        analyzer = DocumentAnalyzerAgent(model_name=self.model_name)
        analysis = analyzer.analyze_documents(
            analyzer.fetch_relevant_docs(topic)
        )

        print("\n🟧 STEP 3 — Final Report\n")
        return analysis
    
if __name__ == "__main__":
    topic = input("Enter your research topic: ")
    supervisor = SupervisorAgeent(model_name="phi")
    report = supervisor.run_research(topic)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    