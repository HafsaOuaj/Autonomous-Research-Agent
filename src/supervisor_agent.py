from src.retriever_agent import ResearchRetrieverAgent
from src.analyzer_agent import DocumentAnalyzerAgent
import json
import logging
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.llms import Ollama
class SupervisorAgent:
    def __init__(self, model_name: str = "phi"):
        self.model_name = model_name
        self.llm = Ollama(model=model_name, base_url="http://host.docker.internal:11434")
        self.analysis = None

    def run_research(self, topic: str):
        print("\n🟦 STEP 1 — Retrieving information\n")
        retriever = ResearchRetrieverAgent(topic)
        docs = retriever.retrieve_documents()
        retriever.process_and_store(docs)

        print("\n🟩 STEP 2 — Analyzing retrieved documents\n")
        analyzer = DocumentAnalyzerAgent(model_name=self.model_name)
        self.analysis = analyzer.analyze_documents(analyzer.fetch_relevant_docs(topic))

        print("\n🟧 STEP 3 — Final Report\n")
        return self.analysis

    def prepare_report(self):
        if not self.analysis:
            raise ValueError("No research analysis found. Run `run_research()` first.")

        prompt = f"""
                    You are an AI assistant.
                    Analyze the following research information and prepare a technical report including:
                    - Introduction
                    - Key findings
                    - Methods used
                    - Limitations
                    - Conclusion
                    You may add any relevant context.
                    Return the report in markdown format.

                    Research Information:
                    {self.analysis}
                    """                                             
        response = self.llm.invoke(prompt)
        return {"markdown_report": response}

    
if __name__ == "__main__":
    topic = input("Enter your research topic: ")
    supervisor = SupervisorAgent(model_name="phi")
    analysis = supervisor.run_research(topic)
    report = supervisor.prepare_report()

    print(json.dumps(report, indent=2, ensure_ascii=False))

    