# streamlit_app.py

import streamlit as st

from src.supervisor_agent import SupervisorAgent
from src.utils.logger import setup_logger

# Remove deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(page_title="Multi-Agent Researcher", layout="wide")

logger = setup_logger()

st.title("🧠 Multi-Agent Research System")
st.write("Enter a topic and let the agents do the research for you.")

topic = st.text_input("Research Topic:", placeholder="ex: AI agents, federated learning, etc.")

run_button = st.button("Run Research")

if run_button and topic:
    supervisor = SupervisorAgent()

    with st.spinner("⏳ Agents are working..."):
        try:
            analysis = supervisor.run_research(topic)
            report = supervisor.prepare_report()["markdown_report"]
            st.success("Research complete.")
            st.markdown(report)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            logger.exception(e)

st.divider()

with st.expander("🛠 Developer Logs"):
    with open("logs/app.log", "r") as f:
        st.code(f.read())
