import typer
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.embedder.sentence_transformer import SentenceTransformersEmbedder  # 👈 local embeddings
from phi.llm.groq import GroqLLM

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Local embedding model (no API calls, runs on CPU/GPU)
embedder = SentenceTransformersEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

knowledge_base = PDFKnowledgeBase(
    path="data/pdfs",  
    vector_db=PgVector2(collection="recipe", db_url=db_url, embedder=embedder),
)

knowledge_base.load()
storage = PgAssistantStorage(table_name="pdf_assistant", db_url=db_url)

def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
        llm=GroqLLM(
            model="llama3-70b-8192",
            api_key=os.environ["GROQ_API_KEY"],
        ),
    )

    if run_id is None:
        run_id = assistant.run_id
        print(f"New run started with ID: {run_id}")
    else:
        print(f"Continuing run with ID: {run_id}")

    assistant.cli_app(markdown=True)

if __name__ == "__main__":
    typer.run(pdf_assistant)
