from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Ensure Groq API key is set
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# Define a web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=["Always include the source of the information you find."],
    show_tool_calls=True,
    markdown=True,
)

# Define a finance agent
finance_agent = Agent(
    name="Finance Agent",
    role="Provide financial information and analysis",
    model=Groq(id="llama3-70b-8192"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True
    )],
    instructions=["Use tables to display the data."],
    show_tool_calls=True,
    markdown=True,
)

# Create a multi-agent team
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id="llama3-70b-8192"),
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

if __name__ == "__main__":
    multi_ai_agent.print_response(
        "Summarize analyst recommendations and share the latest news for NVDA",
        stream=True
    )
