from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv

import phi
from phi.playground import Playground , serve_playground_app    
load_dotenv()

phi.api = os.getenv("PHI_API_KEY")

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

app = Playground(
    agents=[finance_agent,web_search_agent]
).get_app()

if __name__ == "__main__":
    # Serve the playground app
    serve_playground_app("playground:app", reload=True)
    
    
    