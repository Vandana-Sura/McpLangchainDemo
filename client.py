from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

async def main():
    # Check if GROQ_API_KEY is set
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY not found in .env or environment.")
        print("👉 You can get one from https://console.groq.com/")
        return

    # Connect to your math and weather MCP tools
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathserver.py"],  # Make sure this path is correct
                "transport": "stdio"
            },
            "weather": {
                "url": "http://localhost:8000/mcp",  # Ensure this server is running
                "transport": "streamable_http"
            }
        }
    )

    tools = await client.get_tools()
    print(f"✅ Successfully loaded {len(tools)} tools.")

    model = ChatGroq(model="llama3-70b-8192")

    agent = create_react_agent(model, tools)

    print("\n🤖 Testing math calculation...")
    math_response = agent.invoke(
        {"messages": [{"role": "user", "content": "What is 8 times 12?"}]}
    )
    print("📐 Math Response:", math_response['messages'][-1].content)

    print("\n🌦️  Testing weather query...")
    weather_response = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the weather in California?"}]}
    )
    print("☁️  Weather Response:", weather_response['messages'][-1].content)

# Run the main function
asyncio.run(main())
