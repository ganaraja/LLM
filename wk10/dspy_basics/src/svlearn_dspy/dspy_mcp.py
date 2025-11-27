import asyncio
import dspy
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

async def main():
    # Connect to HTTP MCP server
    async with streamablehttp_client("http://localhost:9000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # List and convert tools
            response = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            # Create a ReAct agent with the tools
            class QuestionAnswer(dspy.Signature):
                """Answer questions using available tools."""
                question: str = dspy.InputField()
                answer: str = dspy.OutputField()

            react_agent = dspy.ReAct(
                signature=QuestionAnswer,
                tools=dspy_tools,
                max_iters=5
            )

            # Use the agent
            result = await react_agent.acall(
                question="What is 25 + 17?"
            )
            print(result.answer)

asyncio.run(main())