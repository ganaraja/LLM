from fastmcp import FastMCP

mcp = FastMCP(name="mcp server")

@mcp.tool()
def addition(a: int, b: int) -> int:
    """
    Add two numbers together.
    """
    print(f"Adding {a} and {b}")
    return a + b

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=9000)