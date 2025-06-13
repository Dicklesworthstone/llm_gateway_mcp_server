#!/usr/bin/env python3
"""
Test script to verify Ultimate MCP Server is working correctly
"""

import asyncio
import json
from fastmcp import Client

async def test_streamable_http():
    """Test connection to streamable-http server"""
    server_url = "http://127.0.0.1:8013/mcp"
    
    print("🧪 Testing Streamable-HTTP Connection")
    print("=" * 40)
    
    try:
        async with Client(server_url) as client:
            print("✅ Connected successfully!")
            
            # Test basic functionality
            tools = await client.list_tools()
            print(f"📋 Found {len(tools)} tools")
            
            # Test echo
            echo_result = await client.call_tool("echo", {"message": "Connection test successful!"})
            print(f"📢 Echo: {json.loads(echo_result[0].text)['message']}")
            
            print("🎉 All tests passed!")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_streamable_http())