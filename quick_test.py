#!/usr/bin/env python3
"""
Quick test script for Ultimate MCP Server connectivity
"""

import asyncio
import json

from fastmcp import Client


async def quick_test():
    """Quick connectivity and basic functionality test."""
    server_url = "http://127.0.0.1:8013/mcp"
    
    print(f"🔗 Testing connection to {server_url}")
    
    try:
        async with Client(server_url) as client:
            print("✅ Connected successfully!")
            
            # Test 1: Echo
            echo_result = await client.call_tool("echo", {"message": "Quick test"})
            print(f"📢 Echo: {echo_result[0].text}")
            
            # Test 2: Provider status
            provider_result = await client.call_tool("get_provider_status", {})
            provider_data = json.loads(provider_result[0].text)
            available_providers = [name for name, status in provider_data.get('providers', {}).items() 
                                 if status.get('available')]
            print(f"🔌 Available providers: {', '.join(available_providers)}")
            
            # Test 3: Tool count
            tools = await client.list_tools()
            print(f"🛠️  Available tools: {len(tools)}")
            
            # Test 4: Resources
            resources = await client.list_resources()
            print(f"📚 Available resources: {len(resources)}")
            
            print("🎉 All tests passed!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(quick_test()) 