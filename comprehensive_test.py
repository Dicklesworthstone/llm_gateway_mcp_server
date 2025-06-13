#!/usr/bin/env python3
"""
Comprehensive test script for Ultimate MCP Server
Tests specific tools and REST API endpoints
"""

import asyncio
import json

import aiohttp
from fastmcp import Client


async def test_mcp_interface():
    """Test the MCP interface functionality."""
    server_url = "http://127.0.0.1:8013/mcp"
    
    print("🔧 Testing MCP Interface")
    print("=" * 40)
    
    try:
        async with Client(server_url) as client:
            print("✅ MCP client connected")
            
            # Test core tools
            tools_to_test = [
                ("echo", {"message": "Hello MCP!"}),
                ("get_provider_status", {}),
                ("list_models", {}),
            ]
            
            for tool_name, params in tools_to_test:
                try:
                    result = await client.call_tool(tool_name, params)
                    if result:
                        print(f"✅ {tool_name}: OK")
                        # Show sample of result for key tools
                        if tool_name == "get_provider_status":
                            data = json.loads(result[0].text)
                            provider_count = len(data.get('providers', {}))
                            print(f"   → {provider_count} providers configured")
                        elif tool_name == "list_models":
                            data = json.loads(result[0].text)
                            total_models = sum(len(models) for models in data.get('models', {}).values())
                            print(f"   → {total_models} total models available")
                    else:
                        print(f"❌ {tool_name}: No response")
                except Exception as e:
                    print(f"❌ {tool_name}: {e}")
            
            # Test filesystem tools
            print("\n📁 Testing filesystem access...")
            try:
                dirs_result = await client.call_tool("list_allowed_directories", {})
                if dirs_result:
                    print("✅ Filesystem access configured")
            except Exception as e:
                print(f"❌ Filesystem access: {e}")
            
            # Test Python execution
            print("\n🐍 Testing Python sandbox...")
            try:
                python_result = await client.call_tool("execute_python", {
                    "code": "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}')"
                })
                if python_result:
                    result_data = json.loads(python_result[0].text)
                    if result_data.get('success'):
                        print("✅ Python sandbox working")
                        print(f"   → {result_data.get('output', '').strip()}")
                    else:
                        print("❌ Python sandbox failed")
            except Exception as e:
                print(f"❌ Python sandbox: {e}")
                
    except Exception as e:
        print(f"❌ MCP interface failed: {e}")


async def test_rest_api():
    """Test the REST API endpoints."""
    base_url = "http://127.0.0.1:8013"
    
    print("\n🌐 Testing REST API Endpoints")
    print("=" * 40)
    
    async with aiohttp.ClientSession() as session:
        # Test discovery endpoint
        try:
            async with session.get(f"{base_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Discovery endpoint: {data.get('type')}")
                    print(f"   → Transport: {data.get('transport')}")
                    print(f"   → Endpoint: {data.get('endpoint')}")
                else:
                    print(f"❌ Discovery endpoint: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Discovery endpoint: {e}")
        
        # Test health endpoint
        try:
            async with session.get(f"{base_url}/api/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health endpoint: {data.get('status')}")
                else:
                    print(f"❌ Health endpoint: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Health endpoint: {e}")
        
        # Test OpenAPI docs
        try:
            async with session.get(f"{base_url}/api/docs") as response:
                if response.status == 200:
                    print("✅ Swagger UI accessible")
                else:
                    print(f"❌ Swagger UI: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Swagger UI: {e}")
        
        # Test cognitive states endpoint
        try:
            async with session.get(f"{base_url}/api/cognitive-states") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Cognitive states: {data.get('total', 0)} states")
                else:
                    print(f"❌ Cognitive states: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Cognitive states: {e}")
        
        # Test performance overview
        try:
            async with session.get(f"{base_url}/api/performance/overview") as response:
                if response.status == 200:
                    data = await response.json()
                    overview = data.get('overview', {})
                    print(f"✅ Performance overview: {overview.get('total_actions', 0)} actions")
                else:
                    print(f"❌ Performance overview: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Performance overview: {e}")
        
        # Test artifacts endpoint
        try:
            async with session.get(f"{base_url}/api/artifacts") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Artifacts: {data.get('total', 0)} artifacts")
                else:
                    print(f"❌ Artifacts: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Artifacts: {e}")


async def test_tool_completions():
    """Test actual completions with available providers."""
    server_url = "http://127.0.0.1:8013/mcp"
    
    print("\n🤖 Testing LLM Completions")
    print("=" * 40)
    
    try:
        async with Client(server_url) as client:
            # Get available providers first
            provider_result = await client.call_tool("get_provider_status", {})
            provider_data = json.loads(provider_result[0].text)
            
            available_providers = []
            for name, status in provider_data.get('providers', {}).items():
                if status.get('available') and status.get('models'):
                    available_providers.append((name, status['models'][0]))
            
            if not available_providers:
                print("❌ No providers available for testing")
                return
            
            # Test with first available provider
            provider_name, model_info = available_providers[0]
            model_id = model_info.get('id')
            
            print(f"🧪 Testing with {provider_name} / {model_id}")
            
            try:
                result = await client.call_tool("generate_completion", {
                    "prompt": "Count from 1 to 5",
                    "provider": provider_name,
                    "model": model_id,
                    "max_tokens": 50
                })
                
                if result:
                    response_data = json.loads(result[0].text)
                    if response_data.get('success', True):
                        print("✅ Completion successful")
                        print(f"   → Response: {response_data.get('text', '')[:100]}...")
                        if 'usage' in response_data:
                            usage = response_data['usage']
                            print(f"   → Tokens: {usage.get('total_tokens', 'N/A')}")
                    else:
                        print(f"❌ Completion failed: {response_data.get('error')}")
                else:
                    print("❌ No completion response")
                    
            except Exception as e:
                print(f"❌ Completion error: {e}")
                
    except Exception as e:
        print(f"❌ Completion test failed: {e}")


async def test_memory_system():
    """Test the memory and cognitive state system."""
    server_url = "http://127.0.0.1:8013/mcp"
    
    print("\n🧠 Testing Memory System")
    print("=" * 40)
    
    try:
        async with Client(server_url) as client:
            # Test memory storage
            try:
                memory_result = await client.call_tool("store_memory", {
                    "memory_type": "test",
                    "content": "This is a test memory for the test client",
                    "importance": 7.5,
                    "tags": ["test", "client"]
                })
                
                if memory_result:
                    memory_data = json.loads(memory_result[0].text)
                    if memory_data.get('success'):
                        memory_id = memory_data.get('memory_id')
                        print(f"✅ Memory stored: {memory_id}")
                        
                        # Test memory retrieval
                        try:
                            get_result = await client.call_tool("get_memory_by_id", {
                                "memory_id": memory_id
                            })
                            
                            if get_result:
                                print("✅ Memory retrieved successfully")
                        except Exception as e:
                            print(f"❌ Memory retrieval: {e}")
                            
                    else:
                        print(f"❌ Memory storage failed: {memory_data.get('error')}")
                        
            except Exception as e:
                print(f"❌ Memory system: {e}")
                
            # Test cognitive state
            try:
                state_result = await client.call_tool("save_cognitive_state", {
                    "state_type": "test_state",
                    "description": "Test cognitive state from client",
                    "data": {"test": True, "client": "test_client"}
                })
                
                if state_result:
                    state_data = json.loads(state_result[0].text)
                    if state_data.get('success'):
                        print("✅ Cognitive state saved")
                    else:
                        print(f"❌ Cognitive state failed: {state_data.get('error')}")
                        
            except Exception as e:
                print(f"❌ Cognitive state: {e}")
                
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")


async def main():
    """Run all comprehensive tests."""
    print("🚀 Ultimate MCP Server Comprehensive Test Suite")
    print("=" * 60)
    
    # Test MCP interface
    await test_mcp_interface()
    
    # Test REST API
    await test_rest_api()
    
    # Test completions
    await test_tool_completions()
    
    # Test memory system
    await test_memory_system()
    
    print("\n🎯 Comprehensive testing completed!")
    print("\nIf you see mostly ✅ symbols, your server is working correctly!")
    print("Any ❌ symbols indicate areas that may need attention.")


if __name__ == "__main__":
    asyncio.run(main()) 