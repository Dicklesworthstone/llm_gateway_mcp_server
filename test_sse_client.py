#!/usr/bin/env python3
"""
SSE Test Client for Ultimate MCP Server
Tests server functionality over SSE (Server-Sent Events) transport
"""

import asyncio
import json

from fastmcp import Client


async def test_sse_server():
    """Test Ultimate MCP Server over SSE transport."""
    # SSE endpoint - note the /sse path for SSE transport
    server_url = "http://127.0.0.1:8013/sse"
    
    print("🔥 Ultimate MCP Server SSE Test Client")
    print("=" * 50)
    print(f"🔗 Connecting to Ultimate MCP Server SSE endpoint at {server_url}")
    
    try:
        async with Client(server_url) as client:
            print("✅ Successfully connected to SSE server")
            
            # Test 1: List available tools
            print("\n📋 Testing tool discovery...")
            tools = await client.list_tools()
            print(f"Found {len(tools)} tools via SSE transport:")
            for i, tool in enumerate(tools[:10]):  # Show first 10
                print(f"  {i+1:2d}. {tool.name}")
            if len(tools) > 10:
                print(f"  ... and {len(tools) - 10} more tools")
            
            # Test 2: List available resources
            print("\n📚 Testing resource discovery...")
            resources = await client.list_resources()
            print(f"Found {len(resources)} resources:")
            for resource in resources:
                print(f"  - {resource.uri}")
            
            # Test 3: Echo tool test
            print("\n🔊 Testing echo tool over SSE...")
            echo_result = await client.call_tool("echo", {"message": "Hello from SSE client!"})
            if echo_result:
                echo_data = json.loads(echo_result[0].text)
                print(f"✅ Echo response: {json.dumps(echo_data, indent=2)}")
            
            # Test 4: Provider status test
            print("\n🔌 Testing provider status over SSE...")
            try:
                provider_result = await client.call_tool("get_provider_status", {})
                if provider_result:
                    provider_data = json.loads(provider_result[0].text)
                    providers = provider_data.get("providers", {})
                    print(f"✅ Found {len(providers)} providers via SSE:")
                    for name, status in providers.items():
                        available = "✅" if status.get("available") else "❌"
                        model_count = len(status.get("models", []))
                        print(f"  {available} {name}: {model_count} models")
            except Exception as e:
                print(f"❌ Provider status failed: {e}")
            
            # Test 5: Resource reading test
            print("\n📖 Testing resource reading over SSE...")
            if resources:
                try:
                    resource_uri = resources[0].uri
                    resource_content = await client.read_resource(resource_uri)
                    if resource_content:
                        content = resource_content[0].text
                        preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"✅ Resource {resource_uri} content preview:")
                        print(f"  {preview}")
                except Exception as e:
                    print(f"❌ Resource reading failed: {e}")
            
            # Test 6: Simple completion test (if providers available)
            print("\n🤖 Testing completion over SSE...")
            try:
                completion_result = await client.call_tool(
                    "generate_completion",
                    {
                        "prompt": "Say hello in exactly 3 words",
                        "provider": "ollama",
                        "model": "mix_77/gemma3-qat-tools:27b",
                        "max_tokens": 10,
                    },
                )
                if completion_result:
                    result_data = json.loads(completion_result[0].text)
                    print("✅ Completion via SSE:")
                    print(f"  Text: '{result_data.get('text', 'No text')}'")
                    print(f"  Model: {result_data.get('model', 'Unknown')}")
                    print(f"  Success: {result_data.get('success', False)}")
                    print(f"  Processing time: {result_data.get('processing_time', 0):.2f}s")
            except Exception as e:
                print(f"⚠️ Completion test failed (expected if no providers): {e}")
            
            # Test 7: Filesystem tool test
            print("\n📁 Testing filesystem tools over SSE...")
            try:
                dirs_result = await client.call_tool("list_allowed_directories", {})
                if dirs_result:
                    dirs_data = json.loads(dirs_result[0].text)
                    print(f"✅ Allowed directories via SSE: {dirs_data.get('count', 0)} directories")
            except Exception as e:
                print(f"❌ Filesystem test failed: {e}")
            
            # Test 8: Text processing tool test
            print("\n📝 Testing text processing over SSE...")
            try:
                ripgrep_result = await client.call_tool(
                    "run_ripgrep", 
                    {
                        "args_str": "'async' . -t py --max-count 5",
                        "input_dir": "."
                    }
                )
                if ripgrep_result:
                    ripgrep_data = json.loads(ripgrep_result[0].text)
                    if ripgrep_data.get("success"):
                        lines = ripgrep_data.get("output", "").split('\n')
                        line_count = len([l for l in lines if l.strip()])  # noqa: E741
                        print(f"✅ Ripgrep via SSE found {line_count} matching lines")
                    else:
                        print("⚠️ Ripgrep completed but found no matches")
            except Exception as e:
                print(f"❌ Text processing test failed: {e}")
            
            print("\n🎉 SSE transport functionality test completed!")
            return True
            
    except Exception as e:
        print(f"❌ SSE connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the server is running in SSE mode:")
        print("   umcp run -t sse")
        print("2. Verify the server is accessible at http://127.0.0.1:8013")
        print("3. Check that the SSE endpoint is available at /sse")
        return False


async def test_sse_interactive():
    """Interactive SSE testing mode."""
    server_url = "http://127.0.0.1:8013/sse"
    
    print("\n🎮 Entering SSE interactive mode...")
    print("Type 'list' to see available tools, 'quit' to exit")
    
    try:
        async with Client(server_url) as client:
            tools = await client.list_tools()
            resources = await client.list_resources()
            
            while True:
                try:
                    command = input("\nSSE> ").strip()
                    
                    if command.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    elif command.lower() == 'list':
                        print("Available tools:")
                        for i, tool in enumerate(tools[:20]):
                            print(f"  {i+1:2d}. {tool.name}")
                        if len(tools) > 20:
                            print(f"  ... and {len(tools) - 20} more")
                    elif command.lower() == 'resources':
                        print("Available resources:")
                        for resource in resources:
                            print(f"  - {resource.uri}")
                    elif command.startswith("tool "):
                        # Call tool: tool <tool_name> <json_params>
                        parts = command[5:].split(' ', 1)
                        tool_name = parts[0]
                        params = json.loads(parts[1]) if len(parts) > 1 else {}
                        
                        try:
                            result = await client.call_tool(tool_name, params)
                            if result:
                                print(f"✅ Tool result: {result[0].text}")
                            else:
                                print("❌ No result returned")
                        except Exception as e:
                            print(f"❌ Tool call failed: {e}")
                    elif command.startswith("read "):
                        # Read resource: read <resource_uri>
                        resource_uri = command[5:].strip()
                        try:
                            result = await client.read_resource(resource_uri)
                            if result:
                                content = result[0].text
                                preview = content[:500] + "..." if len(content) > 500 else content
                                print(f"✅ Resource content: {preview}")
                            else:
                                print("❌ No content returned")
                        except Exception as e:
                            print(f"❌ Resource read failed: {e}")
                    else:
                        print("Commands:")
                        print("  list          - List available tools")
                        print("  resources     - List available resources")
                        print("  tool <name> <params>  - Call a tool with JSON params")
                        print("  read <uri>    - Read a resource")
                        print("  quit          - Exit interactive mode")
                
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Command error: {e}")
    
    except Exception as e:
        print(f"❌ SSE interactive mode failed: {e}")


async def main():
    """Main test function."""
    # Run basic functionality test
    success = await test_sse_server()
    
    if success:
        # Ask if user wants interactive mode
        try:
            response = input("\nWould you like to enter SSE interactive mode? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                await test_sse_interactive()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")


if __name__ == "__main__":
    asyncio.run(main()) 