#!/usr/bin/env python3
"""
Simple test client for Ultimate MCP Server
Tests basic functionality over streamable-HTTP transport
"""

import asyncio
import json

from fastmcp import Client


async def test_server_connection():
    """Test basic server connection and functionality."""
    server_url = "http://127.0.0.1:8013/mcp"

    print(f"🔗 Connecting to Ultimate MCP Server at {server_url}")

    try:
        async with Client(server_url) as client:
            print("✅ Successfully connected to server")

            # Test 1: List available tools
            print("\n📋 Listing available tools...")
            tools = await client.list_tools()
            print(f"Found {len(tools)} tools:")
            for i, tool in enumerate(tools[:10]):  # Show first 10
                print(f"  {i + 1:2d}. {tool.name}")
            if len(tools) > 10:
                print(f"  ... and {len(tools) - 10} more tools")

            # Test 2: List available resources
            print("\n📚 Listing available resources...")
            resources = await client.list_resources()
            print(f"Found {len(resources)} resources:")
            for resource in resources:
                print(f"  - {resource.uri}")

            # Test 3: Test echo tool (should be available)
            print("\n🔊 Testing echo tool...")
            try:
                echo_result = await client.call_tool("echo", {"message": "Hello from test client!"})
                if echo_result:
                    print(f"✅ Echo response: {echo_result[0].text}")
                else:
                    print("❌ Echo tool returned no response")
            except Exception as e:
                print(f"❌ Echo tool failed: {e}")

            # Test 4: Test provider status tool
            print("\n🔌 Testing provider status...")
            try:
                provider_result = await client.call_tool("get_provider_status", {})
                if provider_result:
                    provider_data = json.loads(provider_result[0].text)
                    providers = provider_data.get("providers", {})
                    print(f"✅ Found {len(providers)} providers:")
                    for name, status in providers.items():
                        available = "✅" if status.get("available") else "❌"
                        model_count = len(status.get("models", []))
                        enabled = status.get("enabled", False)
                        api_key_configured = status.get("api_key_configured", False)
                        
                        status_str = f"{available} {name}: {model_count} models"
                        if not enabled:
                            status_str += " (disabled)"
                        elif not api_key_configured:
                            status_str += " (no API key)"
                        elif status.get("error"):
                            status_str += f" (error: {status['error'][:50]}...)"
                        
                        print(f"  {status_str}")
            except Exception as e:
                print(f"❌ Provider status failed: {e}")

            # Test 5: Read a resource
            print("\n📖 Testing resource reading...")
            if resources:
                try:
                    resource_uri = resources[0].uri
                    resource_content = await client.read_resource(resource_uri)
                    if resource_content:
                        # FastMCP uses 'text' attribute, not 'content'
                        content = resource_content[0].text if hasattr(resource_content[0], 'text') else str(resource_content[0])
                        preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"✅ Resource {resource_uri} content preview:")
                        print(f"  {preview}")
                except Exception as e:
                    print(f"❌ Resource reading failed: {e}")

            # Test 6: Test a completion tool
            print("\n🤖 Testing completion tool...")
            try:
                completion_result = await client.call_tool(
                    "generate_completion",
                    {
                        "prompt": "Say hello in a creative way",
                        "provider": "ollama",  # Using local Ollama since it's available
                        "model": "mix_77/gemma3-qat-tools:27b",
                        "max_tokens": 100,  # Increased for better response
                    },
                )
                if completion_result:
                    result_data = json.loads(completion_result[0].text)
                    response_text = result_data.get('text', '').strip()
                    if response_text:
                        print(f"✅ Completion response: {response_text}")
                    else:
                        print("⚠️ Completion succeeded but returned empty text")
                        print(f"  Model: {result_data.get('model', 'unknown')}")
                        print(f"  Processing time: {result_data.get('processing_time', 0):.2f}s")
                        print(f"  Tokens: {result_data.get('tokens', {})}")
            except Exception as e:
                print(f"❌ Completion tool failed: {e}")
                # Try with a different provider
                try:
                    completion_result = await client.call_tool(
                        "generate_completion",
                        {
                            "prompt": "Say hello in a creative way",
                            "provider": "anthropic",
                            "model": "claude-3-haiku-20240307",
                            "max_tokens": 100,
                        },
                    )
                    if completion_result:
                        result_data = json.loads(completion_result[0].text)
                        response_text = result_data.get('text', '').strip()
                        if response_text:
                            print(f"✅ Completion response (anthropic): {response_text}")
                        else:
                            print("⚠️ Anthropic completion succeeded but returned empty text")
                except Exception as e2:
                    print(f"❌ Completion with anthropic also failed: {e2}")

            print("\n🎉 Basic functionality test completed!")

    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        print("Make sure the server is running at the correct address.")


async def test_specific_tools():
    """Test some specific tools that should be available."""
    server_url = "http://127.0.0.1:8013/mcp"

    print("\n🔧 Testing specific tools...")

    try:
        async with Client(server_url) as client:
            # Test filesystem tools
            print("\n📁 Testing filesystem tools...")
            try:
                # List allowed directories
                dirs_result = await client.call_tool("list_allowed_directories", {})
                if dirs_result:
                    print(f"✅ Allowed directories: {dirs_result[0].text}")

                # Try to list current directory
                ls_result = await client.call_tool("list_directory", {"path": "."})
                if ls_result:
                    ls_data = json.loads(ls_result[0].text)
                    files = ls_data.get("files", [])
                    print(f"✅ Current directory has {len(files)} items")
            except Exception as e:
                print(f"❌ Filesystem tools failed: {e}")

            # Test Python execution
            print("\n🐍 Testing Python execution...")
            try:
                python_result = await client.call_tool(
                    "execute_python",
                    {
                        "code": "print('Hello from Python!'); result = 2 + 2; print(f'2 + 2 = {result}')"
                    },
                )
                if python_result:
                    print("✅ Python execution result:")
                    result_data = json.loads(python_result[0].text)
                    # The field is called 'stdout', not 'output'
                    print(f"  Output: {result_data.get('stdout', 'No output')}")
                    print(f"  Success: {result_data.get('success', False)}")
                    if result_data.get('result') is not None:
                        print(f"  Result: {result_data.get('result')}")
            except Exception as e:
                print(f"❌ Python execution failed: {e}")

            # Test text processing tools
            print("\n📝 Testing text processing tools...")
            try:
                # Test ripgrep if available - tool expects args_str parameter
                ripgrep_result = await client.call_tool(
                    "run_ripgrep", {
                        "args_str": "'FastMCP' . -t py",
                        "input_dir": True  # Since we're searching in a directory
                    }
                )
                if ripgrep_result:
                    result_data = json.loads(ripgrep_result[0].text)
                    if result_data.get('success'):
                        print("✅ Ripgrep executed successfully")
                        stdout = result_data.get('stdout', '')
                        if stdout.strip():
                            print(f"  Found matches: {len(stdout.strip().splitlines())} lines")
                        else:
                            print("  No matches found")
                    else:
                        print(f"❌ Ripgrep failed: {result_data.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"❌ Text processing tools failed: {e}")

    except Exception as e:
        print(f"❌ Failed during specific tool testing: {e}")


async def interactive_mode():
    """Interactive mode for testing tools manually."""
    server_url = "http://127.0.0.1:8013/mcp"

    print("\n🎮 Entering interactive mode...")
    print("Type 'list' to see available tools, 'quit' to exit")

    try:
        async with Client(server_url) as client:
            tools = await client.list_tools()
            tool_names = [tool.name for tool in tools]

            while True:
                try:
                    command = input("\n> ").strip()

                    if command.lower() in ["quit", "exit", "q"]:
                        break
                    elif command.lower() == "list":
                        print("Available tools:")
                        for i, name in enumerate(tool_names[:20]):  # Show first 20
                            print(f"  {i + 1:2d}. {name}")
                        if len(tool_names) > 20:
                            print(f"  ... and {len(tool_names) - 20} more")
                    elif command.lower() == "resources":
                        resources = await client.list_resources()
                        print("Available resources:")
                        for resource in resources:
                            print(f"  - {resource.uri}")
                    elif command.startswith("call "):
                        # Parse tool call: call tool_name {"param": "value"}
                        parts = command[5:].split(" ", 1)
                        tool_name = parts[0]
                        params = {}
                        if len(parts) > 1:
                            try:
                                params = json.loads(parts[1])
                            except json.JSONDecodeError:
                                print("❌ Invalid JSON for parameters")
                                continue

                        if tool_name in tool_names:
                            try:
                                result = await client.call_tool(tool_name, params)
                                if result:
                                    print(f"✅ Result: {result[0].text}")
                                else:
                                    print("❌ No result returned")
                            except Exception as e:
                                print(f"❌ Tool call failed: {e}")
                        else:
                            print(f"❌ Tool '{tool_name}' not found")
                    elif command.startswith("read "):
                        # Read resource: read resource_uri
                        resource_uri = command[5:].strip()
                        try:
                            result = await client.read_resource(resource_uri)
                            if result:
                                # FastMCP uses 'text' attribute, not 'content'
                                content = result[0].text if hasattr(result[0], 'text') else str(result[0])
                                preview = content[:500] + "..." if len(content) > 500 else content
                                print(f"✅ Resource content: {preview}")
                            else:
                                print("❌ No content returned")
                        except Exception as e:
                            print(f"❌ Resource read failed: {e}")
                    else:
                        print("Commands:")
                        print("  list                     - List available tools")
                        print("  resources                - List available resources")
                        print("  call <tool> <json_params> - Call a tool")
                        print("  read <resource_uri>      - Read a resource")
                        print("  quit                     - Exit interactive mode")

                except KeyboardInterrupt:
                    break
                except EOFError:
                    break

    except Exception as e:
        print(f"❌ Interactive mode failed: {e}")


async def main():
    """Main test function."""
    print("🚀 Ultimate MCP Server Test Client")
    print("=" * 50)

    # Run basic connection test
    await test_server_connection()

    # Run specific tool tests
    await test_specific_tools()

    # Ask if user wants interactive mode
    try:
        response = input("\nWould you like to enter interactive mode? (y/n): ").strip().lower()
        if response in ["y", "yes"]:
            await interactive_mode()
    except (KeyboardInterrupt, EOFError):
        pass

    print("\n👋 Test client finished!")


if __name__ == "__main__":
    asyncio.run(main())
