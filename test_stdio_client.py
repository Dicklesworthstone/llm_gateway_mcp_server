#!/usr/bin/env python3
"""
Stdio Test Client for Ultimate MCP Server
Tests server functionality over stdio transport
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path

from fastmcp import Client


async def test_stdio_server():
    """Test Ultimate MCP Server over stdio transport."""
    print("📡 Ultimate MCP Server Stdio Test Client")
    print("=" * 50)
    print("🔗 Starting Ultimate MCP Server in stdio mode...")
    
    # Find the umcp command
    umcp_cmd = None
    if os.path.exists("uv.lock"):
        # Try uv run first
        umcp_cmd = ["uv", "run", "umcp", "run"]
    else:
        # Try direct umcp command
        umcp_cmd = ["umcp", "run"]
    
    print(f"📡 Command: {' '.join(umcp_cmd)}")
    
    try:
        # Start the server process in stdio mode
        # Note: stdio is the default mode, so no -t flag needed
        process = subprocess.Popen(
            umcp_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbuffered
            cwd=Path.cwd(),
            env=os.environ.copy()
        )
        
        print("✅ Server process started, connecting via stdio...")
        
        # Create FastMCP client for stdio transport
        # Use the process stdin/stdout for communication
        async with Client.stdio(
            process.stdin,
            process.stdout
        ) as client:
            print("✅ Successfully connected to stdio server")
            
            # Test 1: List available tools
            print("\n📋 Testing tool discovery via stdio...")
            tools = await client.list_tools()
            print(f"Found {len(tools)} tools via stdio transport:")
            for i, tool in enumerate(tools[:10]):  # Show first 10
                print(f"  {i+1:2d}. {tool.name}")
            if len(tools) > 10:
                print(f"  ... and {len(tools) - 10} more tools")
            
            # Test 2: List available resources
            print("\n📚 Testing resource discovery via stdio...")
            resources = await client.list_resources()
            print(f"Found {len(resources)} resources:")
            for resource in resources:
                print(f"  - {resource.uri}")
            
            # Test 3: Echo tool test
            print("\n🔊 Testing echo tool via stdio...")
            echo_result = await client.call_tool("echo", {"message": "Hello from stdio client!"})
            if echo_result:
                echo_data = json.loads(echo_result[0].text)
                print(f"✅ Echo response: {json.dumps(echo_data, indent=2)}")
            
            # Test 4: Provider status test
            print("\n🔌 Testing provider status via stdio...")
            try:
                provider_result = await client.call_tool("get_provider_status", {})
                if provider_result:
                    provider_data = json.loads(provider_result[0].text)
                    providers = provider_data.get("providers", {})
                    print(f"✅ Found {len(providers)} providers via stdio:")
                    for name, status in providers.items():
                        available = "✅" if status.get("available") else "❌"
                        model_count = len(status.get("models", []))
                        print(f"  {available} {name}: {model_count} models")
            except Exception as e:
                print(f"❌ Provider status failed: {e}")
            
            # Test 5: Resource reading test
            print("\n📖 Testing resource reading via stdio...")
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
            print("\n🤖 Testing completion via stdio...")
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
                    print("✅ Completion via stdio:")
                    print(f"  Text: '{result_data.get('text', 'No text')}'")
                    print(f"  Model: {result_data.get('model', 'Unknown')}")
                    print(f"  Success: {result_data.get('success', False)}")
                    print(f"  Processing time: {result_data.get('processing_time', 0):.2f}s")
            except Exception as e:
                print(f"⚠️ Completion test failed (expected if no providers): {e}")
            
            # Test 7: Filesystem tool test
            print("\n📁 Testing filesystem tools via stdio...")
            try:
                dirs_result = await client.call_tool("list_allowed_directories", {})
                if dirs_result:
                    dirs_data = json.loads(dirs_result[0].text)
                    print(f"✅ Allowed directories via stdio: {dirs_data.get('count', 0)} directories")
            except Exception as e:
                print(f"❌ Filesystem test failed: {e}")
            
            # Test 8: Text processing tool test
            print("\n📝 Testing text processing via stdio...")
            try:
                ripgrep_result = await client.call_tool(
                    "run_ripgrep", 
                    {
                        "args_str": "'import' . -t py --max-count 3",
                        "input_dir": "."
                    }
                )
                if ripgrep_result:
                    ripgrep_data = json.loads(ripgrep_result[0].text)
                    if ripgrep_data.get("success"):
                        lines = ripgrep_data.get("output", "").split('\n')
                        line_count = len([l for l in lines if l.strip()])  # noqa: E741
                        print(f"✅ Ripgrep via stdio found {line_count} matching lines")
                    else:
                        print("⚠️ Ripgrep completed but found no matches")
            except Exception as e:
                print(f"❌ Text processing test failed: {e}")
            
            print("\n🎉 Stdio transport functionality test completed!")
            
        # Clean up process
        print("\n🔄 Shutting down server process...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print("✅ Server process terminated cleanly")
        except subprocess.TimeoutExpired:
            print("⚠️ Server process didn't terminate, forcing kill...")
            process.kill()
            process.wait()
        
        return True
        
    except FileNotFoundError:
        print("❌ Could not find umcp command")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the Ultimate MCP Server directory")
        print("2. Make sure umcp is installed and in PATH")
        print("3. Try running 'uv run umcp run' manually to test")
        return False
    except Exception as e:
        print(f"❌ Stdio connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the server can start in stdio mode")
        print("2. Check for any startup errors in stderr")
        print("3. Verify all dependencies are installed")
        
        # Try to get stderr from process if available
        if 'process' in locals():
            try:
                stderr_output = process.stderr.read() if process.stderr else ""
                if stderr_output:
                    print(f"\nServer stderr:\n{stderr_output}")
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                pass
        
        return False


async def test_stdio_interactive():
    """Interactive stdio testing mode."""
    print("\n🎮 Entering stdio interactive mode...")
    print("⚠️ Note: Interactive mode with stdio requires careful process management")
    print("Type 'list' to see available tools, 'quit' to exit")
    
    # Find the umcp command
    umcp_cmd = None
    if os.path.exists("uv.lock"):
        umcp_cmd = ["uv", "run", "umcp", "run"]
    else:
        umcp_cmd = ["umcp", "run"]
    
    try:
        # Start the server process
        process = subprocess.Popen(
            umcp_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            cwd=Path.cwd(),
            env=os.environ.copy()
        )
        
        async with Client.stdio(process.stdin, process.stdout) as client:
            tools = await client.list_tools()
            resources = await client.list_resources()
            
            while True:
                try:
                    command = input("\nStdio> ").strip()
                    
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
        
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    
    except Exception as e:
        print(f"❌ Stdio interactive mode failed: {e}")


def check_prerequisites():
    """Check if prerequisites are available."""
    print("🔍 Checking prerequisites...")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Not in Ultimate MCP Server directory (no pyproject.toml found)")
        return False
    
    # Check if umcp is available
    try:
        if Path("uv.lock").exists():
            result = subprocess.run(["uv", "run", "umcp", "--version"], 
                                  capture_output=True, text=True, timeout=10)
        else:
            result = subprocess.run(["umcp", "--version"], 
                                  capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ umcp command is available")
            return True
        else:
            print(f"❌ umcp command failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ umcp command not found")
        print("Try: pip install -e . or uv sync")
        return False
    except subprocess.TimeoutExpired:
        print("❌ umcp command timed out")
        return False
    except Exception as e:
        print(f"❌ Error checking umcp: {e}")
        return False


async def main():
    """Main test function."""
    # Check prerequisites first
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    print("✅ Prerequisites check passed\n")
    
    # Run basic functionality test
    success = await test_stdio_server()
    
    if success:
        # Ask if user wants interactive mode
        try:
            response = input("\nWould you like to enter stdio interactive mode? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                await test_stdio_interactive()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    else:
        print("\n❌ Basic stdio test failed. Skipping interactive mode.")


if __name__ == "__main__":
    asyncio.run(main()) 