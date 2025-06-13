# Ultimate MCP Server Test Scripts

This directory contains test scripts to validate your Ultimate MCP Server functionality.

## Prerequisites

Make sure you have FastMCP installed:
```bash
pip install fastmcp
# or
uv add fastmcp
```

Also install aiohttp for REST API testing:
```bash
pip install aiohttp
# or  
uv add aiohttp
```

## Test Scripts

### 1. `quick_test.py` - Quick Connectivity Test
**Purpose**: Fast basic connectivity and functionality check
**Runtime**: ~5 seconds

```bash
python quick_test.py
```

This script tests:
- ✅ Basic MCP connection
- 📢 Echo tool functionality  
- 🔌 Provider availability
- 🛠️ Tool count
- 📚 Resource count

### 2. `test_client.py` - Interactive Test Client
**Purpose**: Comprehensive testing with interactive mode
**Runtime**: Variable (can be used interactively)

```bash
python test_client.py
```

This script tests:
- 🔗 Server connection
- 📋 Tool listing and calling
- 📚 Resource reading
- 🤖 LLM completions
- 📁 Filesystem tools
- 🐍 Python execution
- 📝 Text processing tools
- 🎮 Interactive command mode

**Interactive Commands**:
- `list` - Show available tools
- `resources` - Show available resources  
- `call <tool_name> <json_params>` - Call a tool
- `read <resource_uri>` - Read a resource
- `quit` - Exit

### 3. `comprehensive_test.py` - Full Test Suite
**Purpose**: Complete validation of MCP and REST API functionality
**Runtime**: ~30 seconds

```bash
python comprehensive_test.py
```

This script tests:
- 🔧 MCP Interface (tools, providers, filesystem, Python)
- 🌐 REST API Endpoints (discovery, health, docs, cognitive states, performance, artifacts)
- 🤖 LLM Completions (actual generation with available providers)
- 🧠 Memory System (storage, retrieval, cognitive states)

## Understanding Results

### ✅ Green Check - Working Correctly
The feature is functioning as expected.

### ❌ Red X - Needs Attention  
The feature failed or is not available. Common reasons:
- API keys not configured
- Provider services unavailable
- Database connection issues
- Missing dependencies

## Your Server Configuration

Based on your server startup logs, your server has:
- **107 tools** loaded (all available tools mode)
- **7 LLM providers** configured:
  - ✅ Anthropic (3 models)
  - ✅ DeepSeek (2 models) 
  - ✅ Gemini (4 models)
  - ✅ OpenRouter (3 models)
  - ✅ Ollama (3 models) - Local
  - ✅ Grok (4 models)
  - ✅ OpenAI (47 models)

## Endpoints Available

### MCP Protocol
- `http://127.0.0.1:8013/mcp` - Main MCP streamable-HTTP endpoint

### REST API
- `http://127.0.0.1:8013/` - Discovery endpoint
- `http://127.0.0.1:8013/api/health` - Health check
- `http://127.0.0.1:8013/api/docs` - Swagger UI documentation
- `http://127.0.0.1:8013/api/cognitive-states` - Cognitive state management
- `http://127.0.0.1:8013/api/performance/overview` - Performance metrics
- `http://127.0.0.1:8013/api/artifacts` - Artifact management

### UMS Explorer
- `http://127.0.0.1:8013/api/ums-explorer` - Memory system explorer UI

## Troubleshooting

### Connection Failed
- Verify server is running on port 8013
- Check firewall settings
- Ensure no other service is using the port

### Provider Errors  
- Check API keys in environment variables
- Verify provider service availability
- Test with local Ollama first (no API key needed)

### Tool Errors
- Check filesystem permissions
- Verify Python sandbox configuration
- Check database connectivity

## Example Usage

```bash
# Quick smoke test
python quick_test.py

# Interactive exploration
python test_client.py
# Then type: list
# Then type: call echo {"message": "Hello!"}

# Full validation
python comprehensive_test.py
```

## Next Steps

After successful testing:
1. Check the Swagger UI at `http://127.0.0.1:8013/api/docs`
2. Explore the UMS Explorer at `http://127.0.0.1:8013/api/ums-explorer`  
3. Test with a real MCP client like Claude Desktop
4. Start building your applications using the MCP tools! 