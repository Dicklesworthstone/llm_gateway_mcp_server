# 🧠 Ultimate MCP Server

<div align="center">

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Protocol](https://img.shields.io/badge/Protocol-MCP-purple.svg)](https://github.com/modelcontextprotocol)

### A comprehensive Model Context Protocol (MCP) server providing advanced AI agents with dozens of powerful capabilities for cognitive augmentation, tool use, and intelligent orchestration

<img src="https://raw.githubusercontent.com/Dicklesworthstone/ultimate_mcp_server/refs/heads/main/ultimate_mcp_banner.webp" alt="Illustration" width="600"/>

**[Getting Started](#getting-started) • [Key Features](#key-features) • [Usage Examples](#usage-examples) • [Architecture](#architecture)**

</div>

---

## 🤖 What is Ultimate MCP Server?

**Ultimate MCP Server** is a comprehensive MCP-native system that serves as a complete AI agent operating system. It exposes dozens of powerful capabilities through the Model Context Protocol, enabling advanced AI agents to access a rich ecosystem of tools, cognitive systems, and specialized services.

While it includes intelligent task delegation from sophisticated models (e.g., Claude 3.7 Sonnet) to cost-effective ones (e.g., Gemini Flash 2.0 Lite), this is just one facet of its extensive functionality. The server provides unified access to multiple LLM providers while optimizing for **cost**, **performance**, and **quality**.

The system offers integrated cognitive memory systems, browser automation, Excel manipulation, database interactions, document processing, command-line utilities, dynamic API integration, OCR capabilities, vector operations, entity relation graphs, SQL database interactions, audio transcription, and much more. These capabilities transform an AI agent from a conversational interface into a powerful autonomous system capable of complex, multi-step operations across digital environments.

<div align="center">

<img src="https://raw.githubusercontent.com/Dicklesworthstone/ultimate_mcp_server/refs/heads/main/ultimate_mcp_logo.webp" alt="Illustration" width="600"/>

</div>

---## 🎯 Vision: The Complete AI Agent Operating System

At its core, Ultimate MCP Server represents a fundamental shift in how AI agents operate in digital environments. It serves as a comprehensive operating system for AI, providing:

- 🧠 A unified cognitive architecture that enables persistent memory, reasoning, and contextual awareness
- ⚙️ Seamless access to dozens of specialized tools spanning web browsing, document processing, data analysis, and more
- 💻 Direct system-level capabilities for filesystem operations, database interactions, and command-line utilities
- 🔄 Dynamic workflow capabilities for complex multi-step task orchestration and execution
- 🌐 Intelligent integration of various LLM providers with cost, quality, and performance optimization
- 🚀 Advanced vector operations, knowledge graphs, and retrieval-augmented generation for enhanced AI capabilities

This approach mirrors how sophisticated operating systems provide applications with access to hardware, services, and resources - but designed specifically for augmenting AI agents with powerful new capabilities beyond their native abilities.

---

## 🔌 MCP-Native Architecture

The server is built entirely on the [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol), making it specifically designed to work with AI agents like Claude. All functionality is exposed through standardized MCP tools that can be directly called by these agents, creating a seamless integration layer between AI agents and a comprehensive ecosystem of capabilities, services, and external systems.

---

## 🧬 Core Use Cases: AI Agent Augmentation and Ecosystem

The Ultimate MCP Server transforms AI agents like Claude 3.7 Sonnet into autonomous systems capable of sophisticated operations across digital environments:

```plaintext
                        interacts with
┌─────────────┐ ────────────────────────► ┌───────────────────┐         ┌──────────────┐
│ Claude 3.7  │                           │   Ultimate MCP     │ ───────►│ LLM Providers│
│   (Agent)   │ ◄──────────────────────── │     Server        │ ◄───────│ External     │
└─────────────┘      returns results      └───────────────────┘         │ Systems      │
                                                │                        └──────────────┘
                                                ▼
                      ┌─────────────────────────────────────────────┐
                      │ Cognitive Memory Systems                    │
                      │ Web & Data: Browser, DB, RAG, Vector Search │
                      │ Documents: Excel, OCR, PDF, Filesystem      │
                      │ Analysis: Entity Graphs, Classification     │
                      │ Integration: APIs, CLI, Audio, Multimedia   │
                      └─────────────────────────────────────────────┘
```

**Example workflow:**

1. An AI agent receives a complex task requiring multiple capabilities beyond its native abilities
2. The agent uses the Ultimate MCP Server to access specialized tools and services as needed
3. The agent can leverage the cognitive memory system to maintain state and context across operations
4. Complex tasks like research, data analysis, document creation, and multimedia processing become possible
5. The agent can orchestrate multi-step workflows combining various tools in sophisticated sequences
6. Results are returned in standard MCP format, enabling the agent to understand and work with them
7. One important benefit is cost optimization through delegating appropriate tasks to more efficient models

This integration unlocks transformative capabilities that enable AI agents to autonomously complete complex projects while intelligently utilizing resources - including potentially saving 70-90% on API costs by using specialized tools and cost-effective models where appropriate.

---

## 💡 Why Use Ultimate MCP Server?

### 🧰 Comprehensive AI Agent Toolkit
A unified hub enabling advanced AI agents to access an extensive ecosystem of tools:
-   🌐 Perform complex web automation tasks (**Playwright** integration).
-   📊 Manipulate and analyze **Excel** spreadsheets with deep integration.
-   🧠 Access rich **cognitive memory** systems for persistent agent state.
-   💾 Interact securely with the **filesystem**.
-   🗄️ Interact with **databases** through SQL operations.
-   🖼️ Process documents with **OCR** capabilities.
-   🔍 Perform sophisticated **vector search** and **RAG** operations.
-   🏷️ Utilize specialized **text processing** and **classification**.
-   ⌨️ Leverage command-line tools like **ripgrep**, **awk**, **sed**, **jq**.
-   🔌 Dynamically integrate external **REST APIs**.
-   ✨ Use **meta tools** for self-discovery, optimization, and documentation refinement.

### 💵 Cost Optimization
API costs for advanced models can be substantial. Ultimate MCP Server helps reduce costs by:
-   📉 Routing appropriate tasks to cheaper models (e.g., $0.01/1K tokens vs $0.15/1K tokens).
-   ⚡ Implementing **advanced caching** (exact, semantic, task-aware) to avoid redundant API calls.
-   💰 Tracking and **optimizing costs** across providers.
-   🧭 Enabling **cost-aware task routing** decisions.
-   🛠️ Handling routine processing with specialized non-LLM tools (filesystem, CLI utils, etc.).

### 🌐 Provider Abstraction
Avoid provider lock-in with a unified interface:
-   🔗 Standard API for **OpenAI**, **Anthropic (Claude)**, **Google (Gemini)**, **xAI (Grok)**, **DeepSeek**, and **OpenRouter**.
-   ⚙️ Consistent parameter handling and response formatting.
-   🔄 Ability to **swap providers** without changing application code.
-   🛡️ Protection against provider-specific outages and limitations through fallback mechanisms.

### 📑 Comprehensive Document and Data Processing
Process documents and data efficiently:
-   ✂️ Break documents into semantically meaningful **chunks**.
-   🚀 Process chunks in **parallel** across multiple models.
-   📊 Extract **structured data** (JSON, tables, key-value) from unstructured text.
-   ✍️ Generate **summaries** and insights from large texts.
-   🔁 Convert formats (**HTML to Markdown**, documents to structured data).
-   👁️ Apply **OCR** to images and PDFs with optional LLM enhancement.

---

## 🚀 Key Features

### 🔌 MCP Protocol Integration
-   **Native MCP Server**: Built on the Model Context Protocol for seamless AI agent integration.
-   **MCP Tool Framework**: All functionality exposed through standardized MCP tools with clear schemas.
-   **Tool Composition**: Tools can be combined in workflows using dependencies.
-   **Tool Discovery**: Supports dynamic listing and capability discovery for agents.

### 🤖 Intelligent Task Delegation
-   **Task Routing**: Analyzes tasks and routes to appropriate models or specialized tools.
-   **Provider Selection**: Chooses provider/model based on task requirements, cost, quality, or speed preferences.
-   **Cost-Performance Balancing**: Optimizes delegation strategy.
-   **Delegation Tracking**: Monitors delegation patterns, costs, and outcomes (via Analytics).

### 🌍 Provider Integration
-   **Multi-Provider Support**: First-class support for OpenAI, Anthropic, Google, DeepSeek, xAI (Grok), OpenRouter. Extensible architecture.
-   **Model Management**: Handles different model capabilities, context windows, and pricing. Automatic selection and fallback mechanisms.

### 💾 Advanced Caching
-   **Multi-level Caching**: Exact match, semantic similarity, and task-aware strategies.
-   **Persistent Cache**: Disk-based persistence (e.g., DiskCache) with fast in-memory access layer.
-   **Cache Analytics**: Tracks cache hit rates, estimated cost savings.

### 📄 Document Tools
-   **Smart Chunking**: Token-based, semantic boundary detection, structural analysis methods. Configurable overlap.
-   **Document Operations**: Summarization (paragraph, bullets), entity extraction, question generation, batch processing.

### 📁 Secure Filesystem Operations
-   **Path Management**: Robust validation, normalization, symlink security checks, configurable allowed directories.
-   **File Operations**: Read/write with encoding handling, smart text editing/replacement, metadata retrieval.
-   **Directory Operations**: Creation, listing, tree visualization, secure move/copy.
-   **Search Capabilities**: Recursive search with pattern matching and filtering.
-   **Security Focus**: Designed to prevent directory traversal and enforce boundaries.

### ✨ Autonomous Tool Documentation Refiner
-   **Automated Improvement**: Systematically analyzes, tests, and refines MCP tool documentation (docstrings, schemas, examples).
-   **Agent Simulation**: Identifies ambiguities from an LLM agent's perspective.
-   **Adaptive Testing**: Generates and executes schema-aware test cases.
-   **Failure Analysis**: Uses LLM ensembles to diagnose documentation weaknesses.
-   **Iterative Refinement**: Continuously improves documentation quality.
-   **(See dedicated section for more details)**

### 🌐 Browser Automation with Playwright
-   **Full Control**: Navigate, click, type, scrape data, screenshots, PDFs, file up/download, JS execution.
-   **Research**: Automate searches across engines, extract structured data, monitor sites.
-   **Synthesis**: Combine findings from multiple web sources into reports.

### 🧠 Cognitive & Agent Memory System
-   **Memory Hierarchy**: Working, episodic, semantic, procedural levels.
-   **Knowledge Management**: Store/retrieve memories with metadata, relationships, importance tracking.
-   **Workflow Tracking**: Record agent actions, reasoning chains, artifacts, dependencies.
-   **Smart Operations**: Memory consolidation, reflection generation, relevance-based optimization, decay.

### 📊 Excel Spreadsheet Automation
-   **Direct Manipulation**: Create, modify, format Excel files via natural language or structured instructions. Analyze formulas.
-   **Template Learning**: Learn from examples, adapt templates, apply formatting patterns.
-   **VBA Macro Generation**: Generate VBA code from instructions for complex automation.

### 🏗️ Structured Data Extraction
-   **JSON Extraction**: Extract structured JSON with schema validation.
-   **Table Extraction**: Extract tables in multiple formats (JSON, CSV, Markdown).
-   **Key-Value Extraction**: Simple K/V pair extraction.
-   **Semantic Schema Inference**: Attempt to generate schemas from text.

### ⚔️ Tournament Mode
-   **Model Competitions**: Run head-to-head comparisons for code or text generation tasks.
-   **Multi-Model Evaluation**: Compare outputs from different models/providers simultaneously.
-   **Performance Metrics**: Evaluate correctness, efficiency, style, etc. Persist results.

### 🗄️ SQL Database Interactions
-   **Query Execution**: Run SQL queries against various DB types (SQLite, PostgreSQL, etc. via SQLAlchemy).
-   **Schema Analysis**: Analyze schemas, suggest optimizations (using LLM).
-   **Data Exploration**: Browse tables, visualize contents.
-   **Query Generation**: Generate SQL from natural language descriptions.

### 🔗 Entity Relation Graphs
-   **Entity Extraction**: Identify entities (people, orgs, locations, etc.).
-   **Relationship Mapping**: Discover and map connections between entities.
-   **Knowledge Graph Construction**: Build persistent graphs (e.g., using NetworkX).
-   **Graph Querying**: Extract insights using graph traversal or LLM-based queries.

### 🔎 Advanced Vector Operations
-   **Semantic Search**: Find similar content using vector embeddings.
-   **Vector Storage Integration**: Interfaces with vector databases or local stores.
-   **Hybrid Search**: Combines keyword and semantic search (e.g., via Marqo integration).
-   **Batched Processing**: Efficient embedding generation and searching for large datasets.

### 📚 Retrieval-Augmented Generation (RAG)
-   **Contextual Generation**: Augments prompts with relevant retrieved documents/chunks.
-   **Accuracy Improvement**: Reduces hallucinations by grounding responses in provided context.
-   **Workflow Integration**: Seamlessly combines retrieval (vector/keyword search) with generation. Customizable strategies.

### 🎙️ Audio Transcription
-   **Speech-to-Text**: Convert audio files (e.g., WAV, MP3) to text using models like Whisper.
-   **Speaker Diarization**: Identify different speakers (if supported by the model/library).
-   **Transcript Enhancement**: Clean and format transcripts using LLMs.
-   **Multi-language Support**: Handles various languages based on the underlying transcription model.

### 🏷️ Text Classification
-   **Custom Classifiers**: Apply text classification models (potentially fine-tuned or using zero-shot LLMs).
-   **Multi-label Classification**: Assign multiple categories.
-   **Confidence Scoring**: Provide probabilities for classifications.
-   **Batch Processing**: Classify large document sets efficiently.

### 👁️ OCR Tools
-   **PDF/Image Extraction**: Uses Tesseract or other OCR engines, enhanced with LLM correction/formatting.
-   **Preprocessing**: Image denoising, thresholding, deskewing options.
-   **Structure Analysis**: Extracts PDF metadata and structure.
-   **Batch Processing**: Handles multiple files concurrently.
-   **(Requires `ocr` extra dependencies: `uv pip install -e ".[ocr]"`)**

### 📝 Text Redline Tools
-   **HTML Redline Generation**: Visual diffs (insertions, deletions, moves) between text/HTML. Standalone HTML output.
-   **Document Comparison**: Compares various formats with intuitive highlighting.

### 🔄 HTML to Markdown Conversion
-   **Intelligent Conversion**: Detects content type, uses libraries like `readability-lxml`, `trafilatura`, `markdownify`.
-   **Content Extraction**: Filters boilerplate, preserves structure (tables, links).
-   **Markdown Optimization**: Cleans and normalizes output.

### 📈 Workflow Optimization Tools
-   **Cost Estimation/Comparison**: Pre-execution cost estimates, model cost comparisons.
-   **Model Selection Guidance**: Recommends models based on task, budget, performance needs.
-   **Workflow Execution Engine**: Runs multi-stage pipelines with dependencies, parallel execution, variable passing.

### 💻 Local Text Processing Tools (CLI Integration)
-   **Offline Power**: Securely wrap and expose command-line tools like `ripgrep` (fast regex search), `awk` (text processing), `sed` (stream editor), `jq` (JSON processing) as MCP tools. Process text locally without API calls.

### ⏱️ Model Performance Benchmarking
-   **Empirical Measurement**: Tools to measure actual speed (tokens/sec), latency across providers/models.
-   **Performance Profiles**: Generate comparative reports based on real-world performance.
-   **Data-Driven Optimization**: Use benchmark data to inform routing decisions.

### 📡 Multiple Transport Modes
-   **Streamable-HTTP (Recommended)**: Modern HTTP transport with streaming request/response bodies, optimal for HTTP-based MCP clients.
-   **Server-Sent Events (SSE)**: Legacy HTTP transport using server-sent events for real-time streaming.
-   **Standard I/O (stdio)**: Direct process communication for embedded integrations.
-   **Real-time Streaming**: Token-by-token updates for LLM completions across all HTTP transports.
-   **Progress Monitoring**: Track progress of long-running jobs (chunking, batch processing).
-   **Event-Based Architecture**: Subscribe to specific server events.

### ✨ Multi-Model Synthesis
-   **Comparative Analysis**: Analyze outputs from multiple models side-by-side.
-   **Response Synthesis**: Combine best elements, generate meta-responses, create consensus outputs.
-   **Collaborative Reasoning**: Implement workflows where different models handle different steps.

### 🧩 Extended Model Support
-   **Grok Integration**: Native support for xAI's Grok.
-   **DeepSeek Support**: Optimized handling for DeepSeek models.
-   **OpenRouter Integration**: Access a wide variety via OpenRouter API key.
-   **Gemini Integration**: Comprehensive support for Google's Gemini models.
-   **Anthropic Integration**: Full support for Claude models including Claude 3.5 Sonnet and Haiku.
-   **OpenAI Integration**: Complete support for GPT-3.5, GPT-4.0, and newer models.

### 🔧 Meta Tools for Self-Improvement & Dynamic Integration
-   **Tool Discovery**: Agents can query available tools, parameters, descriptions (`list_tools`).
-   **Usage Recommendations**: Get AI-driven advice on tool selection/combination for tasks.
-   **External API Integration**: Dynamically register REST APIs via OpenAPI specs, making endpoints available as callable MCP tools (`register_api`, `call_dynamic_tool`).
-   **Documentation Generation**: Part of the Autonomous Refiner feature.

### 📊 Analytics and Reporting
-   **Usage Tracking**: Monitors tokens, costs, requests, success/error rates per provider/model/tool.
-   **Real-Time Monitoring**: Live dashboard or stream of usage stats.
-   **Detailed Reporting**: Generate historical cost/usage reports, identify trends, export data.
-   **Optimization Insights**: Helps identify expensive operations or inefficient patterns.

### 📜 Prompt Templates and Management
-   **Jinja2 Templates**: Create reusable, dynamic prompts with variables, conditionals, includes.
-   **Prompt Repository**: Store, retrieve, categorize, and version control prompts.
-   **Metadata**: Add descriptions, authorship, usage examples to templates.
-   **Optimization**: Test and compare template performance and token usage.

### 🛡️ Error Handling and Resilience
-   **Intelligent Retries**: Automatic retries with exponential backoff for transient errors (rate limits, network issues).
-   **Fallback Mechanisms**: Configurable provider fallbacks on primary failure.
-   **Detailed Error Reporting**: Captures comprehensive error context for debugging.
-   **Input Validation**: Pre-flight checks for common issues (e.g., token limits, required parameters).

### ⚙️ System Features
-   **Rich Logging**: Colorful, informative console logs via `Rich`.
-   **Health Monitoring**: `/healthz` endpoint for readiness checks.
-   **Command-Line Interface**: `umcp` CLI for management and interaction.

---

## 📦 Getting Started

### 🧪 Install

```bash
# Install uv (fast Python package manager) if you don't have it:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/Dicklesworthstone/ultimate_mcp_server.git
cd ultimate_mcp_server

# Create a virtual environment and install dependencies using uv:
uv venv --python 3.13
source .venv/bin/activate
uv lock --upgrade
uv sync --all-extras
```
*Note: The `uv sync --all-extras` command installs all optional extras defined in the project (e.g., OCR, Browser Automation, Excel). If you only need specific extras, adjust your project dependencies and run `uv sync` without `--all-extras`.*

### ⚙️ .env Configuration

Create a file named `.env` in the root directory of the cloned repository. Add your API keys and any desired configuration overrides:

```bash
# --- API Keys (at least one provider required) ---
OPENAI_API_KEY=your_openai_sk-...
ANTHROPIC_API_KEY=your_anthropic_sk-...
GEMINI_API_KEY=your_google_ai_studio_key... # For Google AI Studio (Gemini API)
# Or use GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json for Vertex AI
DEEPSEEK_API_KEY=your_deepseek_key...
OPENROUTER_API_KEY=your_openrouter_key...
GROK_API_KEY=your_grok_key... # For Grok via xAI API

# --- Server Configuration (Defaults shown) ---
GATEWAY_SERVER_PORT=8013
GATEWAY_SERVER_HOST=127.0.0.1 # Change to 0.0.0.0 to listen on all interfaces (needed for Docker/external access)
# GATEWAY_API_PREFIX=/

# --- Logging Configuration (Defaults shown) ---
LOG_LEVEL=INFO # DEBUG, INFO, WARNING, ERROR, CRITICAL
USE_RICH_LOGGING=true # Set to false for plain text logs

# --- Cache Configuration (Defaults shown) ---
GATEWAY_CACHE_ENABLED=true
GATEWAY_CACHE_TTL=86400 # Default Time-To-Live in seconds (24 hours)
# GATEWAY_CACHE_TYPE=memory # Options might include 'memory', 'redis', 'diskcache' (check implementation)
# GATEWAY_CACHE_MAX_SIZE=1000 # Example: Max number of items for memory cache
# GATEWAY_CACHE_DIR=./.cache # Directory for disk cache storage

# --- Provider Timeouts & Retries (Defaults shown) ---
# GATEWAY_PROVIDER_TIMEOUT=120 # Default timeout in seconds for API calls
# GATEWAY_PROVIDER_MAX_RETRIES=3 # Default max retries on failure

# --- Provider-Specific Configuration ---
# GATEWAY_OPENAI_DEFAULT_MODEL=gpt-4.1-mini # Customize default model
# GATEWAY_ANTHROPIC_DEFAULT_MODEL=claude-3-5-sonnet-20241022 # Customize default model
# GATEWAY_GEMINI_DEFAULT_MODEL=gemini-2.0-pro # Customize default model

# --- Tool Specific Config (Examples) ---
# FILESYSTEM__ALLOWED_DIRECTORIES=["/path/to/safe/dir1","/path/to/safe/dir2"] # For Filesystem tools (JSON array)
# GATEWAY_AGENT_MEMORY_DB_PATH=unified_agent_memory.db # Path for agent memory database
# GATEWAY_PROMPT_TEMPLATES_DIR=./prompt_templates # Directory for prompt templates
```

### ▶️ Run

Make sure your virtual environment is active (`source .venv/bin/activate`).

```bash
# Start the MCP server with all registered tools found
umcp run

# Start the server including only specific tools
umcp run --include-tools completion chunk_document read_file write_file

# Start the server excluding specific tools
umcp run --exclude-tools browser_init browser_navigate research_and_synthesize_report

# Start with Docker (ensure .env file exists in the project root or pass environment variables)
docker compose up --build # Add --build the first time or after changes
```

Once running, the server will typically be available at `http://localhost:8013` (or the host/port configured in your `.env` or command line). You should see log output indicating the server has started and which tools are registered.

## 💻 Command Line Interface (CLI)

The Ultimate MCP Server provides a powerful command-line interface (CLI) through the `umcp` command that allows you to manage the server, interact with LLM providers, test features, and explore examples. This section details all available commands and their options.

### 🌟 Global Options

The `umcp` command supports the following global option:

```bash
umcp --version  # Display version information
```

### 🚀 Server Management

#### Starting the Server

The `run` command starts the Ultimate MCP Server with specified options:

```bash
# Basic server start with default settings from .env
umcp run

# Run on a specific host (-h) and port (-p)
umcp run -h 0.0.0.0 -p 9000

# Run with multiple worker processes (-w)
umcp run -w 4

# Enable debug logging (-d)
umcp run -d

# Use stdio transport (-t)
umcp run -t stdio

# Use streamable-http transport (recommended for HTTP clients)
umcp run -t shttp

# Run only with specific tools (no shortcut for --include-tools)
umcp run --include-tools completion chunk_document read_file write_file

# Run with all tools except certain ones (no shortcut for --exclude-tools)
umcp run --exclude-tools browser_init browser_navigate
```

Example output:
```
┌─ Starting Ultimate MCP Server ───────────────────┐
│ Host: 0.0.0.0                                    │
│ Port: 9000                                       │
│ Workers: 4                                       │
│ Transport mode: streamable-http                  │
└────────────────────────────────────────────────┘

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```

Available options:
- `-h, --host`: Host or IP address to bind the server to (default: from .env)
- `-p, --port`: Port to listen on (default: from .env)
- `-w, --workers`: Number of worker processes to spawn (default: from .env)
- `-t, --transport-mode`: Transport mode for server communication ('shttp' for streamable-http, 'sse', or 'stdio', default: shttp)
- `-d, --debug`: Enable debug logging
- `--include-tools`: List of tool names to include (comma-separated)
- `--exclude-tools`: List of tool names to exclude (comma-separated)

### 🔌 Provider Management

#### Listing Providers

The `providers` command displays information about configured LLM providers:

```bash
# List all configured providers
umcp providers

# Check API keys (-c) for all configured providers
umcp providers -c

# List available models (no shortcut for --models)
umcp providers --models

# Check keys and list models
umcp providers -c --models
```

Example output:
```
┌─ LLM Providers ──────────────────────────────────────────────────┐
│ Provider   Status   Default Model            API Key             │
├───────────────────────────────────────────────────────────────────┤
│ openai     ✓        gpt-4.1-mini            sk-...5vX [VALID]    │
│ anthropic  ✓        claude-3-5-sonnet-20241022 sk-...Hr [VALID]  │
│ gemini     ✓        gemini-2.0-pro          [VALID]              │
│ deepseek   ✗        deepseek-chat           [NOT CONFIGURED]     │
│ openrouter ✓        --                      [VALID]              │
│ grok       ✓        grok-1                  [VALID]              │
└───────────────────────────────────────────────────────────────────┘
```

With `--models`:
```
OPENAI MODELS:
  - gpt-4.1-mini
  - gpt-4o
  - gpt-4-0125-preview
  - gpt-3.5-turbo

ANTHROPIC MODELS:
  - claude-3-5-sonnet-20241022
  - claude-3-5-haiku-20241022
  - claude-3-opus-20240229
  ...
```

Available options:
- `-c, --check`: Check API keys for all configured providers
- `--models`: List available models for each provider

#### Testing a Provider

The `test` command allows you to test a specific provider:

```bash
# Test the default OpenAI model with a simple prompt
umcp test openai

# Test a specific model (--model) with a custom prompt (--prompt)
umcp test anthropic --model claude-3-5-haiku-20241022 --prompt "Write a short poem about coding."

# Test Gemini with a different prompt
umcp test gemini --prompt "What are three interesting AI research papers from 2024?"
```

Example output:
```
Testing provider 'anthropic'...

Provider: anthropic
Model: claude-3-5-haiku-20241022
Prompt: Write a short poem about coding.

❯ Response:
Code flows like water,
Logic cascades through the mind—
Bugs bloom like flowers.

Tokens: 13 input, 19 output
Cost: $0.00006
Response time: 0.82s
```

Available options:
- `--model`: Model ID to test (defaults to the provider's default)
- `--prompt`: Prompt text to send (default: "Hello, world!")

### ⚡ Direct Text Generation

The `complete` command lets you generate text directly from the CLI:

```bash
# Generate text with default provider (OpenAI) using a prompt (--prompt)
umcp complete --prompt "Write a concise explanation of quantum computing."

# Specify a provider (--provider) and model (--model)
umcp complete --provider anthropic --model claude-3-5-sonnet-20241022 --prompt "What are the key differences between Rust and Go?"

# Use a system prompt (--system)
umcp complete --provider openai --model gpt-4o --system "You are an expert programmer..." --prompt "Explain dependency injection."

# Stream the response token by token (-s)
umcp complete --provider openai --prompt "Count from 1 to 10." -s

# Adjust temperature (--temperature) and token limit (--max-tokens)
umcp complete --provider gemini --temperature 1.2 --max-tokens 250 --prompt "Generate a creative sci-fi story opening."

# Read prompt from stdin (no --prompt needed)
echo "Tell me about space exploration." | umcp complete
```

Example output:
```
Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, unlike classical bits (0 or 1). This quantum superposition, along with entanglement, allows quantum computers to process vast amounts of information in parallel, potentially solving certain complex problems exponentially faster than classical computers. Applications include cryptography, materials science, and optimization problems.

Tokens: 13 input, 72 output
Cost: $0.00006
Response time: 0.37s
```

Available options:
- `--provider`: Provider to use (default: openai)
- `--model`: Model ID (defaults to provider's default)
- `--prompt`: Prompt text (reads from stdin if not provided)
- `--temperature`: Sampling temperature (0.0-2.0, default: 0.7)
- `--max-tokens`: Maximum tokens to generate
- `--system`: System prompt for providers that support it
- `-s, --stream`: Stream the response token by token

### 💾 Cache Management

The `cache` command allows you to view or clear the request cache:

```bash
# Show cache status (default action)
umcp cache

# Explicitly show status (no shortcut for --status)
umcp cache --status

# Clear the cache (no shortcut for --clear, with confirmation prompt)
umcp cache --clear

# Show stats and clear the cache in one command
umcp cache --status --clear
```

Example output:
```
Cache Status:
  Backend: memory
  Enabled: True
  Items: 127
  Hit rate: 73.2%
  Estimated savings: $1.47
```

Available options:
- `--status`: Show cache status (enabled by default if no other flag)
- `--clear`: Clear the cache (will prompt for confirmation)

### 📊 Benchmarking

The `benchmark` command lets you compare performance and cost across providers:

```bash
# Run default benchmark (3 runs per provider)
umcp benchmark

# Benchmark only specific providers
umcp benchmark --providers openai,anthropic

# Benchmark with specific models
umcp benchmark --providers openai,anthropic --models gpt-4o,claude-3.5-sonnet

# Use a custom prompt and more runs (-r)
umcp benchmark --prompt "Explain the process of photosynthesis in detail." -r 5
```

Example output:
```
┌─ Benchmark Results ───────────────────────────────────────────────────────┐
│ Provider    Model               Avg Time   Tokens    Cost      Tokens/sec │
├──────────────────────────────────────────────────────────────────────────┤
│ openai      gpt-4.1-mini        0.47s      76 / 213  $0.00023  454        │
│ anthropic   claude-3-5-haiku    0.52s      76 / 186  $0.00012  358        │
│ gemini      gemini-2.0-pro      0.64s      76 / 201  $0.00010  314        │
│ deepseek    deepseek-chat       0.71s      76 / 195  $0.00006  275        │
└──────────────────────────────────────────────────────────────────────────┘
```

Available options:
- `--providers`: List of providers to benchmark (default: all configured)
- `--models`: Model IDs to benchmark (defaults to default model of each provider)
- `--prompt`: Prompt text to use (default: built-in benchmark prompt)
- `-r, --runs`: Number of runs per provider/model (default: 3)

### 🧰 Tool Management

The `tools` command lists available tools, optionally filtered by category:

```bash
# List all tools
umcp tools

# List tools in a specific category
umcp tools --category document

# Show related example scripts
umcp tools --examples
```

Example output:
```
┌─ Ultimate MCP Server Tools ─────────────────────────────────────────┐
│ Category    Tool                           Example Script            │
├──────────────────────────────────────────────────────────────────────┤
│ completion  generate_completion            simple_completion_demo.py │
│ completion  stream_completion              simple_completion_demo.py │
│ completion  chat_completion                claude_integration_demo.py│
│ document    summarize_document             document_processing.py    │
│ document    chunk_document                 document_processing.py    │
│ extraction  extract_json                   advanced_extraction_demo.py│
│ filesystem  read_file                      filesystem_operations_demo.py│
└──────────────────────────────────────────────────────────────────────┘

Tip: Run examples using the command:
  umcp examples <example_name>
```

Available options:
- `--category`: Filter tools by category
- `--examples`: Show example scripts alongside tools

### 📚 Example Management

The `examples` command lets you list and run example scripts:

```bash
# List all example scripts (default action)
umcp examples

# Explicitly list example scripts (-l)
umcp examples -l

# Run a specific example
umcp examples rag_example.py

# Can also run by just the name without extension
umcp examples rag_example
```

Example output when listing:
```
┌─ Ultimate MCP Server Example Scripts ─────────────────────────────────┐
│ Category             Example Script                                   │
├────────────────────────────────────────────────────────────────────────┤
│ text-generation      simple_completion_demo.py                        │
│ text-generation      claude_integration_demo.py                       │
│ document-processing  document_processing.py                           │
│ search-and-retrieval rag_example.py                                   │
│ browser-automation   browser_automation_demo.py                       │
└────────────────────────────────────────────────────────────────────────┘

Run an example:
  umcp examples <example_name>
```

When running an example:
```
Running example: rag_example.py

Creating vector knowledge base 'demo_kb'...
Adding sample documents...
Retrieving context for query: "What are the benefits of clean energy?"
Generated response:
Based on the retrieved context, clean energy offers several benefits:
...
```

Available options:
- `-l, --list`: List example scripts only
- `--category`: Filter examples by category

### 🔎 Getting Help

Every command has detailed help available:

```bash
# General help
umcp --help

# Help for a specific command
umcp run --help
umcp providers --help
umcp complete --help
```

Example output:
```
Usage: umcp [OPTIONS] COMMAND [ARGS]...

  Ultimate MCP Server: Multi-provider LLM management server
  Unified CLI to run your server, manage providers, and more.

Options:
  --version, -v                   Show the application version and exit.
  --help                          Show this message and exit.

Commands:
  run          Run the Ultimate MCP Server
  providers    List Available Providers
  test         Test a Specific Provider
  complete     Generate Text Completion
  cache        Cache Management
  benchmark    Benchmark Providers
  tools        List Available Tools
  examples     Run or List Example Scripts
```

Command-specific help:
```
Usage: umcp run [OPTIONS]

  Run the Ultimate MCP Server

  Start the server with optional overrides.

  Examples:
    umcp run -h 0.0.0.0 -p 8000 -w 4 -t sse
    umcp run -d

Options:
  -h, --host TEXT                 Host or IP address to bind the server to.
                                  Defaults from config.
  -p, --port INTEGER              Port to listen on. Defaults from config.
  -w, --workers INTEGER           Number of worker processes to spawn.
                                  Defaults from config.
  -t, --transport-mode [shttp|sse|stdio]
                                  Transport mode for server communication (-t
                                  shortcut). Options: 'shttp' (streamable-http, 
                                  recommended), 'sse', or 'stdio'.
  -d, --debug                     Enable debug logging for detailed output (-d
                                  shortcut).
  --include-tools TEXT            List of tool names to include when running
                                  the server.
  --exclude-tools TEXT            List of tool names to exclude when running
                                  the server.
  --help                          Show this message and exit.
```

---

## 🧪 Usage Examples

This section provides Python examples demonstrating how an MCP client (like an application using `mcp-client` or an agent like Claude) would interact with the tools provided by a running Ultimate MCP Server instance.

*Note: These examples assume you have `mcp-client` installed (`pip install mcp-client`) and the Ultimate MCP Server is running at `http://localhost:8013`.*

*(The detailed code blocks from the original input are preserved below for completeness)*

### Basic Completion

```python
import asyncio
from mcp.client import Client

async def basic_completion_example():
    client = Client("http://localhost:8013")
    response = await client.tools.completion(
        prompt="Write a short poem about a robot learning to dream.",
        provider="openai",
        model="gpt-4.1-mini",
        max_tokens=100,
        temperature=0.7
    )
    if response["success"]:
        print(f"Completion: {response['completion']}")
        print(f"Cost: ${response['cost']:.6f}")
    else:
        print(f"Error: {response['error']}")
    await client.close()

# if __name__ == "__main__": asyncio.run(basic_completion_example())
```

### Claude Using Ultimate MCP Server for Document Analysis (Delegation)

```python
import asyncio
from mcp.client import Client

async def document_analysis_example():
    # Assume Claude identifies a large document needing processing
    client = Client("http://localhost:8013")
    document = "... large document content ..." * 100 # Placeholder for large content

    print("Delegating document chunking...")
    # Step 1: Claude delegates document chunking (often a local, non-LLM task on server)
    chunks_response = await client.tools.chunk_document(
        document=document,
        chunk_size=1000, # Target tokens per chunk
        overlap=100,     # Token overlap
        method="semantic" # Use semantic chunking if available
    )
    if not chunks_response["success"]:
        print(f"Chunking failed: {chunks_response['error']}")
        await client.close()
        return

    print(f"Document divided into {chunks_response['chunk_count']} chunks.")

    # Step 2: Claude delegates summarization of each chunk to a cheaper model
    summaries = []
    total_cost = 0.0
    print("Delegating chunk summarization to gemini-2.0-flash-lite...")
    for i, chunk in enumerate(chunks_response["chunks"]):
        # Use Gemini Flash (much cheaper than Claude or GPT-4o) via the server
        summary_response = await client.tools.summarize_document(
            document=chunk,
            provider="gemini", # Explicitly delegate to Gemini via server
            model="gemini-2.0-flash-lite",
            format="paragraph",
            max_length=150 # Request a concise summary
        )
        if summary_response["success"]:
            summaries.append(summary_response["summary"])
            cost = summary_response.get("cost", 0.0)
            total_cost += cost
            print(f"  Processed chunk {i+1}/{chunks_response['chunk_count']} summary. Cost: ${cost:.6f}")
        else:
            print(f"  Chunk {i+1} summarization failed: {summary_response['error']}")

    print("\nDelegating entity extraction to gpt-4.1-mini...")
    # Step 3: Claude delegates entity extraction for the whole document to another cheap model
    entities_response = await client.tools.extract_entities(
        document=document, # Process the original document
        entity_types=["person", "organization", "location", "date", "product"],
        provider="openai", # Delegate to OpenAI's cheaper model
        model="gpt-4.1-mini"
    )

    if entities_response["success"]:
        cost = entities_response.get("cost", 0.0)
        total_cost += cost
        print(f"Extracted entities. Cost: ${cost:.6f}")
        extracted_entities = entities_response['entities']
        # Claude would now process these summaries and entities using its advanced capabilities
        print(f"\nClaude can now use {len(summaries)} summaries and {len(extracted_entities)} entity groups.")
    else:
        print(f"Entity extraction failed: {entities_response['error']}")

    print(f"\nTotal estimated delegation cost for sub-tasks: ${total_cost:.6f}")

    # Claude might perform final synthesis using the collected results
    final_synthesis_prompt = f"""
Synthesize the key information from the following summaries and entities extracted from a large document.
Focus on the main topics, key people involved, and significant events mentioned.

Summaries:
{' '.join(summaries)}

Entities:
{extracted_entities}

Provide a concise final report.
"""
    # This final step would likely use Claude itself (not shown here)

    await client.close()

# if __name__ == "__main__": asyncio.run(document_analysis_example())
```

### Browser Automation for Research

```python
import asyncio
from mcp.client import Client

async def browser_research_example():
    client = Client("http://localhost:8013")
    print("Starting browser-based research task...")
    # This tool likely orchestrates multiple browser actions (search, navigate, scrape)
    # and uses an LLM (specified or default) for synthesis.
    result = await client.tools.research_and_synthesize_report(
        topic="Latest advances in AI-powered drug discovery using graph neural networks",
        instructions={
            "search_query": "graph neural networks drug discovery 2024 research",
            "search_engines": ["google", "duckduckgo"], # Use multiple search engines
            "urls_to_include": ["nature.com", "sciencemag.org", "arxiv.org", "pubmed.ncbi.nlm.nih.gov"], # Prioritize these domains
            "max_urls_to_process": 7, # Limit the number of pages to visit/scrape
            "min_content_length": 500, # Ignore pages with very little content
            "focus_areas": ["novel molecular structures", "binding affinity prediction", "clinical trial results"], # Guide the synthesis
            "report_format": "markdown", # Desired output format
            "report_length": "detailed", # comprehensive, detailed, summary
            "llm_model": "anthropic/claude-3-5-sonnet-20241022" # Specify LLM for synthesis
        }
    )

    if result["success"]:
        print("\nResearch report generated successfully!")
        print(f"Processed {len(result.get('extracted_data', []))} sources.")
        print(f"Total processing time: {result.get('processing_time', 'N/A'):.2f}s")
        print(f"Estimated cost: ${result.get('total_cost', 0.0):.6f}") # Includes LLM synthesis cost
        print("\n--- Research Report ---")
        print(result['report'])
        print("-----------------------")
    else:
        print(f"\nBrowser research failed: {result.get('error', 'Unknown error')}")
        if 'details' in result: print(f"Details: {result['details']}")

    await client.close()

# if __name__ == "__main__": asyncio.run(browser_research_example())
```

### Cognitive Memory System Usage

```python
import asyncio
from mcp.client import Client
import uuid

async def cognitive_memory_example():
    client = Client("http://localhost:8013")
    # Generate a unique ID for this session/workflow if not provided
    workflow_id = str(uuid.uuid4())
    print(f"Using Workflow ID: {workflow_id}")

    print("\nCreating a workflow context...")
    # Create a workflow context to group related memories and actions
    workflow_response = await client.tools.create_workflow(
        workflow_id=workflow_id,
        title="Quantum Computing Investment Analysis",
        description="Analyzing the impact of quantum computing on financial markets.",
        goal="Identify potential investment opportunities or risks."
    )
    if not workflow_response["success"]: print(f"Error creating workflow: {workflow_response['error']}")

    print("\nRecording an agent action...")
    # Record the start of a research action
    action_response = await client.tools.record_action_start(
        workflow_id=workflow_id,
        action_type="research",
        title="Initial literature review on quantum algorithms in finance",
        reasoning="Need to understand the current state-of-the-art before assessing impact."
    )
    action_id = action_response.get("action_id") if action_response["success"] else None
    if not action_id: print(f"Error starting action: {action_response['error']}")

    print("\nStoring facts in semantic memory...")
    # Store some key facts discovered during research
    memory1 = await client.tools.store_memory(
        workflow_id=workflow_id,
        content="Shor's algorithm can break RSA encryption, posing a threat to current financial security.",
        memory_type="fact", memory_level="semantic", importance=9.0,
        tags=["quantum_algorithm", "cryptography", "risk", "shor"]
    )
    memory2 = await client.tools.store_memory(
        workflow_id=workflow_id,
        content="Quantum annealing (e.g., D-Wave) shows promise for portfolio optimization problems.",
        memory_type="fact", memory_level="semantic", importance=7.5,
        tags=["quantum_computing", "finance", "optimization", "annealing"]
    )
    if memory1["success"]: print(f"Stored memory ID: {memory1['memory_id']}")
    if memory2["success"]: print(f"Stored memory ID: {memory2['memory_id']}")

    print("\nStoring an observation (episodic memory)...")
    # Store an observation from a specific event/document
    obs_memory = await client.tools.store_memory(
        workflow_id=workflow_id,
        content="Read Nature article (doi:...) suggesting experimental quantum advantage in a specific financial modeling task.",
        memory_type="observation", memory_level="episodic", importance=8.0,
        source="Nature Article XYZ", timestamp="2024-07-20T10:00:00Z", # Example timestamp
        tags=["research_finding", "publication", "finance_modeling"]
    )
    if obs_memory["success"]: print(f"Stored episodic memory ID: {obs_memory['memory_id']}")

    print("\nSearching for relevant memories...")
    # Search for memories related to financial risks
    search_results = await client.tools.hybrid_search_memories(
        workflow_id=workflow_id,
        query="What are the financial risks associated with quantum computing?",
        top_k=5, memory_type="fact", # Search for facts first
        semantic_weight=0.7, keyword_weight=0.3 # Example weighting for hybrid search
    )
    if search_results["success"]:
        print(f"Found {len(search_results['results'])} relevant memories:")
        for res in search_results["results"]:
            print(f"  - Score: {res['score']:.4f}, ID: {res['memory_id']}, Content: {res['content'][:80]}...")
    else:
        print(f"Memory search failed: {search_results['error']}")

    print("\nGenerating a reflection based on stored memories...")
    # Generate insights or reflections based on the accumulated knowledge in the workflow
    reflection_response = await client.tools.generate_reflection(
        workflow_id=workflow_id,
        reflection_type="summary_and_next_steps", # e.g., insights, risks, opportunities
        context_query="Summarize the key findings about quantum finance impact and suggest next research actions."
    )
    if reflection_response["success"]:
        print("Generated Reflection:")
        print(reflection_response["reflection"])
    else:
        print(f"Reflection generation failed: {reflection_response['error']}")

    # Mark the action as completed (assuming research phase is done)
    if action_id:
        print("\nCompleting the research action...")
        await client.tools.record_action_end(
            workflow_id=workflow_id, action_id=action_id, status="completed",
            outcome="Gathered initial understanding of quantum algorithms in finance and associated risks."
        )

    await client.close()

# if __name__ == "__main__": asyncio.run(cognitive_memory_example())
```

### Excel Spreadsheet Automation

```python
import asyncio
from mcp.client import Client
import os

async def excel_automation_example():
    client = Client("http://localhost:8013")
    output_dir = "excel_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "financial_model.xlsx")

    print(f"Requesting creation of Excel financial model at {output_path}...")
    # Example: Create a financial model using natural language instructions
    create_result = await client.tools.excel_execute(
        instruction="Create a simple 3-year financial projection.\n"
                   "Sheet name: 'Projections'.\n"
                   "Columns: Year 1, Year 2, Year 3.\n"
                   "Rows: Revenue, COGS, Gross Profit, Operating Expenses, Net Income.\n"
                   "Data: Start Revenue at $100,000, grows 20% annually.\n"
                   "COGS is 40% of Revenue.\n"
                   "Operating Expenses start at $30,000, grow 10% annually.\n"
                   "Calculate Gross Profit (Revenue - COGS) and Net Income (Gross Profit - OpEx).\n"
                   "Format currency as $#,##0. Apply bold headers and add a light blue fill to the header row.",
        file_path=output_path, # Server needs write access to this path/directory if relative
        operation_type="create", # create, modify, analyze, format
        # sheet_name="Projections", # Can specify sheet if modifying
        # cell_range="A1:D6", # Can specify range
        show_excel=False # Run Excel in the background (if applicable on the server)
    )

    if create_result["success"]:
        print(f"Excel creation successful: {create_result['message']}")
        print(f"File saved at: {create_result.get('output_file_path', output_path)}") # Confirm output path

        # Example: Modify the created file - add a chart
        print("\nRequesting modification: Add a Revenue chart...")
        modify_result = await client.tools.excel_execute(
            instruction="Add a column chart showing Revenue for Year 1, Year 2, Year 3. "
                       "Place it below the table. Title the chart 'Revenue Projection'.",
            file_path=output_path, # Use the previously created file
            operation_type="modify",
            sheet_name="Projections" # Specify the sheet to modify
        )
        if modify_result["success"]:
             print(f"Excel modification successful: {modify_result['message']}")
             print(f"File updated at: {modify_result.get('output_file_path', output_path)}")
        else:
             print(f"Excel modification failed: {modify_result['error']}")

    else:
        print(f"Excel creation failed: {create_result['error']}")
        if 'details' in create_result: print(f"Details: {create_result['details']}")

    # Example: Analyze formulas (if the tool supports it)
    # analysis_result = await client.tools.excel_analyze_formulas(...)

    await client.close()

# if __name__ == "__main__": asyncio.run(excel_automation_example())
```

### Multi-Provider Comparison

```python
import asyncio
from mcp.client import Client

async def multi_provider_completion_example():
    client = Client("http://localhost:8013")
    prompt = "Explain the concept of 'Chain of Thought' prompting for Large Language Models."

    print(f"Requesting completions for prompt: '{prompt}' from multiple providers...")
    # Request the same prompt from different models/providers
    multi_response = await client.tools.multi_completion(
        prompt=prompt,
        providers=[
            {"provider": "openai", "model": "gpt-4.1-mini", "temperature": 0.5},
            {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "temperature": 0.5},
            {"provider": "gemini", "model": "gemini-2.0-pro", "temperature": 0.5},
            # {"provider": "deepseek", "model": "deepseek-chat", "temperature": 0.5}, # Add others if configured
        ],
        # Common parameters applied to all if not specified per provider
        max_tokens=300
    )

    if multi_response["success"]:
        print("\n--- Multi-completion Results ---")
        total_cost = multi_response.get("total_cost", 0.0)
        print(f"Total Estimated Cost: ${total_cost:.6f}\n")

        for provider_key, result in multi_response["results"].items():
            print(f"--- Provider: {provider_key} ---")
            if result["success"]:
                print(f"  Model: {result.get('model', 'N/A')}")
                print(f"  Cost: ${result.get('cost', 0.0):.6f}")
                print(f"  Tokens: Input={result.get('input_tokens', 'N/A')}, Output={result.get('output_tokens', 'N/A')}")
                print(f"  Completion:\n{result['completion']}\n")
            else:
                print(f"  Error: {result['error']}\n")
        print("------------------------------")
        # An agent could now analyze these responses for consistency, detail, accuracy etc.
    else:
        print(f"\nMulti-completion request failed: {multi_response['error']}")

    await client.close()

# if __name__ == "__main__": asyncio.run(multi_provider_completion_example())
```

### Cost-Optimized Workflow Execution

```python
import asyncio
from mcp.client import Client

async def optimized_workflow_example():
    client = Client("http://localhost:8013")
    # Example document to process through the workflow
    document_content = """
    Project Alpha Report - Q3 2024
    Lead: Dr. Evelyn Reed (e.reed@example.com)
    Status: On Track
    Budget: $50,000 remaining. Spent $25,000 this quarter.
    Key Findings: Successful prototype development (v0.8). User testing feedback positive.
    Next Steps: Finalize documentation, prepare for Q4 deployment. Target date: 2024-11-15.
    Risks: Potential delay due to supplier issues for component X. Mitigation plan in place.
    """

    print("Defining a multi-stage workflow...")
    # Define a workflow with stages, dependencies, and provider preferences
    # Use ${stage_id.output_key} to pass outputs between stages
    workflow_definition = [
        {
            "stage_id": "summarize_report",
            "tool_name": "summarize_document",
            "params": {
                "document": document_content,
                "format": "bullet_points",
                "max_length": 100,
                # Let the server choose a cost-effective model for summarization
                "provider_preference": "cost", # 'cost', 'quality', 'speed', or specific like 'openai/gpt-4.1-mini'
            }
            # No 'depends_on', runs first
            # Default output key is 'summary' for this tool, access via ${summarize_report.summary}
        },
        {
            "stage_id": "extract_key_info",
            "tool_name": "extract_json", # Use JSON extraction for structured data
            "params": {
                "document": document_content,
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "project_lead": {"type": "string"},
                        "lead_email": {"type": "string", "format": "email"},
                        "status": {"type": "string"},
                        "budget_remaining": {"type": "string"},
                        "next_milestone_date": {"type": "string", "format": "date"}
                    },
                    "required": ["project_lead", "status", "next_milestone_date"]
                },
                # Prefer a model known for good structured data extraction, balancing cost
                "provider_preference": "quality", # Prioritize quality for extraction
                "preferred_models": ["openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022"] # Suggest specific models
            }
        },
        {
            "stage_id": "generate_follow_up_questions",
            "tool_name": "generate_qa", # Assuming a tool that generates questions
            "depends_on": ["summarize_report"], # Needs the summary first
            "params": {
                # Use the summary from the first stage as input
                "document": "${summarize_report.summary}",
                "num_questions": 3,
                "provider_preference": "speed" # Use a fast model for question generation
            }
            # Default output key 'qa_pairs', access via ${generate_follow_up_questions.qa_pairs}
        }
    ]

    print("Executing the optimized workflow...")
    # Execute the workflow - the server handles dependencies and model selection
    results = await client.tools.execute_optimized_workflow(
        workflow=workflow_definition
        # Can also pass initial documents if workflow steps reference 'original_document'
        # documents = {"report.txt": document_content}
    )

    if results["success"]:
        print("\nWorkflow executed successfully!")
        print(f"  Total processing time: {results.get('processing_time', 'N/A'):.2f}s")
        print(f"  Total estimated cost: ${results.get('total_cost', 0.0):.6f}\n")

        print("--- Stage Outputs ---")
        for stage_id, output in results.get("stage_outputs", {}).items():
            print(f"Stage: {stage_id}")
            if output["success"]:
                print(f"  Provider/Model Used: {output.get('provider', 'N/A')}/{output.get('model', 'N/A')}")
                print(f"  Cost: ${output.get('cost', 0.0):.6f}")
                print(f"  Output: {output.get('result', 'N/A')}") # Access the primary result
                # You might access specific keys like output.get('result', {}).get('summary') etc.
            else:
                print(f"  Error: {output.get('error', 'Unknown error')}")
            print("-" * 20)

    else:
        print(f"\nWorkflow execution failed: {results.get('error', 'Unknown error')}")
        if 'details' in results: print(f"Details: {results['details']}")

    await client.close()

# if __name__ == "__main__": asyncio.run(optimized_workflow_example())
```

### Entity Relation Graph Example

```python
import asyncio
from mcp.client import Client
# import networkx as nx # To process the graph data if needed
# import matplotlib.pyplot as plt # To visualize

async def entity_graph_example():
    client = Client("http://localhost:8013")
    document_text = """
    Meta Platforms, Inc., led by CEO Mark Zuckerberg, announced a partnership with IBM
    on developing new AI hardware accelerators. The collaboration aims to challenge Nvidia's dominance.
    IBM, headquartered in Armonk, New York, brings its deep expertise in semiconductor design.
    The project, codenamed 'Synergy', is expected to yield results by late 2025.
    """

    print("Extracting entity relationships from text...")
    # Request extraction of entities and their relationships
    entity_graph_response = await client.tools.extract_entity_relations(
        document=document_text,
        entity_types=["organization", "person", "location", "date", "project"], # Specify desired entity types
        relationship_types=["led_by", "partnership_with", "aims_to_challenge", "headquartered_in", "expected_by"], # Specify relationship types
        # Optional parameters:
        # provider_preference="quality", # Choose model strategy
        # llm_model="anthropic/claude-3-5-sonnet-20241022", # Suggest a specific model
        include_visualization=False # Set True to request image data if tool supports it
    )

    if entity_graph_response["success"]:
        print("Entity relationship extraction successful.")
        print(f"Estimated Cost: ${entity_graph_response.get('cost', 0.0):.6f}")

        # The graph data might be in various formats (e.g., node-link list, adjacency list)
        graph_data = entity_graph_response.get("graph_data")
        print("\n--- Graph Data (Nodes & Edges) ---")
        print(graph_data)
        print("------------------------------------")

        # Example: Query the extracted graph using another tool or LLM call
        # (Assuming a separate query tool or using a general completion tool)
        print("\nQuerying the extracted graph (example)...")
        query_prompt = f"""
Based on the following graph data representing relationships extracted from a text:
{graph_data}

Answer the question: Who is the CEO of Meta Platforms, Inc.?
"""
        query_response = await client.tools.completion(
             prompt=query_prompt, provider="openai", model="gpt-4.1-mini", max_tokens=50
        )
        if query_response["success"]:
             print(f"Graph Query Answer: {query_response['completion']}")
        else:
             print(f"Graph query failed: {query_response['error']}")


    else:
        print(f"Entity relationship extraction failed: {entity_graph_response.get('error', 'Unknown error')}")

    await client.close()

# if __name__ == "__main__": asyncio.run(entity_graph_example())
```

### Document Chunking

```python
import asyncio
from mcp.client import Client

async def document_chunking_example():
    client = Client("http://localhost:8013")
    large_document = """
    This is the first paragraph of a potentially very long document. It discusses various concepts.
    The second paragraph continues the discussion, adding more details and nuances. Proper chunking
    is crucial for processing large texts with Large Language Models, especially those with limited
    context windows. Different strategies exist, such as fixed token size, sentence splitting,
    or more advanced semantic chunking that tries to keep related ideas together. Overlap between
    chunks helps maintain context across boundaries. This paragraph is intentionally made longer
    to demonstrate how chunking might split it. It keeps going and going, describing the benefits
    of effective text splitting for downstream tasks like summarization, question answering, and
    retrieval-augmented generation (RAG). The goal is to create manageable pieces of text that
    still retain coherence. Semantic chunking often uses embedding models to find natural breakpoints
    in the text's meaning, potentially leading to better results than simple fixed-size chunks.
    The final sentence of this example paragraph.
    """ * 5 # Make it a bit longer for demonstration

    print("Requesting document chunking...")
    # Request chunking using a specific method and size
    chunking_response = await client.tools.chunk_document(
        document=large_document,
        chunk_size=100,     # Target size in tokens (approximate)
        overlap=20,         # Token overlap between consecutive chunks
        method="semantic"   # Options: "token", "sentence", "semantic", "structural" (if available)
    )

    if chunking_response["success"]:
        print(f"Document successfully divided into {chunking_response['chunk_count']} chunks.")
        print(f"Method Used: {chunking_response.get('method_used', 'N/A')}") # Confirm method if returned

        print("\n--- Example Chunks ---")
        for i, chunk in enumerate(chunking_response['chunks'][:3]): # Show first 3 chunks
            print(f"Chunk {i+1} (Length: {len(chunk)} chars):")
            print(f"'{chunk}'\n")
        if chunking_response['chunk_count'] > 3: print("...")
        print("----------------------")

        # These chunks can now be passed individually to other tools (e.g., summarize_document)
    else:
        print(f"Document chunking failed: {chunking_response['error']}")

    await client.close()

# if __name__ == "__main__": asyncio.run(document_chunking_example())
```

### Multi-Provider Completion (Duplicate of earlier example, kept for structure)

```python
import asyncio
from mcp.client import Client

async def multi_provider_completion_example():
    client = Client("http://localhost:8013")
    prompt = "What are the main benefits of using the Model Context Protocol (MCP)?"

    print(f"Requesting completions for prompt: '{prompt}' from multiple providers...")
    multi_response = await client.tools.multi_completion(
        prompt=prompt,
        providers=[
            {"provider": "openai", "model": "gpt-4.1-mini"},
            {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"},
            {"provider": "gemini", "model": "gemini-2.0-flash-lite"}
            # Add more configured providers as needed
        ],
        temperature=0.5,
        max_tokens=250
    )

    if multi_response["success"]:
        print("\n--- Multi-completion Results ---")
        total_cost = multi_response.get("total_cost", 0.0)
        print(f"Total Estimated Cost: ${total_cost:.6f}\n")
        for provider_key, result in multi_response["results"].items():
            print(f"--- Provider: {provider_key} ---")
            if result["success"]:
                print(f"  Model: {result.get('model', 'N/A')}")
                print(f"  Cost: ${result.get('cost', 0.0):.6f}")
                print(f"  Completion:\n{result['completion']}\n")
            else:
                print(f"  Error: {result['error']}\n")
        print("------------------------------")
    else:
        print(f"\nMulti-completion request failed: {multi_response['error']}")

    await client.close()

# if __name__ == "__main__": asyncio.run(multi_provider_completion_example())
```

### Structured Data Extraction (JSON)

```python
import asyncio
from mcp.client import Client
import json

async def json_extraction_example():
    client = Client("http://localhost:8013")
    text_with_data = """
    Meeting Minutes - Project Phoenix - 2024-07-21

    Attendees: Alice (Lead), Bob (Dev), Charlie (QA)
    Date: July 21, 2024
    Project ID: PX-001

    Discussion Points:
    - Reviewed user feedback from v1.1 testing. Mostly positive.
    - Identified performance bottleneck in data processing module. Bob to investigate. Assigned High priority.
    - QA cycle for v1.2 planned to start next Monday (2024-07-29). Charlie confirmed readiness.

    Action Items:
    1. Bob: Investigate performance issue. Due: 2024-07-26. Priority: High. Status: Open.
    2. Alice: Prepare v1.2 release notes. Due: 2024-07-28. Priority: Medium. Status: Open.
    """

    # Define the desired JSON structure (schema)
    desired_schema = {
        "type": "object",
        "properties": {
            "project_name": {"type": "string", "description": "Name of the project"},
            "meeting_date": {"type": "string", "format": "date", "description": "Date of the meeting"},
            "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of attendee names"},
            "action_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "assigned_to": {"type": "string"},
                        "due_date": {"type": "string", "format": "date"},
                        "priority": {"type": "string", "enum": ["Low", "Medium", "High"]},
                        "status": {"type": "string", "enum": ["Open", "In Progress", "Done"]}
                    },
                    "required": ["task", "assigned_to", "due_date", "priority", "status"]
                }
            }
        },
        "required": ["project_name", "meeting_date", "attendees", "action_items"]
    }

    print("Requesting JSON extraction based on schema...")
    # Request extraction using a model capable of following JSON schema instructions
    json_response = await client.tools.extract_json(
        document=text_with_data,
        json_schema=desired_schema,
        provider="openai", # OpenAI models are generally good at this
        model="gpt-4o", # Use a capable model like GPT-4o or Claude 3.5 Sonnet
        # provider_preference="quality" # Could also use preference
    )

    if json_response["success"]:
        print("JSON extraction successful.")
        print(f"Estimated Cost: ${json_response.get('cost', 0.0):.6f}")

        # The extracted data should conform to the schema
        extracted_json_data = json_response.get('json_data')
        print("\n--- Extracted JSON Data ---")
        # Pretty print the JSON
        print(json.dumps(extracted_json_data, indent=2))
        print("---------------------------")

        # Optionally, validate the output against the schema client-side (requires jsonschema library)
        # try:
        #     from jsonschema import validate
        #     validate(instance=extracted_json_data, schema=desired_schema)
        #     print("\nClient-side validation successful: Output matches schema.")
        # except ImportError:
        #     print("\n(Install jsonschema to perform client-side validation)")
        # except Exception as e:
        #     print(f"\nClient-side validation failed: {e}")

    else:
        print(f"JSON Extraction Error: {json_response.get('error', 'Unknown error')}")
        if 'details' in json_response: print(f"Details: {json_response['details']}")

    await client.close()

# if __name__ == "__main__": asyncio.run(json_extraction_example())
```

### Retrieval-Augmented Generation (RAG) Query

```python
import asyncio
from mcp.client import Client

async def rag_query_example():
    # This example assumes the Ultimate MCP Server has been configured with a RAG pipeline,
    # including a vector store/index containing relevant documents.
    client = Client("http://localhost:8013")
    query = "What are the latest treatment options for mitigating Alzheimer's disease according to recent studies?"

    print(f"Performing RAG query: '{query}'...")
    # Call the RAG tool, which handles retrieval and generation
    rag_response = await client.tools.rag_query( # Assuming the tool name is 'rag_query'
        query=query,
        # Optional parameters to control the RAG process:
        index_name="medical_research_papers", # Specify the index/collection to search
        top_k=3, # Retrieve top 3 most relevant documents/chunks
        # filter={"year": {"$gte": 2023}}, # Example filter (syntax depends on vector store)
        # generation_model={"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"}, # Specify generation model
        # instruction_prompt="Based on the provided context, answer the user's query concisely." # Customize generation prompt
    )

    if rag_response["success"]:
        print("\nRAG query successful.")
        print(f"Estimated Cost: ${rag_response.get('cost', 0.0):.6f}") # Includes retrieval + generation cost

        print("\n--- Generated Answer ---")
        print(rag_response.get('answer', 'No answer generated.'))
        print("------------------------")

        # The response might also include details about the retrieved sources
        retrieved_sources = rag_response.get('sources', [])
        if retrieved_sources:
            print("\n--- Retrieved Sources ---")
            for i, source in enumerate(retrieved_sources):
                print(f"Source {i+1}:")
                print(f"  ID: {source.get('id', 'N/A')}")
                print(f"  Score: {source.get('score', 'N/A'):.4f}")
                # Depending on RAG setup, might include metadata or text snippet
                print(f"  Content Snippet: {source.get('text', '')[:150]}...")
                print("-" * 15)
            print("-----------------------")
        else:
            print("\nNo sources information provided in the response.")

    else:
        print(f"\nRAG Query Error: {rag_response.get('error', 'Unknown error')}")
        if 'details' in rag_response: print(f"Details: {rag_response['details']}")

    await client.close()

# if __name__ == "__main__": asyncio.run(rag_query_example())
```

### Fused Search (Keyword + Semantic)

```python
import asyncio
from mcp.client import Client

async def fused_search_example():
    # This example assumes the server is configured with a hybrid search provider like Marqo.
    client = Client("http://localhost:8013")
    query = "impact of AI on software development productivity and code quality"

    print(f"Performing fused search for: '{query}'...")
    # Call the fused search tool
    fused_search_response = await client.tools.fused_search( # Assuming tool name is 'fused_search'
        query=query,
        # --- Parameters specific to the hybrid search backend (e.g., Marqo) ---
        index_name="tech_articles_index", # Specify the target index
        searchable_attributes=["title", "content"], # Fields to search within
        limit=5, # Number of results to return
        # Tunable weights for keyword vs. semantic relevance (example)
        hybrid_factors={"keyword_weight": 0.4, "semantic_weight": 0.6},
        # Optional filter string (syntax depends on backend)
        filter_string="publication_year >= 2023 AND source_type='journal'"
        # --------------------------------------------------------------------
    )

    if fused_search_response["success"]:
        print("\nFused search successful.")
        results = fused_search_response.get("results", [])
        print(f"Found {len(results)} hits.")

        if results:
            print("\n--- Search Results ---")
            for i, hit in enumerate(results):
                print(f"Result {i+1}:")
                # Fields depend on Marqo index structure and what's returned
                print(f"  ID: {hit.get('_id', 'N/A')}")
                print(f"  Score: {hit.get('_score', 'N/A'):.4f}") # Combined score
                print(f"  Title: {hit.get('title', 'N/A')}")
                print(f"  Content Snippet: {hit.get('content', '')[:150]}...")
                # Print highlight info if available
                highlights = hit.get('_highlights', {})
                if highlights: print(f"  Highlights: {highlights}")
                print("-" * 15)
            print("--------------------")
        else:
            print("No results found matching the criteria.")

    else:
        print(f"\nFused Search Error: {fused_search_response.get('error', 'Unknown error')}")
        if 'details' in fused_search_response: print(f"Details: {fused_search_response['details']}")

    await client.close()

# if __name__ == "__main__": asyncio.run(fused_search_example())
```

### Local Text Processing

```python
import asyncio
from mcp.client import Client

async def local_text_processing_example():
    client = Client("http://localhost:8013")
    # Example assumes a tool named 'process_local_text' exists on the server
    # that bundles various non-LLM text operations.
    raw_text = "  This text has   EXTRA whitespace,\n\nmultiple newlines, \t tabs, and needs Case Normalization.  "

    print("Requesting local text processing operations...")
    local_process_response = await client.tools.process_local_text(
        text=raw_text,
        operations=[
            {"action": "trim_whitespace"},       # Remove leading/trailing whitespace
            {"action": "normalize_whitespace"},  # Collapse multiple spaces/tabs to single space
            {"action": "remove_blank_lines"},    # Remove empty lines
            {"action": "lowercase"}              # Convert to lowercase
            # Other potential actions: uppercase, remove_punctuation, normalize_newlines, etc.
        ]
    )

    if local_process_response["success"]:
        print("\nLocal text processing successful.")
        print(f"Original Text:\n'{raw_text}'")
        print(f"\nProcessed Text:\n'{local_process_response['processed_text']}'")
        # Note: This operation should ideally have zero LLM cost.
        print(f"Cost: ${local_process_response.get('cost', 0.0):.6f}")
    else:
        print(f"\nLocal Text Processing Error: {local_process_response['error']}")

    await client.close()

# if __name__ == "__main__": asyncio.run(local_text_processing_example())
```

### Browser Automation Example: Getting Started and Basic Interaction

```python
import asyncio
from mcp.client import Client

async def browser_basic_interaction_example():
    # This example shows fundamental browser actions controlled by an agent
    client = Client("http://localhost:8013")
    print("--- Browser Automation: Basic Interaction ---")

    # 1. Initialize the browser (creates a browser instance on the server)
    print("\nInitializing browser (headless)...")
    # `headless=True` runs without a visible GUI window (common for automation)
    init_response = await client.tools.browser_init(headless=True, browser_type="chromium")
    if not init_response["success"]:
        print(f"Browser initialization failed: {init_response.get('error', 'Unknown error')}")
        await client.close()
        return
    print("Browser initialized successfully.")
    # Might return session ID if needed for subsequent calls, depends on tool design

    # 2. Navigate to a page
    target_url = "https://example.com/"
    print(f"\nNavigating to {target_url}...")
    # `wait_until` controls when navigation is considered complete
    nav_response = await client.tools.browser_navigate(
        url=target_url,
        wait_until="domcontentloaded" # Options: load, domcontentloaded, networkidle, commit
    )
    if nav_response["success"]:
        print(f"Navigation successful.")
        print(f"  Current URL: {nav_response.get('url', 'N/A')}")
        print(f"  Page Title: {nav_response.get('title', 'N/A')}")
        # The 'snapshot' gives the agent context about the page state (accessibility tree)
        # print(f"  Snapshot: {nav_response.get('snapshot', 'N/A')}")
    else:
        print(f"Navigation failed: {nav_response.get('error', 'Unknown error')}")
        # Attempt to close browser even if navigation failed
        await client.tools.browser_close()
        await client.close()
        return

    # 3. Extract text content using a CSS selector
    selector = "h1" # CSS selector for the main heading
    print(f"\nExtracting text from selector '{selector}'...")
    text_response = await client.tools.browser_get_text(selector=selector)
    if text_response["success"]:
        print(f"Extracted text: '{text_response.get('text', 'N/A')}'")
    else:
        print(f"Text extraction failed: {text_response.get('error', 'Unknown error')}")
        # Optionally check text_response['snapshot'] for page state at time of failure

    # 4. Take a screenshot (optional)
    print("\nTaking a screenshot...")
    screenshot_response = await client.tools.browser_screenshot(
        file_path="example_com_screenshot.png", # Path where server saves the file
        full_page=False, # Capture only the viewport
        image_format="png" # png or jpeg
    )
    if screenshot_response["success"]:
        print(f"Screenshot saved successfully on server at: {screenshot_response.get('saved_path', 'N/A')}")
        # Agent might use this path with a filesystem tool to retrieve the image if needed
    else:
         print(f"Screenshot failed: {screenshot_response.get('error', 'Unknown error')}")

    # 5. Close the browser session
    print("\nClosing the browser...")
    close_response = await client.tools.browser_close()
    if close_response["success"]:
        print("Browser closed successfully.")
    else:
        # Log error, but might happen if browser already crashed
        print(f"Browser close failed (might be expected if previous steps failed): {close_response.get('error', 'Unknown error')}")

    print("--- Browser Automation Example Complete ---")
    await client.close()

# if __name__ == "__main__": asyncio.run(browser_basic_interaction_example())

```

### Running a Model Tournament

```python
import asyncio
from mcp.client import Client
import json

async def model_tournament_example():
    client = Client("http://localhost:8013")
    # Define the task and prompt for the tournament
    task_prompt = "Write a Python function that takes a list of integers and returns a new list containing only the even numbers."
    # Optional: Provide ground truth for automated evaluation if the tool supports it
    ground_truth_code = """
def get_even_numbers(numbers):
    \"\"\"Returns a new list containing only the even numbers from the input list.\"\"\"
    return [num for num in numbers if num % 2 == 0]
"""

    print("Setting up and running a model tournament for code generation...")
    # Call the tournament tool
    tournament_response = await client.tools.run_model_tournament(
        task_type="code_generation", # Helps select appropriate evaluation metrics
        prompt=task_prompt,
        # List of models/providers to compete
        competitors=[
            {"provider": "openai", "model": "gpt-4.1-mini", "temperature": 0.2},
            {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "temperature": 0.2},
            {"provider": "deepseek", "model": "deepseek-coder", "temperature": 0.2}, # Specialized coder model
            {"provider": "gemini", "model": "gemini-2.0-pro", "temperature": 0.2},
        ],
        # Criteria for evaluating the generated code
        evaluation_criteria=["correctness", "efficiency", "readability", "docstring_quality"],
        # Provide ground truth if available for automated correctness checks
        ground_truth=ground_truth_code,
        # Optional: Specify an LLM to act as the judge for qualitative criteria
        evaluation_model={"provider": "anthropic", "model": "claude-3-5-opus-20240229"}, # Use a powerful model for judging
        num_rounds=1 # Run multiple rounds for stability if needed
    )

    if tournament_response["success"]:
        print("\n--- Model Tournament Results ---")
        print(f"Task Prompt: {task_prompt}")
        print(f"Total Estimated Cost: ${tournament_response.get('total_cost', 0.0):.6f}\n")

        # Display the ranking
        ranking = tournament_response.get("ranking", [])
        if ranking:
            print("Overall Ranking:")
            for i, result in enumerate(ranking):
                provider = result.get('provider', 'N/A')
                model = result.get('model', 'N/A')
                score = result.get('overall_score', 'N/A')
                cost = result.get('cost', 0.0)
                print(f"  {i+1}. {provider}/{model} - Score: {score:.2f}/10 - Cost: ${cost:.6f}")
        else:
            print("No ranking information available.")

        # Display detailed results for each competitor
        detailed_results = tournament_response.get("results", {})
        if detailed_results:
            print("\nDetailed Scores per Competitor:")
            for competitor_key, details in detailed_results.items():
                 print(f"  Competitor: {competitor_key}")
                 print(f"    Generated Code:\n```python\n{details.get('output', 'N/A')}\n```")
                 scores = details.get('scores', {})
                 if scores:
                     for criterion, score_value in scores.items():
                         print(f"    - {criterion}: {score_value}")
                 print("-" * 10)
        print("------------------------------")

    else:
        print(f"\nModel Tournament Failed: {tournament_response.get('error', 'Unknown error')}")
        if 'details' in tournament_response: print(f"Details: {tournament_response['details']}")

    await client.close()

# if __name__ == "__main__": asyncio.run(model_tournament_example())
```

### Meta Tools for Tool Discovery

```python
import asyncio
from mcp.client import Client
import json

async def meta_tools_example():
    client = Client("http://localhost:8013")
    print("--- Meta Tools Example ---")

    # 1. List all available tools
    print("\nFetching list of available tools...")
    # Assumes a tool named 'list_tools' provides this info
    list_tools_response = await client.tools.list_tools(include_schemas=False) # Set True for full schemas

    if list_tools_response["success"]:
        tools = list_tools_response.get("tools", {})
        print(f"Found {len(tools)} available tools:")
        for tool_name, tool_info in tools.items():
            description = tool_info.get('description', 'No description available.')
            print(f"  - {tool_name}: {description[:100]}...") # Print truncated description
    else:
        print(f"Failed to list tools: {list_tools_response.get('error', 'Unknown error')}")

    # 2. Get detailed information about a specific tool
    tool_to_inspect = "extract_json"
    print(f"\nFetching details for tool: '{tool_to_inspect}'...")
    # Assumes a tool like 'get_tool_info' or using list_tools with specific name/schema flag
    tool_info_response = await client.tools.list_tools(tool_names=[tool_to_inspect], include_schemas=True)

    if tool_info_response["success"] and tool_to_inspect in tool_info_response.get("tools", {}):
        tool_details = tool_info_response["tools"][tool_to_inspect]
        print(f"\nDetails for '{tool_to_inspect}':")
        print(f"  Description: {tool_details.get('description', 'N/A')}")
        # Print the parameter schema if available
        schema = tool_details.get('parameters', {}).get('json_schema', {})
        if schema:
            print(f"  Parameter Schema:\n{json.dumps(schema, indent=2)}")
        else:
            print("  Parameter Schema: Not available.")
    else:
        print(f"Failed to get info for tool '{tool_to_inspect}': {tool_info_response.get('error', 'Not found or error')}")

    # 3. Get tool recommendations for a task (if such a meta tool exists)
    task_description = "Read data from a PDF file, extract tables, and save them as CSV."
    print(f"\nGetting tool recommendations for task: '{task_description}'...")
    # Assumes a tool like 'get_tool_recommendations'
    recommendations_response = await client.tools.get_tool_recommendations(
        task=task_description,
        constraints={"priority": "accuracy", "max_cost_per_doc": 0.10} # Example constraints
    )

    if recommendations_response["success"]:
        print("Recommended Tool Workflow:")
        recommendations = recommendations_response.get("recommendations", [])
        if recommendations:
            for i, step in enumerate(recommendations):
                print(f"  Step {i+1}: Tool='{step.get('tool', 'N/A')}' - Reason: {step.get('reason', 'N/A')}")
        else:
            print("  No recommendations provided.")
    else:
         print(f"Failed to get recommendations: {recommendations_response.get('error', 'Unknown error')}")

    print("\n--- Meta Tools Example Complete ---")
    await client.close()

# if __name__ == "__main__": asyncio.run(meta_tools_example())
```

### Local Command-Line Text Processing (e.g., jq)

```python
import asyncio
from mcp.client import Client
import json

async def local_cli_tool_example():
    client = Client("http://localhost:8013")
    print("--- Local CLI Tool Example (jq) ---")

    # Example JSON data to be processed by jq
    json_input_data = json.dumps({
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com", "status": "active"},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "status": "inactive"},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com", "status": "active"}
        ],
        "metadata": {"timestamp": "2024-07-21T12:00:00Z"}
    })

    # Define the jq filter to apply
    # This filter selects active users and outputs their name and email
    jq_filter = '.users[] | select(.status=="active") | {name: .name, email: .email}'

    print(f"\nRunning jq with filter: '{jq_filter}' on input JSON...")
    # Call the server tool that wraps jq (e.g., 'run_jq')
    jq_result = await client.tools.run_jq(
        args_str=jq_filter, # Pass the filter as arguments (check tool spec how it expects filters)
        input_data=json_input_data, # Provide the JSON string as input
        # Additional options might be available depending on the tool wrapper:
        # e.g., output_format="json_lines" or "compact_json"
    )

    if jq_result["success"]:
        print("jq execution successful.")
        # stdout typically contains the result of the jq filter
        print("\n--- jq Output (stdout) ---")
        print(jq_result.get("stdout", "No output"))
        print("--------------------------")
        # stderr might contain warnings or errors from jq itself
        stderr_output = jq_result.get("stderr")
        if stderr_output:
            print("\n--- jq Stderr ---")
            print(stderr_output)
            print("-----------------")
        # This should have minimal or zero cost as it runs locally on the server
        print(f"\nCost: ${jq_result.get('cost', 0.0):.6f}")
    else:
        print(f"\njq Execution Error: {jq_result.get('error', 'Unknown error')}")
        print(f"Stderr: {jq_result.get('stderr', 'N/A')}")

    print("\n--- Local CLI Tool Example Complete ---")
    await client.close()

# if __name__ == "__main__": asyncio.run(local_cli_tool_example())
```

### Dynamic API Integration

```python
import asyncio
from mcp.client import Client
import json

async def dynamic_api_example():
    # This example assumes the server has tools like 'register_api', 'list_registered_apis',
    # 'call_dynamic_tool', and 'unregister_api'.
    client = Client("http://localhost:8013")
    print("--- Dynamic API Integration Example ---")

    # 1. Register an external API using its OpenAPI (Swagger) specification URL
    api_name_to_register = "public_cat_facts"
    openapi_spec_url = "https://catfact.ninja/docs/api-docs.json" # Example public API spec
    print(f"\nRegistering API '{api_name_to_register}' from {openapi_spec_url}...")

    register_response = await client.tools.register_api(
        api_name=api_name_to_register,
        openapi_url=openapi_spec_url,
        # Optional: Provide authentication details if needed (e.g., Bearer token, API Key)
        # authentication={"type": "bearer", "token": "your_api_token"},
        # Optional: Set default headers
        # default_headers={"X-Custom-Header": "value"},
        # Optional: Cache settings for API responses (if tool supports it)
        cache_ttl=300 # Cache responses for 5 minutes
    )

    if register_response["success"]:
        print(f"API '{api_name_to_register}' registered successfully.")
        print(f"  Registered {register_response.get('tools_count', 0)} new MCP tools derived from the API.")
        print(f"  Tools Registered: {register_response.get('tools_registered', [])}")
    else:
        print(f"API registration failed: {register_response.get('error', 'Unknown error')}")
        await client.close()
        return

    # 2. List currently registered dynamic APIs
    print("\nListing registered dynamic APIs...")
    list_apis_response = await client.tools.list_registered_apis()
    if list_apis_response["success"]:
        registered_apis = list_apis_response.get("apis", {})
        print(f"Currently registered APIs: {list(registered_apis.keys())}")
        # print(json.dumps(registered_apis, indent=2)) # Print full details
    else:
        print(f"Failed to list registered APIs: {list_apis_response.get('error', 'Unknown error')}")

    # 3. Call a dynamically created tool corresponding to an API endpoint
    # The tool name is typically derived from the API name and endpoint's operationId or path.
    # Check the 'tools_registered' list from step 1 or documentation for the exact name.
    # Let's assume the tool for GET /fact is 'public_cat_facts_getFact'
    dynamic_tool_name = "public_cat_facts_getFact" # Adjust based on actual registered name
    print(f"\nCalling dynamic tool '{dynamic_tool_name}'...")

    call_response = await client.tools.call_dynamic_tool(
        tool_name=dynamic_tool_name,
        # Provide inputs matching the API endpoint's parameters
        inputs={
            # Example query parameter for GET /fact (check API spec)
             "max_length": 100
        }
    )

    if call_response["success"]:
        print("Dynamic tool call successful.")
        # The result usually contains the API's response body and status code
        print(f"  Status Code: {call_response.get('status_code', 'N/A')}")
        print(f"  Response Body:\n{json.dumps(call_response.get('response_body', {}), indent=2)}")
    else:
        print(f"Dynamic tool call failed: {call_response.get('error', 'Unknown error')}")
        print(f"  Status Code: {call_response.get('status_code', 'N/A')}")
        print(f"  Response Body: {call_response.get('response_body', 'N/A')}")

    # 4. Unregister the API when no longer needed (optional cleanup)
    print(f"\nUnregistering API '{api_name_to_register}'...")
    unregister_response = await client.tools.unregister_api(api_name=api_name_to_register)
    if unregister_response["success"]:
        print(f"API unregistered successfully. Removed {unregister_response.get('tools_count', 0)} tools.")
    else:
        print(f"API unregistration failed: {unregister_response.get('error', 'Unknown error')}")

    print("\n--- Dynamic API Integration Example Complete ---")
    await client.close()

# if __name__ == "__main__": asyncio.run(dynamic_api_example())
```

### OCR Usage Example

```python
import asyncio
from mcp.client import Client
import os

async def ocr_example():
    # Requires 'ocr' extras installed: uv pip install -e ".[ocr]"
    # Also requires Tesseract OCR engine installed on the server host system.
    client = Client("http://localhost:8013")
    print("--- OCR Tool Example ---")

    # --- Create dummy files for testing ---
    # In a real scenario, these files would exist on a path accessible by the server.
    # Ensure the server process has permissions to read these files.
    dummy_files_dir = "ocr_test_files"
    os.makedirs(dummy_files_dir, exist_ok=True)
    dummy_pdf_path = os.path.join(dummy_files_dir, "dummy_document.pdf")
    dummy_image_path = os.path.join(dummy_files_dir, "dummy_image.png")

    # Create a simple dummy PDF (requires reportlab - pip install reportlab)
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        c = canvas.Canvas(dummy_pdf_path, pagesize=letter)
        c.drawString(100, 750, "This is page 1 of a dummy PDF.")
        c.drawString(100, 730, "It contains some text for OCR testing.")
        c.showPage()
        c.drawString(100, 750, "This is page 2.")
        c.save()
        print(f"Created dummy PDF: {dummy_pdf_path}")
    except ImportError:
        print("Could not create dummy PDF: reportlab not installed. Skipping PDF test.")
        dummy_pdf_path = None
    except Exception as e:
        print(f"Error creating dummy PDF: {e}. Skipping PDF test.")
        dummy_pdf_path = None

    # Create a simple dummy PNG image (requires Pillow - pip install Pillow)
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (400, 100), color = (255, 255, 255))
        d = ImageDraw.Draw(img)
        # Use a default font if possible, otherwise basic text
        try: font = ImageFont.truetype("arial.ttf", 15)
        except IOError: font = ImageFont.load_default()
        d.text((10,10), "Dummy Image Text for OCR\nLine 2 of text.", fill=(0,0,0), font=font)
        img.save(dummy_image_path)
        print(f"Created dummy Image: {dummy_image_path}")
    except ImportError:
        print("Could not create dummy Image: Pillow not installed. Skipping Image test.")
        dummy_image_path = None
    except Exception as e:
        print(f"Error creating dummy Image: {e}. Skipping Image test.")
        dummy_image_path = None
    # --- End of dummy file creation ---


    # 1. Extract text from the PDF using OCR and LLM correction
    if dummy_pdf_path:
        print(f"\nExtracting text from PDF: {dummy_pdf_path} (using hybrid method)...")
        pdf_text_result = await client.tools.extract_text_from_pdf(
            file_path=dummy_pdf_path, # Server needs access to this path
            extraction_method="hybrid", # Try direct extraction, fallback to OCR
            max_pages=2, # Limit pages to process
            reformat_as_markdown=True, # Request markdown formatting
            # Optional: Use an LLM to correct/improve the raw OCR text
            llm_correction_model={"provider": "openai", "model": "gpt-4.1-mini"}
        )
        if pdf_text_result["success"]:
            print("PDF text extraction successful.")
            print(f"  Method Used: {pdf_text_result.get('extraction_method_used', 'N/A')}")
            print(f"  Cost (incl. LLM correction): ${pdf_text_result.get('cost', 0.0):.6f}")
            print("\n--- Extracted PDF Text (Markdown) ---")
            print(pdf_text_result.get("text", "No text extracted."))
            print("-------------------------------------")
        else:
            print(f"PDF OCR failed: {pdf_text_result.get('error', 'Unknown error')}")
            if 'details' in pdf_text_result: print(f"Details: {pdf_text_result['details']}")
    else:
         print("\nSkipping PDF OCR test as dummy file could not be created.")


    # 2. Process the image file with OCR and preprocessing
    if dummy_image_path:
        print(f"\nProcessing image OCR: {dummy_image_path} with preprocessing...")
        image_text_result = await client.tools.process_image_ocr(
            image_path=dummy_image_path, # Server needs access to this path
            # Optional preprocessing steps (require OpenCV on server)
            preprocessing_options={
                "grayscale": True,
                # "threshold": "otsu", # e.g., otsu, adaptive
                # "denoise": True,
                # "deskew": True
            },
            ocr_language="eng" # Specify language(s) for Tesseract e.g., "eng+fra"
            # Optional LLM enhancement for image OCR results
            # llm_enhancement_model={"provider": "gemini", "model": "gemini-2.0-flash-lite"}
        )
        if image_text_result["success"]:
            print("Image OCR successful.")
            print(f"  Cost (incl. LLM enhancement): ${image_text_result.get('cost', 0.0):.6f}")
            print("\n--- Extracted Image Text ---")
            print(image_text_result.get("text", "No text extracted."))
            print("----------------------------")
        else:
            print(f"Image OCR failed: {image_text_result.get('error', 'Unknown error')}")
            if 'details' in image_text_result: print(f"Details: {image_text_result['details']}")
    else:
         print("\nSkipping Image OCR test as dummy file could not be created.")

    # --- Clean up dummy files ---
    # try:
    #     if dummy_pdf_path and os.path.exists(dummy_pdf_path): os.remove(dummy_pdf_path)
    #     if dummy_image_path and os.path.exists(dummy_image_path): os.remove(dummy_image_path)
    #     if os.path.exists(dummy_files_dir): os.rmdir(dummy_files_dir) # Only if empty
    # except Exception as e:
    #      print(f"\nError cleaning up dummy files: {e}")
    # --- End cleanup ---

    print("\n--- OCR Tool Example Complete ---")
    await client.close()

# if __name__ == "__main__": asyncio.run(ocr_example())
```

*(Note: Many examples involving file paths assume the server process has access to those paths. For Docker deployments, volume mapping is usually required.)*

---

## ✨ Autonomous Documentation Refiner

The Ultimate MCP Server includes a powerful feature for autonomously analyzing, testing, and refining the documentation of registered MCP tools. This feature, implemented in `ultimate/tools/docstring_refiner.py`, helps improve the usability and reliability of tools when invoked by Large Language Models (LLMs) like Claude.

### How It Works

The documentation refiner follows a methodical, iterative approach:

1.  **Agent Simulation**: Simulates how an LLM agent would interpret the current documentation (docstring, schema, examples) to identify potential ambiguities or missing information crucial for correct invocation.
2.  **Adaptive Test Generation**: Creates diverse test cases based on the tool's input schema (parameter types, constraints, required fields), simulation results, and failures from previous refinement iterations. Aims for good coverage.
3.  **Schema-Aware Testing**: Validates generated test inputs against the tool's schema *before* execution. Executes valid tests against the actual tool implementation within the server environment.
4.  **Ensemble Failure Analysis**: If a test fails (e.g., wrong output, error thrown), multiple LLMs analyze the failure in the context of the specific documentation version used for that test run to pinpoint the documentation's weaknesses.
5.  **Structured Improvement Proposals**: Based on the analysis, the system generates specific, targeted improvements:
    *   **Description:** Rewording or adding clarity.
    *   **Schema:** Proposing changes via JSON Patch operations (e.g., adding descriptions to parameters, refining types, adding examples).
    *   **Usage Examples:** Generating new or refining existing examples.
6.  **Validated Schema Patching**: Applies proposed JSON patches to the schema *in-memory* and validates the resulting schema structure before accepting the change for the next iteration.
7.  **Iterative Refinement**: Repeats the cycle (generate tests -> execute -> analyze failures -> propose improvements -> patch schema) until tests consistently pass or a maximum iteration count is reached.
8.  **Optional Winnowing**: After iterations, performs a final pass to condense and streamline the documentation while ensuring critical information discovered during testing is preserved.

### Benefits

-   **Reduces Manual Effort**: Automates the often tedious process of writing and maintaining high-quality tool documentation for LLM consumption.
-   **Improves Agent Performance**: Creates clearer, more precise documentation, leading to fewer errors when LLMs try to use the tools.
-   **Identifies Edge Cases**: The testing process can uncover ambiguities and edge cases that human writers might miss.
-   **Increases Consistency**: Helps establish a more uniform style and level of detail across documentation for all tools.
-   **Adapts to Feedback**: Learns directly from simulated agent failures to target specific documentation weaknesses.
-   **Schema Evolution**: Allows for gradual, validated improvement of tool schemas based on usage simulation.
-   **Detailed Reporting**: Provides comprehensive logs and reports on the entire refinement process, including tests run, failures encountered, and changes made.

### Limitations and Considerations

-   **Cost & Time**: Can be computationally expensive and time-consuming, as it involves multiple LLM calls (for simulation, test generation, failure analysis, improvement proposal) per tool per iteration.
-   **Resource Intensive**: May require significant CPU/memory, especially when refining many tools or using large LLMs for analysis.
-   **LLM Dependency**: The quality of the refinement heavily depends on the capabilities of the LLMs used for the analysis and generation steps.
-   **Schema Complexity**: Generating correct and meaningful JSON Patches for highly complex or nested schemas can be challenging for the LLM.
-   **Determinism**: The process involves LLMs, so results might not be perfectly deterministic between runs.
-   **Maintenance Complexity**: The refiner itself is a complex system with dependencies that require maintenance.

### When to Use

This feature is particularly valuable when:

-   You have a large number of MCP tools exposed to LLM agents.
-   You observe frequent tool usage failures potentially caused by agent misinterpretation of documentation.
-   You are actively developing or expanding your tool ecosystem and need to ensure consistent, high-quality documentation.
-   You want to proactively improve agent reliability and performance without necessarily modifying the underlying tool code itself.
-   You have the budget (LLM credits) and time to invest in this automated quality improvement process.

### Usage Example (Server-Side Invocation)

The documentation refiner is typically invoked as a server-side maintenance or administrative task, not directly exposed as an MCP tool for external agents to call.

```python
# This code snippet shows how the refiner might be called from within the
# server's environment (e.g., via a CLI command or admin interface).

# Assume necessary imports and context setup:
# from ultimate_mcp_server.tools.docstring_refiner import refine_tool_documentation
# from ultimate_mcp_server.core import mcp_context # Represents the server's context

async def invoke_doc_refiner_task():
    # Ensure mcp_context is properly initialized with registered tools, config, etc.
    print("Starting Autonomous Documentation Refinement Task...")

    # Example: Refine documentation for a specific list of tools
    refinement_result = await refine_tool_documentation(
        tool_names=["extract_json", "browser_navigate", "chunk_document"], # Tools to refine
        max_iterations=3, # Limit refinement cycles per tool
        refinement_model_config={ # Specify LLM for refinement tasks
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022"
        },
        testing_model_config={ # Optional: Specify LLM for test generation/simulation
            "provider": "openai",
            "model": "gpt-4o"
        },
        enable_winnowing=True, # Apply final streamlining pass
        stop_on_first_error=False, # Continue refining other tools if one fails
        ctx=mcp_context # Pass the server's MCP context
    )

    # Example: Refine all available tools (potentially very long running)
    # refinement_result = await refine_tool_documentation(
    #     refine_all_available=True,
    #     max_iterations=2,
    #     ctx=mcp_context
    # )

    print("\nDocumentation Refinement Task Complete.")

    # Process the results
    if refinement_result["success"]:
        print(f"Successfully processed {len(refinement_result.get('refined_tools', []))} tools.")
        # The actual docstrings/schemas of the tools in mcp_context might be updated in-memory.
        # Persisting these changes would require additional logic (e.g., writing back to source files).
        print("Detailed report available in the result object.")
        # print(refinement_result.get('report')) # Contains detailed logs and changes
    else:
        print(f"Refinement task encountered errors: {refinement_result.get('error', 'Unknown error')}")
        # Check the report for details on which tools failed and why.

# To run this, it would need to be integrated into the server's startup sequence,
# a dedicated CLI command, or an administrative task runner.
# e.g., await invoke_doc_refiner_task()
```

---

## ✅ Example Library and Testing Framework

The Ultimate MCP Server includes an extensive collection of **35+ end-to-end examples** located in the `examples/` directory. These serve a dual purpose:

1.  **Living Documentation**: They demonstrate practical, real-world usage patterns for nearly every tool and feature.
2.  **Integration Test Suite**: They form a comprehensive test suite ensuring all components work together correctly.

### Example Structure and Organization

-   **Categorized**: Examples are grouped by functionality (e.g., `model_integration`, `tool_specific`, `workflows`, `advanced_features`).
-   **Standalone**: Each example (`*.py`) is a runnable Python script using `mcp-client` to interact with a running server instance.
-   **Clear Output**: They utilize the `Rich` library for formatted, color-coded console output, clearly showing requests, responses, costs, timings, and results.
-   **Error Handling**: Examples include basic error checking for robust demonstration.

### Rich Visual Output

Expect informative console output, including:

-   📊 Tables summarizing results and statistics.
-   🎨 Syntax highlighting for code and JSON.
-   ⏳ Progress indicators or detailed step logging.
-   🖼️ Panels organizing output sections.

*Example output snippet:*
```
╭────────────────────── Tournament Results ───────────────────────╮
│ [1] claude-3-5-haiku-20241022: Score 8.7/10                    │
│     Cost: $0.00013                                             │
│ ...                                                            │
╰────────────────────────────────────────────────────────────────╯
```

### Customizing and Learning

-   **Adaptable**: Easily modify examples to use your API keys (via `.env`), different models, custom prompts, or input files.
-   **Command-Line Args**: Many examples accept arguments for customization (e.g., `--model`, `--input-file`, `--headless`).
-   **Educational**: Learn best practices for AI application structure, tool selection, parameter tuning, error handling, cost optimization, and integration patterns.

### Comprehensive Testing Framework

The `run_all_demo_scripts_and_check_for_errors.py` script orchestrates the execution of all examples as a test suite:

-   **Automated Execution**: Discovers and runs `examples/*.py` sequentially.
-   **Validation**: Checks exit codes and `stderr` against predefined patterns to distinguish real errors from expected messages (e.g., missing API key warnings).
-   **Reporting**: Generates a summary report of passed, failed, and skipped tests, along with detailed logs.

*Example test framework configuration snippet:*
```python
"sql_database_interactions_demo.py": {
    "expected_exit_code": 0,
    "allowed_stderr_patterns": [
        r"Could not compute statistics...", # Known non-fatal warning
        r"Connection failed...", # Expected if DB not set up
        r"Configuration not yet loaded..." # Standard info message
    ]
}
```

### Running the Example Suite

```bash
# Ensure the Ultimate MCP Server is running in a separate terminal

# Run the entire test suite
python run_all_demo_scripts_and_check_for_errors.py

# Run a specific example script directly
python examples/browser_automation_demo.py --headless

# Run an example with custom arguments
python examples/text_redline_demo.py --input-file1 path/to/doc1.txt --input-file2 path/to/doc2.txt
```

This combined example library and testing framework provides invaluable resources for understanding, utilizing, and verifying the functionality of the Ultimate MCP Server.

---

## 💻 CLI Commands

Ultimate MCP Server comes with a command-line interface (`umcp`) for server management and tool interaction:

```bash
# Show available commands and global options
umcp --help

# --- Server Management ---
# Start the server (loads .env, registers tools)
umcp run [--host HOST] [--port PORT] [--include-tools tool1 tool2] [--exclude-tools tool3 tool4]

# --- Information ---
# List configured LLM providers
umcp providers [--check] [--models]

# List available tools
umcp tools [--category CATEGORY] [--examples]

# --- Testing & Interaction ---
# Test connection and basic generation for a specific provider
umcp test <provider_name> [--model MODEL_NAME] [--prompt TEXT]

# Generate a completion directly from the CLI
umcp complete --provider <provider_name> --model <model_name> --prompt "Your prompt here" [--temperature N] [--max-tokens N] [--system TEXT] [--stream]

# --- Cache Management ---
# View or clear the request cache
umcp cache [--status] [--clear]

# --- Benchmark ---
umcp benchmark [--providers P1 P2] [--models M1 M2] [--prompt TEXT] [--runs N]

# --- Examples ---
umcp examples [--list] [<example_name>] [--category CATEGORY]
```

Each command typically has additional options. Use `umcp COMMAND --help` to see options for a specific command (e.g., `umcp complete --help`).

---

## 🛠️ Advanced Configuration

Configuration is primarily managed through **environment variables**, often loaded from a `.env` file in the project root upon startup.

### Server Configuration
-   `SERVER_HOST`: (Default: `127.0.0.1`) Network interface to bind to. Use `0.0.0.0` to listen on all interfaces (necessary for Docker containers or external access).
-   `SERVER_PORT`: (Default: `8013`) Port the server listens on.
-   `API_PREFIX`: (Default: `/`) URL prefix for all API endpoints (e.g., set to `/mcp/v1` to serve under that path).
-   `WORKERS`: (Optional, e.g., `4`) Number of worker processes for the web server (e.g., Uvicorn). Adjust based on CPU cores.

### Tool Filtering (Startup Control)
Control which tools are registered when the server starts using CLI flags:
-   `--include-tools tool1,tool2,...`: Only register the specified tools.
-   `--exclude-tools tool3,tool4,...`: Register all tools *except* those specified.
    ```bash
    # Example: Start with only filesystem and basic completion tools
    umcp run --include-tools read_file,write_file,list_directory,completion
    # Example: Start with all tools except browser automation
    umcp run --exclude-tools browser_init,browser_navigate,browser_click
    ```
    This is useful for creating lightweight instances, managing dependencies, or restricting agent capabilities.

### Logging Configuration
-   `LOG_LEVEL`: (Default: `INFO`) Controls log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). `DEBUG` is very verbose.
-   `USE_RICH_LOGGING`: (Default: `true`) Enables colorful, structured console logs via the Rich library. Set to `false` for plain text logs (better for file redirection or some logging systems).
-   `LOG_FORMAT`: (Optional) Specify a Python `logging` format string for custom log formats (if `USE_RICH_LOGGING=false`).
-   `LOG_TO_FILE`: (Optional, e.g., `/var/log/ultimate_mcp_server.log`) Path to a file where logs should *also* be written (in addition to console). Ensure the server process has write permissions.

### Cache Configuration
-   `CACHE_ENABLED`: (Default: `true`) Globally enable or disable response caching.
-   `CACHE_TTL`: (Default: `86400` seconds = 24 hours) Default Time-To-Live for cached items. Specific tools might have overrides.
-   `CACHE_TYPE`: (Default: `memory`) Backend storage. Check implementation for supported types (e.g., `memory`, `redis`, `diskcache`). `diskcache` provides persistence.
-   `CACHE_DIR`: (Default: `./.cache`) Directory used if `CACHE_TYPE=diskcache`. Ensure write permissions.
-   `CACHE_MAX_SIZE`: (Optional, e.g., `1000` for items or `536870912` for 512MB for `diskcache`) Sets size limits for the cache.
-   `REDIS_URL`: (Required if `CACHE_TYPE=redis`) Connection URL for Redis server (e.g., `redis://localhost:6379/0`).

### Provider Timeouts & Retries
-   `PROVIDER_TIMEOUT`: (Default: `120`) Default timeout in seconds for waiting for a response from an LLM provider API.
-   `PROVIDER_MAX_RETRIES`: (Default: `3`) Default number of times to retry a failed request to a provider (for retryable errors like rate limits or temporary server issues). Uses exponential backoff.
-   Specific provider overrides might exist via dedicated variables (e.g., `OPENAI_TIMEOUT`, `ANTHROPIC_MAX_RETRIES`). Check configuration loading logic or documentation.

### Tool-Specific Configuration
Individual tools might load their own configuration from environment variables. Examples:
-   `ALLOWED_DIRS`: Comma-separated list of base directories filesystem tools are restricted to. **Crucially for security.**
-   `PLAYWRIGHT_BROWSER_TYPE`: (Default: `chromium`) Browser used by Playwright tools (`chromium`, `firefox`, `webkit`).
-   `PLAYWRIGHT_TIMEOUT`: Default timeout for Playwright actions.
-   `DATABASE_URL`: Connection string for the SQL Database Interaction tools (uses SQLAlchemy).
-   `MARQO_URL`: URL for the Marqo instance used by the fused search tool.
-   `TESSERACT_CMD`: Path to the Tesseract executable if not in standard system PATH (for OCR).

*Always ensure environment variables are set correctly **before** starting the server. Changes typically require a server restart to take effect.*

---

## ☁️ Deployment Considerations

While `umcp run` or `docker compose up` are fine for development, consider these for more robust deployments:

### 1. Running as a Background Service
Ensure the server runs continuously and restarts automatically.
-   **`systemd` (Linux):** Create a service unit file (`.service`) to manage the process with `systemctl start|stop|restart|status`. Provides robust control and logging integration.
-   **`supervisor`:** A process control system written in Python. Configure `supervisord` to monitor and manage the server process.
-   **Docker Restart Policies:** Use `--restart unless-stopped` or `--restart always` in your `docker run` command or in `docker-compose.yml` to have Docker manage restarts.

### 2. Using a Reverse Proxy (Nginx, Caddy, Apache, Traefik)
Placing a reverse proxy in front of the Ultimate MCP Server is **highly recommended**:
-   🔒 **HTTPS/SSL Termination:** Handles SSL certificates (e.g., via Let's Encrypt with Caddy/Certbot) encrypting external traffic.
-   ⚖️ **Load Balancing:** Distribute traffic if running multiple instances of the server for high availability or scaling.
-   🗺️ **Path Routing:** Map a clean external URL (e.g., `https://api.yourdomain.com/mcp/`) to the internal server (`http://localhost:8013`). Configure `API_PREFIX` if needed.
-   🛡️ **Security Headers:** Add important headers like `Strict-Transport-Security` (HSTS), `Content-Security-Policy` (CSP).
-   🚦 **Access Control:** Implement IP allow-listing, basic authentication, or integrate with OAuth2 proxies.
-   ⏳ **Buffering/Caching:** May offer additional request/response buffering or caching layers.
-   ⏱️ **Timeouts:** Manage connection timeouts independently from the application server.

*Example Nginx `location` block (simplified):*
```nginx
location /mcp/ { # Match your desired public path (corresponds to API_PREFIX if set)
    proxy_pass http://127.0.0.1:8013/; # Point to the internal server (note trailing /)
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Increase timeouts for potentially long-running AI tasks
    proxy_connect_timeout 60s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;

    # Optional: Add basic authentication
    # auth_basic "Restricted Access";
    # auth_basic_user_file /etc/nginx/.htpasswd;
}
```

### 3. Container Orchestration (Kubernetes, Docker Swarm)
For scalable, managed deployments:
-   ❤️ **Health Checks:** Implement and configure liveness and readiness probes using the server's `/healthz` endpoint (or similar) in your deployment manifests.
-   🔑 **Configuration:** Use ConfigMaps and Secrets (Kubernetes) or Docker Secrets/Configs to manage environment variables and API keys securely, rather than baking them into images or relying solely on `.env` files.
-   ⚙️ **Resource Limits:** Define appropriate CPU and memory requests/limits for the container(s) to ensure stable performance and avoid resource starvation on the node.
-   🌐 **Service Discovery:** Utilize the orchestrator's built-in service discovery instead of hardcoding IPs or hostnames. Expose the service internally (e.g., ClusterIP) and use an Ingress controller for external access.
-   💾 **Persistent Storage:** If using features requiring persistence (e.g., `diskcache`, persistent memory, file storage), configure persistent volumes (PVs/PVCs).

### 4. Resource Allocation
-   **RAM:** Ensure sufficient memory, especially if using large models, in-memory caching, processing large documents, or running memory-intensive tools (like browser automation or certain data processing tasks). Monitor usage.
-   **CPU:** Monitor CPU load. LLM inference itself might not be CPU-bound (often GPU/TPU), but other tools (OCR, local processing, web server handling requests) can be. Consider the number of workers (`WORKERS` env var).
-   **Disk I/O:** Can be a bottleneck if using persistent caching (`diskcache`) or extensive filesystem operations. Use fast storage (SSDs) if needed.
-   **Network:** Ensure adequate bandwidth, especially if handling large documents, images, or frequent/large API responses.

---

## 💸 Cost Savings With Delegation

Using Ultimate MCP Server for intelligent delegation can yield significant cost savings compared to using only a high-end model like Claude 3.7 Sonnet or GPT-4o for every task.

| Task Scenario                   | High-End Model Only (Est.) | Delegated via MCP Server (Est.) | Estimated Savings | Notes                                        |
| :------------------------------ | :------------------------- | :------------------------------ | :---------------- | :------------------------------------------- |
| Summarize 100-page document   | ~$4.50 - $6.00             | ~$0.45 - $0.70 (Gemini Flash)   | **~90%**          | Chunking + parallel cheap summaries          |
| Extract data from 50 records  | ~$2.25 - $3.00             | ~$0.35 - $0.50 (GPT-4.1 Mini)   | **~84%**          | Batch processing with cost-effective model |
| Generate 20 content ideas     | ~$0.90 - $1.20             | ~$0.12 - $0.20 (DeepSeek/Haiku) | **~87%**          | Simple generation task on cheaper model    |
| Process 1,000 customer queries| ~$45.00 - $60.00           | ~$7.50 - $12.00 (Mixed Models)  | **~83%**          | Routing based on query complexity          |
| OCR & Extract from 10 Scans   | ~$1.50 - $2.50 (If LLM OCR)| ~$0.20 - $0.40 (OCR + LLM Fix)  | **~85%**          | Using dedicated OCR + cheap LLM correction |
| Basic Web Scrape & Summarize  | ~$0.50 - $1.00             | ~$0.10 - $0.20 (Browser + Haiku)| **~80%**          | Browser tool + cheap LLM for summary       |

*(Costs are highly illustrative, based on typical token counts and approximate 2024 pricing. Actual costs depend heavily on document size, complexity, specific models used, and current provider pricing.)*

**How savings are achieved:**

-   **Matching Model to Task:** Using expensive models only for tasks requiring deep reasoning, creativity, or complex instruction following.
-   **Leveraging Cheaper Models:** Delegating summarization, extraction, simple Q&A, formatting, etc., to significantly cheaper models (like Gemini Flash, Claude Haiku, GPT-4.1 Mini, DeepSeek Chat).
-   **Using Specialized Tools:** Employing non-LLM tools (Filesystem, OCR, Browser, CLI utils, Database) where appropriate, avoiding LLM API calls entirely for those operations.
-   **Caching:** Reducing redundant API calls for identical or semantically similar requests.

Ultimate MCP Server acts as the intelligent routing layer to make these cost optimizations feasible within a sophisticated agent architecture.

---

## 🧠 Why AI-to-AI Delegation Matters

The strategic importance of AI-to-AI delegation, facilitated by systems like the Ultimate MCP Server, extends beyond simple cost savings:

### Democratizing Advanced AI Capabilities
-   Makes the power of cutting-edge reasoning models (like Claude 3.7, GPT-4o) practically accessible for a wider range of applications by offloading routine work.
-   Allows organizations with budget constraints to leverage top-tier AI capabilities for critical reasoning steps, while managing overall costs effectively.
-   Enables more efficient and widespread use of AI resources across the industry.

### Economic Resource Optimization
-   Represents a fundamental economic optimization in AI usage: applying the most expensive resource (top-tier LLM inference) only where its unique value is required.
-   Complex reasoning, creativity, nuanced understanding, and orchestration are reserved for high-capability models.
-   Routine data processing, extraction, formatting, and simpler Q&A are handled by cost-effective models.
-   Specialized, non-LLM tasks (web scraping, file I/O, DB queries) are handled by purpose-built tools, avoiding unnecessary LLM calls.
-   The overall system aims for near-top-tier performance and capability at a significantly reduced blended cost.
-   Transforms potentially unpredictable LLM API costs into a more controlled expenditure through intelligent routing and caching.

### Sustainable AI Architecture
-   Promotes more sustainable AI usage by reducing the computational demand associated with using the largest models for every single task.
-   Creates a tiered, capability-matched approach to AI resource allocation.
-   Allows for more extensive experimentation and development, as many iterations can utilize cheaper models or tools.
-   Provides a scalable approach to integrating AI that can grow with business needs without costs spiraling uncontrollably.

### Technical Evolution Path
-   Represents an important evolution in AI application architecture, moving beyond monolithic calls to single models towards distributed, multi-agent, multi-model workflows.
-   Enables sophisticated, AI-driven orchestration of complex processing pipelines involving diverse tools and models.
-   Creates a foundation for AI systems that can potentially reason about their own resource usage and optimize dynamically.
-   Builds towards more autonomous, self-optimizing AI systems capable of making intelligent delegation decisions based on context, cost, and required quality.

### The Future of AI Efficiency
-   Ultimate MCP Server points toward a future where AI systems actively manage and optimize their own operational costs and resource usage.
-   Higher-capability models act as intelligent orchestrators or "managers" for ecosystems of specialized tools and more cost-effective "worker" models.
-   AI workflows become increasingly sophisticated, potentially self-organizing and resilient.
-   Organizations can leverage the full spectrum of AI capabilities – from basic processing to advanced reasoning – in a financially viable and scalable manner.

This vision of efficient, intelligently delegated, self-optimizing AI systems represents the next frontier in practical AI deployment, moving beyond the current paradigm of often using a single, powerful (and expensive) model for almost everything.

---

## 🧱 Architecture

### How MCP Integration Works

The Ultimate MCP Server is built natively on the Model Context Protocol (MCP):

1.  **MCP Server Core**: Implements a web server (e.g., using FastAPI) that listens for incoming HTTP requests conforming to the MCP specification (typically POST requests to a specific endpoint).
2.  **Tool Registration**: During startup, the server discovers and registers all available tool implementations. Each tool provides metadata including its name, description, and input/output schemas (often Pydantic models converted to JSON Schema). This registry allows the server (and potentially agents) to know what tools are available and how to use them.
3.  **Tool Invocation**: When an MCP client (like Claude or another application) sends a valid MCP request specifying a tool name and parameters, the server core routes the request to the appropriate registered tool's execution logic.
4.  **Context Passing & Execution**: The tool receives the validated input parameters. It performs its action (calling an LLM, interacting with Playwright, querying a DB, manipulating a file, etc.).
5.  **Structured Response**: The tool's execution result (or error) is packaged into a standard MCP response format, typically including status (success/failure), output data (conforming to the tool's output schema), cost information, and potentially other metadata.
6.  **Return to Client**: The MCP server core sends the structured MCP response back to the originating client over HTTP.

This adherence to the MCP standard ensures seamless, predictable integration with any MCP-compatible agent or client application.

### Component Diagram

```plaintext
+---------------------+       MCP Request        +------------------------------------+       API Request       +-----------------+
|   MCP Agent/Client  | ----------------------> |        Ultimate MCP Server         | ----------------------> |  LLM Providers  |
| (e.g., Claude 3.7)  | <---------------------- | (FastAPI + MCP Core + Tool Logic)  | <---------------------- | (OpenAI, Anthro.)|
+---------------------+      MCP Response       +------------------+-----------------+      API Response       +--------+--------+
                                                            |                                       |
                                                            | Tool Invocation                       | External API Call
                                                            ▼                                       ▼
+-----------------------------------------------------------+------------------------------------------------------------+
| Internal Services & Tool Implementations                                                                               |
| +-------------------+  +-------------------+  +-------------------+  +-------------------+  +-------------------+       |
| | Completion/LLM    |  | Document Proc.    |  | Data Extraction   |  | Browser Automation|  | Excel Automation  |       |
| | (Routing/Provider)|  | (Chunking, Sum.)  |  | (JSON, Table)     |  | (Playwright)      |  | (OpenPyXL/COM)    |       |
| +---------+---------+  +-------------------+  +-------------------+  +-------------------+  +-------------------+       |
|           |                                                                                                            |
| +---------+---------+  +-------------------+  +-------------------+  +-------------------+  +-------------------+       |
| | Cognitive Memory  |  | Filesystem Ops    |  | SQL Database      |  | Entity/Graph      |  | Vector/RAG        |       |
| | (Storage/Query)   |  | (Secure Access)   |  | (SQLAlchemy)      |  | (NetworkX)        |  | (Vector Stores)   |       |
| +-------------------+  +-------------------+  +-------------------+  +-------------------+  +---------+---------+       |
|                                                                                                        |                 |
| +-------------------+  +-------------------+  +-------------------+  +-------------------+  +---------+---------+       |
| | Audio Transcription|  | OCR Tools         |  | Text Classify     |  | CLI Tools         |  | Dynamic API       |       |
| | (Whisper, etc.)   |  | (Tesseract+LLM)   |  |                   |  | (jq, rg, awk)     |  | (OpenAPI->Tool)   |       |
| +-------------------+  +-------------------+  +-------------------+  +-------------------+  +-------------------+       |
|                                                                                                                        |
| +-------------------+  +-------------------+  +-------------------+  +-------------------+  +-------------------+       |
| | Caching Service   |  | Analytics/Metrics |  | Prompt Management |  | Config Service    |  | Meta Tools/Refiner|       |
| | (Memory/Disk/Redis|  | (Cost/Usage Track)|  | (Jinja2/Repo)     |  | (Loads .env)      |  | (list_tools etc.) |       |
| +-------------------+  +-------------------+  +-------------------+  +-------------------+  +-------------------+       |
+------------------------------------------------------------------------------------------------------------------------+
```

### Request Flow for Delegation (Detailed)

1.  **Agent Decision**: An MCP agent determines a need for a specific capability (e.g., summarize a large text, extract JSON, browse a URL) potentially suited for delegation.
2.  **MCP Request Formulation**: The agent constructs an MCP tool invocation request, specifying the `tool_name` and required `inputs` according to the tool's schema (which it might have discovered via `list_tools`).
3.  **HTTP POST to Server**: The agent sends this request (typically as JSON in the body) via HTTP POST to the Ultimate MCP Server's designated endpoint.
4.  **Request Reception & Parsing**: The server's web framework (FastAPI) receives the request. The MCP Core parses the JSON body, validating it against the general MCP request structure.
5.  **Tool Dispatch**: The MCP Core looks up the requested `tool_name` in its registry of registered tools.
6.  **Input Validation**: The server uses the specific tool's input schema (Pydantic model) to validate the `inputs` provided in the request. If validation fails, an MCP error response is generated immediately.
7.  **Tool Execution Context**: A context object might be created, potentially containing configuration, access to shared services (like logging, caching, analytics), etc.
8.  **Caching Check**: The Caching Service is consulted. It generates a cache key based on the `tool_name` and validated `inputs`. If a valid, non-expired cache entry exists for this key, the cached response is retrieved and returned (skipping to step 14).
9.  **Tool Logic Execution**: If not cached, the tool's main execution logic runs:
    *   **LLM Task**: If the tool involves calling an LLM (e.g., `completion`, `summarize_document`, `extract_json`):
        *   The Optimization/Routing logic selects the provider/model based on parameters (`provider`, `model`, `provider_preference`) and server configuration.
        *   The Prompt Management service might format the final prompt using templates.
        *   The Provider Abstraction layer constructs the specific API request for the chosen provider.
        *   The API call is made, handling potential retries and timeouts.
        *   The LLM response is received and parsed.
    *   **Specialized Tool Task**: If it's a non-LLM tool (e.g., `read_file`, `browser_navigate`, `run_sql_query`, `run_ripgrep`):
        *   The tool interacts directly with the relevant system (filesystem, Playwright browser instance, database connection, subprocess execution).
        *   Security checks (e.g., allowed directories, SQL sanitization placeholders) are performed.
        *   The result of the operation is obtained.
10. **Cost Calculation**: For LLM tasks, the Analytics Service calculates the estimated cost based on input/output tokens and provider pricing. For other tasks, the cost is typically zero unless they consume specific metered resources.
11. **Result Formatting**: The tool formats its result (data or error message) according to its defined output schema.
12. **Analytics Recording**: The Analytics Service logs the request, response (or error), execution time, cost, provider/model used, cache status (hit/miss), etc.
13. **Caching Update**: If the operation was successful and caching is enabled for this tool/request, the Caching Service stores the formatted response with its calculated TTL.
14. **MCP Response Formulation**: The MCP Core packages the final result (either from cache or from execution) into a standard MCP response structure, including `status`, `outputs`, `error` (if any), and potentially `cost`, `usage_metadata`.
15. **HTTP Response to Agent**: The server sends the MCP response back to the agent as the HTTP response (typically with a 200 OK status, even if the *tool operation* failed – the MCP request itself succeeded). The agent then parses this response to determine the outcome of the tool call.

---

## 🌍 Real-World Use Cases

### Advanced AI Agent Capabilities
Empower agents like Claude or custom-built autonomous agents to perform complex, multi-modal tasks by giving them tools for:
-   **Persistent Memory & Learning:** Maintain context across long conversations or tasks using the Cognitive Memory system.
-   **Web Interaction & Research:** Automate browsing, data extraction from websites, form submissions, and synthesize information from multiple online sources.
-   **Data Analysis & Reporting:** Create, manipulate, and analyze data within Excel spreadsheets; generate charts and reports.
-   **Database Operations:** Access and query enterprise databases to retrieve or update information based on agent goals.
-   **Document Understanding:** Process PDFs, images (OCR), extract key information, summarize long reports, answer questions based on documents (RAG).
-   **Knowledge Graph Management:** Build and query internal knowledge graphs about specific domains, projects, or entities.
-   **Multimedia Processing:** Transcribe audio recordings from meetings or voice notes.
-   **Code Execution & Analysis:** Use CLI tools or specialized code tools (if added) for development or data tasks.
-   **External Service Integration:** Interact with other company APIs or public APIs dynamically registered via OpenAPI.

### Enterprise Workflow Automation
Build sophisticated automated processes that leverage AI reasoning and specialized tools:
-   **Intelligent Document Processing Pipeline:** Ingest scans/PDFs -> OCR -> Extract structured data (JSON) -> Validate data -> Classify document type -> Route to appropriate system or summarize for human review.
-   **Automated Research Assistant:** Given a topic -> Search academic databases (via Browser/API tool) -> Download relevant papers (Browser/Filesystem) -> Chunk & Summarize papers (Document tools) -> Extract key findings (Extraction tools) -> Store in Cognitive Memory -> Generate synthesized report.
-   **Financial Reporting Automation:** Connect to database (SQL tool) -> Extract financial data -> Populate Excel template (Excel tool) -> Generate charts & variance analysis -> Email report (if an email tool is added).
-   **Customer Support Ticket Enrichment:** Receive ticket text -> Classify issue type (Classification tool) -> Search internal knowledge base & documentation (RAG tool) -> Draft suggested response -> Augment with customer details from CRM (via DB or API tool).
-   **Competitor Monitoring:** Schedule browser automation task -> Visit competitor websites/news feeds -> Extract key announcements/pricing changes -> Summarize findings -> Alert relevant team.

### Data Processing and Integration
Handle complex data tasks beyond simple ETL:
-   **Unstructured to Structured:** Extract specific information (JSON, tables) from emails, reports, chat logs, product reviews.
-   **Knowledge Graph Creation:** Process a corpus of documents (e.g., company wiki, research papers) to build an entity relationship graph for querying insights.
-   **Data Transformation & Cleansing:** Use SQL tools, Excel automation, or local text processing (awk, sed) for complex data manipulation guided by LLM instructions.
-   **Automated Data Categorization:** Apply text classification tools to large datasets (e.g., categorizing user feedback, tagging news articles).
-   **Semantic Data Search:** Build searchable vector indexes over internal documents, enabling users or agents to find information based on meaning, not just keywords (RAG).

### Research and Analysis (Scientific, Market, etc.)
Support research teams with AI-powered tools:
-   **Automated Literature Search & Review:** Use browser/API tools to search databases (PubMed, ArXiv, etc.), download papers, chunk, summarize, and extract key methodologies or results.
-   **Comparative Analysis:** Use multi-provider completion or tournament tools to compare how different models interpret or generate hypotheses based on research data.
-   **Data Extraction from Studies:** Automatically pull structured data (participant numbers, p-values, outcomes) from published papers or reports into a database or spreadsheet.
-   **Budget Tracking:** Utilize the analytics features to monitor LLM API costs associated with research tasks.
-   **Persistent Research Log:** Use the Cognitive Memory system to store findings, hypotheses, observations, and reasoning steps throughout a research project.

### Document Intelligence
Create comprehensive systems for understanding document collections:
-   **End-to-End Pipeline:** OCR scanned documents -> Enhance text with LLMs -> Extract predefined fields (Extraction tools) -> Classify document types -> Identify key entities/relationships -> Generate summaries -> Index text and metadata into a searchable system (Vector/SQL DB).

### Financial Analysis and Modeling
Equip financial professionals with advanced tools:
-   **AI-Assisted Model Building:** Use natural language to instruct the Excel automation tool to create complex financial models, projections, or valuation analyses.
-   **Data Integration:** Pull market data via browser automation or APIs, combine it with internal data from databases (SQL tools).
-   **Report Analysis:** Use RAG or summarization tools to quickly understand long financial reports or filings.
-   **Scenario Testing:** Programmatically modify inputs in Excel models to run sensitivity analyses.
-   **Decision Tracking:** Use Cognitive Memory to log the reasoning behind investment decisions or analyses.

---

## 🔐 Security Considerations

When deploying and operating the Ultimate MCP Server, security must be a primary concern. Consider the following aspects:

1.  🔑 **API Key Management:**
    *   **Never hardcode API keys** in source code or commit them to version control.
    *   Use **environment variables** (`.env` file for local dev, system environment variables, or preferably secrets management tools like HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager for production).
    *   Ensure the `.env` file (if used locally) has **strict file permissions** (e.g., `chmod 600 .env`) readable only by the user running the server.
    *   Use **separate keys** for development and production environments.
    *   Implement **key rotation** policies and revoke suspected compromised keys immediately.

2.  🌐 **Network Exposure & Access Control:**
    *   **Bind to `127.0.0.1` (`SERVER_HOST`)** by default to only allow local connections. Only change to `0.0.0.0` if you intend to expose it, and *only* behind appropriate network controls.
    *   **Use a Reverse Proxy:** (Nginx, Caddy, Traefik, etc.) placed in front of the server is **highly recommended**. It handles SSL/TLS termination, can enforce access controls (IP allow-listing, client certificate auth, Basic Auth, OAuth2 proxy integration), and provides a layer of separation.
    *   **Firewall Rules:** Configure host-based or network firewalls to restrict access to the `SERVER_PORT` only from trusted sources (e.g., the reverse proxy's IP, specific application server IPs, VPN ranges).

3.  👤 **Authentication & Authorization:**
    *   The Ultimate MCP Server itself might not have built-in user/agent authentication. Authentication should typically be handled at a layer *before* the server (e.g., by the reverse proxy or an API gateway).
    *   Ensure that only **authorized clients** (trusted AI agents, specific backend services) can send requests to the server endpoint. Consider using mutual TLS (mTLS) or API keys/tokens managed by the proxy/gateway if needed.
    *   If tools provide different levels of access (e.g., read-only vs. read-write filesystem), consider if authorization logic is needed *within* the server or managed externally.

4.  🚦 **Rate Limiting & Abuse Prevention:**
    *   Implement **rate limiting** at the reverse proxy or API gateway level based on source IP, API key, or other identifiers. This prevents denial-of-service (DoS) attacks and helps control costs from excessive API usage (both LLM and potentially tool usage).
    *   Monitor usage patterns for signs of abuse.

5.  🛡️ **Input Validation & Sanitization:**
    *   While MCP provides a structured format, pay close attention to tools that interact with external systems based on user/agent input:
        *   **Filesystem Tools:** **Crucially**, configure `ALLOWED_DIRS` strictly. Validate and normalize all path inputs rigorously to prevent directory traversal (`../`). Ensure the server process runs with least privilege.
        *   **SQL Tools:** Use parameterized queries or ORMs (like SQLAlchemy) correctly to prevent SQL injection vulnerabilities. Avoid constructing SQL strings directly from agent input.
        *   **Browser Tools:** Be cautious with tools that execute arbitrary JavaScript (`browser_evaluate_script`). Avoid running scripts based directly on untrusted agent input if possible. Playwright's sandboxing helps but isn't foolproof.
        *   **CLI Tools:** Sanitize arguments passed to tools like `run_ripgrep`, `run_jq`, etc., to prevent command injection, especially if constructing complex command strings. Use safe methods for passing input data (e.g., stdin).
    *   Validate input data types and constraints using Pydantic schemas for all tool inputs.

6.  📦 **Dependency Security:**
    *   Regularly **update dependencies** using `uv pip install --upgrade ...` or `uv sync` to patch known vulnerabilities in third-party libraries (FastAPI, Pydantic, Playwright, database drivers, etc.).
    *   Use security scanning tools (`pip-audit`, GitHub Dependabot, Snyk) to automatically identify vulnerable dependencies in your `pyproject.toml` or `requirements.txt`.

7.  📄 **Logging Security:**
    *   Be aware that `DEBUG` level logging might log sensitive information, including full prompts, API responses, file contents, or keys present in data. Configure `LOG_LEVEL` appropriately for production (`INFO` or `WARNING` is usually safer).
    *   Ensure log files (if `LOG_TO_FILE` is used) have appropriate permissions and consider log rotation and retention policies. Avoid logging raw API keys.

8.  ⚙️ **Tool-Specific Security:**
    *   Review the security implications of each specific tool enabled. Does it allow writing files? Executing code? Accessing databases? Ensure configurations (like `ALLOWED_DIRS`, database credentials with limited permissions) follow the principle of least privilege. Disable tools that are not needed or cannot be secured adequately for your environment.

---

## 📃 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## 🙏 Acknowledgements

This project builds upon the work of many fantastic open-source projects and services. Special thanks to:

-   [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) for providing the foundational concepts and protocol specification.
-   [FastAPI](https://fastapi.tiangolo.com/) team for the high-performance web framework.
-   [Pydantic](https://docs.pydantic.dev/) developers for robust data validation and settings management.
-   [Rich](https://github.com/Textualize/rich) library for beautiful and informative terminal output.
-   [uv](https://github.com/astral-sh/uv) from Astral for blazing-fast Python package installation and resolution.
-   [Playwright](https://playwright.dev/) team at Microsoft for the powerful browser automation framework.
-   [OpenPyXL](https://openpyxl.readthedocs.io/en/stable/) maintainers for Excel file manipulation.
-   [SQLAlchemy](https://www.sqlalchemy.org/) developers for the database toolkit.
-   Developers of integrated tools like `Tesseract`, `ripgrep`, `jq`, `awk`, `sed`.
-   All the LLM providers (OpenAI, Anthropic, Google, DeepSeek, xAI, etc.) for making their powerful models accessible via APIs.
-   The broader Python and open-source communities.

---

> _This README provides a comprehensive overview. For specific tool parameters, advanced configuration options, and detailed implementation notes, please refer to the source code and individual tool documentation within the project._

### Running the Server

Start the server using the CLI:

```bash
# Start in default stdio mode
umcp run

# Start in streamable-http mode for web interfaces or remote clients (recommended)
umcp run --transport-mode shttp
# Or use the shortcut:
umcp run -t shttp

# Run on a specific host and port (streamable-http mode)
umcp run -t shttp --host 0.0.0.0 --port 8080
```