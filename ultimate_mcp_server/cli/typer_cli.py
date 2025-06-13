"""Typer CLI implementation for the Ultimate MCP Server."""
import asyncio
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

# Get version hardcoded to avoid import errors
__version__ = "0.1.0"  # Hardcode since there are import issues

from ultimate_mcp_server.cli.commands import (
    benchmark_providers,
    check_cache,
    generate_completion,
    list_providers,
    run_server,
    test_provider,
)
from ultimate_mcp_server.constants import BASE_TOOLSET_CATEGORIES
from ultimate_mcp_server.utils import get_logger

# Use consistent namespace and get console for Rich output
logger = get_logger("ultimate_mcp_server.cli")
console = Console(file=sys.stderr)  # Use stderr to avoid interfering with MCP protocol

# Create typer app
app = typer.Typer(
    name="umcp",
    help= (
        "[bold green]Ultimate MCP Server[/bold green]: Multi-provider LLM management server\n"
        "[italic]Unified CLI to run your server, manage providers, and more.[/italic]"
    ),
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=True,
)


def version_callback(value: bool):
    """Show the version information and exit.
    
    This callback is triggered by the --version/-v flag and displays
    the current version of Ultimate MCP Server before exiting.
    """
    if value:
        console.print(f"Ultimate MCP Server version: [bold]{__version__}[/bold]")
        raise typer.Exit()


class TransportMode(str, Enum):
    """Transport mode for the server."""

    SSE = "sse"
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable-http"
    SHTTP = "shttp"  # Short alias for streamable-http


# Define tool-to-example mapping
TOOL_TO_EXAMPLE_MAP: Dict[str, str] = {
    # Completion tools
    "generate_completion": "simple_completion_demo.py",
    "stream_completion": "simple_completion_demo.py",
    "chat_completion": "claude_integration_demo.py",
    "multi_completion": "multi_provider_demo.py",
    
    # Provider tools
    "get_provider_status": "multi_provider_demo.py",
    "list_models": "multi_provider_demo.py",
    
    # Document tools
    "summarize_document": "document_conversion_and_processing_demo.py",
    "extract_entities": "document_conversion_and_processing_demo.py",
    "chunk_document": "document_conversion_and_processing_demo.py",
    "process_document_batch": "document_conversion_and_processing_demo.py",
    "extract_text_from_pdf": "document_conversion_and_processing_demo.py",
    "process_image_ocr": "document_conversion_and_processing_demo.py",
    
    # Extraction tools
    "extract_json": "advanced_extraction_demo.py",
    "extract_table": "advanced_extraction_demo.py",
    "extract_key_value_pairs": "advanced_extraction_demo.py",
    "extract_semantic_schema": "advanced_extraction_demo.py",
    
    # Entity graph tools
    "extract_entity_graph": "entity_relation_graph_demo.py",
    
    # RAG tools
    "create_knowledge_base": "rag_example.py",
    "add_documents": "rag_example.py",
    "retrieve_context": "rag_example.py",
    "generate_with_rag": "rag_example.py",
    
    # Classification tools
    "text_classification": "text_classification_demo.py",
    
    # Tournament tools
    "create_tournament": "tournament_text_demo.py",
    "list_tournaments": "tournament_text_demo.py",
    "get_tournament_results": "tournament_text_demo.py",
    
    # Optimization tools
    "estimate_cost": "cost_optimization.py",
    "compare_models": "cost_optimization.py",
    "recommend_model": "cost_optimization.py",
    
    # Filesystem tools
    "read_file": "filesystem_operations_demo.py",
    "write_file": "filesystem_operations_demo.py",
    "list_directory": "filesystem_operations_demo.py",
    "search_files": "filesystem_operations_demo.py",
    

    # HTML tools
    "clean_and_format_text_as_markdown": "html_to_markdown_demo.py",
    
    # Text comparison tools
    "compare_documents_redline": "text_redline_demo.py",
    
    # Marqo search tools
    "marqo_fused_search": "marqo_fused_search_demo.py",
    
    # SQL tools
    "connect_to_database": "sql_database_interactions_demo.py",
    "execute_query": "sql_database_interactions_demo.py",
    
    # Audio tools
    "transcribe_audio": "audio_transcription_demo.py",
    
    # Browser automation tools
    "browser_init": "browser_automation_demo.py",
    "execute_web_workflow": "browser_automation_demo.py",
}

# Group examples by category
EXAMPLE_CATEGORIES: Dict[str, List[str]] = {
    "text-generation": [
        "simple_completion_demo.py",
        "claude_integration_demo.py",
        "multi_provider_demo.py",
        "grok_integration_demo.py",
    ],
    "document-conversion-and-processing": [
        "document_conversion_and_processing_demo.py",
        "advanced_extraction_demo.py",
    ],
    "search-and-retrieval": [
        "rag_example.py",
        "vector_search_demo.py",
        "advanced_vector_search_demo.py",
        "marqo_fused_search_demo.py",
    ],
    "browser-automation": [
        "browser_automation_demo.py",
        "sse_client_demo.py",
    ],
    "data-analysis": [
        "sql_database_interactions_demo.py",
        "analytics_reporting_demo.py",
    ],
    "specialized-tools": [
        "audio_transcription_demo.py",
        "text_redline_demo.py",
        "html_to_markdown_demo.py",
        "entity_relation_graph_demo.py",
    ],
    "workflows": [
        "workflow_delegation_demo.py",
        "tool_composition_examples.py",
        "research_workflow_demo.py",
    ],
}


# Define option constants to avoid function calls in default arguments
HOST_OPTION = typer.Option(
    None,
    "-h",
    "--host",
    help="[cyan]Host[/cyan] or [cyan]IP address[/cyan] to bind the server to (-h shortcut). Defaults from config.",
    rich_help_panel="Server Options",
)
PORT_OPTION = typer.Option(
    None,
    "-p",
    "--port",
    help="[cyan]Port[/cyan] to listen on (-p shortcut). Defaults from config.",
    rich_help_panel="Server Options",
)
WORKERS_OPTION = typer.Option(
    None,
    "-w",
    "--workers",
    help="[cyan]Number of worker[/cyan] processes to spawn (-w shortcut). Defaults from config.",
    rich_help_panel="Server Options",
)
TRANSPORT_MODE_OPTION = typer.Option(
    TransportMode.STREAMABLE_HTTP,
    "-t",
    "--transport-mode",
    help="[cyan]Transport mode[/cyan] for server communication (-t shortcut). Options: 'sse', 'stdio', 'streamable-http', 'shttp'.",
    rich_help_panel="Server Options",
)
DEBUG_OPTION = typer.Option(
    False,
    "-d",
    "--debug",
    help="[yellow]Enable debug logging[/yellow] for detailed output (-d shortcut).",
    rich_help_panel="Server Options",
)
INCLUDE_TOOLS_OPTION = typer.Option(
    None,
    "--include-tools",
    help="[green]List of tool names to include[/green] when running the server. Adds to the 'Base Toolset' by default, or to all tools if --load-all-tools is used.",
    rich_help_panel="Server Options",
)
EXCLUDE_TOOLS_OPTION = typer.Option(
    None,
    "--exclude-tools",
    help="[red]List of tool names to exclude[/red] when running the server. Applies after including tools.",
    rich_help_panel="Server Options",
)
LOAD_ALL_TOOLS_OPTION = typer.Option(
    False,
    "-a",
    "--load-all-tools",
    help="[yellow]Load all available tools[/yellow] instead of just the default 'Base Toolset' (-a shortcut).",
    rich_help_panel="Server Options",
)

CHECK_OPTION = typer.Option(
    False,
    "-c",
    "--check",
    help="[yellow]Check API keys[/yellow] for all configured providers (-c shortcut).",
    rich_help_panel="Provider Options",
)
MODELS_OPTION = typer.Option(
    False,
    "--models",
    help="[green]List available models[/green] for each provider.",
    rich_help_panel="Provider Options",
)

MODEL_OPTION = typer.Option(
    None,
    "--model",
    help="[cyan]Model ID[/cyan] to test (defaults to the provider's default).",
    rich_help_panel="Test Options",
)
PROMPT_OPTION = typer.Option(
    "Hello, world!",
    "--prompt",
    help="[magenta]Prompt text[/magenta] to send to the provider.",
    rich_help_panel="Test Options",
)

PROVIDER_OPTION = typer.Option(
    "openai",
    "--provider",
    help="[cyan]Provider[/cyan] to use (default: openai)",
    rich_help_panel="Completion Options",
)
COMPLETION_MODEL_OPTION = typer.Option(
    None,
    "--model",
    help="[blue]Model ID[/blue] for completion (defaults to the provider's default)",
    rich_help_panel="Completion Options",
)
COMPLETION_PROMPT_OPTION = typer.Option(
    None,
    "--prompt",
    help="[magenta]Prompt text[/magenta] for generation (reads from stdin if not provided)",
    rich_help_panel="Completion Options",
)
TEMPERATURE_OPTION = typer.Option(
    0.7,
    "--temperature",
    help="[yellow]Sampling temperature[/yellow] (0.0 - 2.0, default: 0.7)",
    rich_help_panel="Completion Options",
)
MAX_TOKENS_OPTION = typer.Option(
    None,
    "--max-tokens",
    help="[green]Max tokens[/green] to generate (defaults to provider's setting)",
    rich_help_panel="Completion Options",
)
SYSTEM_OPTION = typer.Option(
    None,
    "--system",
    help="[blue]System prompt[/blue] for providers that support it.",
    rich_help_panel="Completion Options",
)
STREAM_OPTION = typer.Option(
    False,
    "-s",
    "--stream",
    help="[cyan]Stream[/cyan] the response token by token (-s shortcut).",
    rich_help_panel="Completion Options",
)

STATUS_OPTION = typer.Option(
    True,
    "--status",
    help="[green]Show cache status[/green]",
    rich_help_panel="Cache Options",
)
CLEAR_OPTION = typer.Option(
    False,
    "--clear",
    help="[red]Clear the cache[/red]",
    rich_help_panel="Cache Options",
)

DEFAULT_PROVIDERS = ["openai", "anthropic", "deepseek", "gemini", "openrouter"]
PROVIDERS_OPTION = typer.Option(
    DEFAULT_PROVIDERS,
    "--providers",
    help="[cyan]Providers list[/cyan] to benchmark (default: all)",
    rich_help_panel="Benchmark Options",
)
BENCHMARK_MODELS_OPTION = typer.Option(
    None,
    "--models",
    help="[blue]Model IDs[/blue] to benchmark (defaults to default model of each provider)",
    rich_help_panel="Benchmark Options",
)
BENCHMARK_PROMPT_OPTION = typer.Option(
    None,
    "--prompt",
    help="[magenta]Prompt text[/magenta] to use for benchmarking (default built-in)",
    rich_help_panel="Benchmark Options",
)
RUNS_OPTION = typer.Option(
    3,
    "-r",
    "--runs",
    help="[green]Number of runs[/green] per provider/model (-r shortcut, default: 3)",
    rich_help_panel="Benchmark Options",
)

VERSION_OPTION = typer.Option(
    False,
    "--version",
    "-v",
    callback=version_callback,
    is_eager=True,
    help="[yellow]Show the application version and exit.[/yellow]",
    rich_help_panel="Global Options",
)

# Options for tools command
CATEGORY_OPTION = typer.Option(
    None,
    "--category",
    help="[cyan]Filter category[/cyan] when listing tools.",
    rich_help_panel="Tools Options",
)
CATEGORY_FILTER_OPTION = typer.Option(
    None,
    "--category",
    help="[cyan]Filter category[/cyan] when listing examples.",
    rich_help_panel="Examples Options",
)

# Options for examples command
SHOW_EXAMPLES_OPTION = typer.Option(
    False,
    "--examples",
    help="[magenta]Show example scripts[/magenta] alongside tools.",
    rich_help_panel="Tools Options",
)

LIST_OPTION = typer.Option(
    False,
    "-l",
    "--list",
    help="[green]List examples[/green] instead of running one (-l shortcut).",
    rich_help_panel="Examples Options",
)


@app.command(name="run")
def run(
    host: Optional[str] = HOST_OPTION,
    port: Optional[int] = PORT_OPTION,
    workers: Optional[int] = WORKERS_OPTION,
    transport_mode: TransportMode = TRANSPORT_MODE_OPTION,
    debug: bool = DEBUG_OPTION,
    include_tools: List[str] = INCLUDE_TOOLS_OPTION,
    exclude_tools: List[str] = EXCLUDE_TOOLS_OPTION,
    load_all_tools: bool = LOAD_ALL_TOOLS_OPTION,
):
    """
    [bold green]Run the Ultimate MCP Server[/bold green]

    Start the MCP server with configurable networking, performance, and tool options.
    The server exposes MCP-protocol compatible endpoints that AI agents can use to
    access various tools and capabilities.

    By default, only the [yellow]'Base Toolset'[/yellow] is loaded to optimize context window usage.
    Use `--load-all-tools` to load all available tools.

    Network settings control server accessibility, workers affect concurrency,
    and tool filtering lets you customize which capabilities are exposed.

    [bold]Examples:[/bold]
      [cyan]umcp run --host 0.0.0.0 --port 8000 --workers 4[/cyan] (Runs with Base Toolset)
      [cyan]umcp run --load-all-tools --debug[/cyan] (Runs with all tools and debug logging)
      [cyan]umcp run --include-tools browser,audio[/cyan] (Adds browser and audio tools to the Base Toolset)
      [cyan]umcp run --load-all-tools --exclude-tools filesystem[/cyan] (Loads all tools except filesystem)
    """
    # Set debug mode if requested
    if debug:
        os.environ["LOG_LEVEL"] = "DEBUG"

    # Print server info
    server_info_str = (
        f"Host: [cyan]{host or 'default from config'}[/cyan]\n"
        f"Port: [cyan]{port or 'default from config'}[/cyan]\n"
        f"Workers: [cyan]{workers or 'default from config'}[/cyan]\n"
        f"Transport mode: [cyan]{transport_mode}[/cyan]"
    )
    
    # Tool Loading Status
    if load_all_tools:
        server_info_str += "\nTool Loading: [yellow]All Available Tools[/yellow]"
    else:
        server_info_str += "\nTool Loading: [yellow]Base Toolset Only[/yellow] (Use --load-all-tools to load all)"
        # Format the categories for display
        category_lines = []
        for category, tools in BASE_TOOLSET_CATEGORIES.items():
            category_lines.append(f"    [cyan]{category}[/cyan]: {', '.join(tools)}")
        
        server_info_str += "\n  [bold]Includes:[/bold]\n" + "\n".join(category_lines)


    # Print tool filtering info if enabled
    if include_tools or exclude_tools:
        server_info_str += "\n[bold]Tool Filtering:[/bold]"
        if include_tools:
            server_info_str += f"\nIncluding: [cyan]{', '.join(include_tools)}[/cyan]"
        if exclude_tools:
            server_info_str += f"\nExcluding: [red]{', '.join(exclude_tools)}[/red]"

    console.print(Panel(server_info_str, title="[bold blue]Starting Ultimate MCP Server[/bold blue]", expand=False))
    console.print() # Add a newline for spacing
    
    # Convert transport_mode enum to string and handle aliases
    if transport_mode == TransportMode.SHTTP:
        actual_transport_mode = "streamable-http"
    else:
        # Convert enum to string value (e.g., TransportMode.SSE -> "sse")
        actual_transport_mode = transport_mode.value
    
    # Run the server
    run_server(
        host=host,
        port=port,
        workers=workers,
        transport_mode=actual_transport_mode,
        include_tools=include_tools,
        exclude_tools=exclude_tools,
        load_all_tools=load_all_tools,
    )


@app.command(name="providers")
def providers(
    check: bool = CHECK_OPTION,
    models: bool = MODELS_OPTION,
):
    """
    [bold green]List Available Providers[/bold green]

    Display configured LLM providers (OpenAI, Anthropic, Gemini, etc.) 
    with their connection status, default models, and API key validation.
    
    Use this command to verify your configuration, troubleshoot API keys,
    or explore available models across all providers.

    Usage:
      umcp providers               # Basic provider listing
      umcp providers --check       # Validate API keys with providers
      umcp providers --models      # List available models for each provider

    Examples:
      umcp providers --check --models  # Comprehensive provider diagnostics
    """
    asyncio.run(list_providers(check_keys=check, list_models=models))


@app.command(name="test")
def test(
    provider: str = typer.Argument(..., help="Provider to test (openai, anthropic, deepseek, gemini)", rich_help_panel="Test Options"),
    model: Optional[str] = MODEL_OPTION,
    prompt: str = PROMPT_OPTION,
):
    """
    [bold green]Test a Specific Provider[/bold green]

    Verify connectivity and functionality of an LLM provider by sending a test
    prompt and displaying the response. This command performs a full API round-trip
    to validate your credentials, model availability, and proper configuration.
    
    The output includes the response text, token counts, cost estimate,
    and response time metrics to help diagnose performance issues.

    Usage:
      umcp test openai                          # Test default OpenAI model
      umcp test anthropic --model claude-3-5-haiku-20241022  # Test specific model
      umcp test gemini --prompt "Summarize quantum computing"  # Custom prompt

    Examples:
      umcp test openai  # Quick health check with default settings
    """
    with console.status(f"[bold green]Testing provider '{provider}'..."): 
        try:
            asyncio.run(test_provider(provider=provider, model=model, prompt=prompt))
        except Exception as e:
            console.print(Panel(f"Failed to test provider '{provider}':\n{str(e)}", title="[bold red]Test Error[/bold red]", border_style="red"))
            raise typer.Exit(code=1) from e


@app.command(name="complete")
def complete(
    provider: str = PROVIDER_OPTION,
    model: Optional[str] = COMPLETION_MODEL_OPTION,
    prompt: Optional[str] = COMPLETION_PROMPT_OPTION,
    temperature: float = TEMPERATURE_OPTION,
    max_tokens: Optional[int] = MAX_TOKENS_OPTION,
    system: Optional[str] = SYSTEM_OPTION,
    stream: bool = STREAM_OPTION,
):
    """
    [bold green]Generate Text Completion[/bold green]

    Request text generation directly from an LLM provider through the CLI.
    This command bypasses the server's MCP endpoint and sends requests
    directly to the provider's API, useful for testing or quick generations.
    
    Supports input from arguments, stdin (piped content), or interactive prompt.
    The command provides full control over provider selection, model choice, 
    and generation parameters. Results include token counts and cost estimates.

    Usage:
      echo "Tell me about Mars" | umcp complete             # Pipe content as prompt
      umcp complete --prompt "Write a haiku"                # Direct prompt
      umcp complete --provider anthropic --model claude-3-5-sonnet-20241022  # Specify model
      umcp complete --temperature 1.5 --max-tokens 250      # Adjust generation params
      umcp complete --system "You are a helpful assistant"  # Set system prompt
      umcp complete --stream                                # Stream tokens in real-time

    Examples:
      umcp complete --prompt "Write a haiku about autumn." --stream
    """
    # Get prompt from stdin if not provided
    if prompt is None:
        if sys.stdin.isatty():
            console.print("Enter prompt (Ctrl+D to finish):")
        prompt = sys.stdin.read().strip()
    
    asyncio.run(
        generate_completion(
            provider=provider,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            stream=stream,
        )
    )


@app.command(name="cache")
def cache(
    status: bool = STATUS_OPTION,
    clear: bool = CLEAR_OPTION,
):
    """
    [bold green]Cache Management[/bold green]

    Monitor and maintain the server's response cache system.
    Caching stores previous LLM responses to avoid redundant API calls, 
    significantly reducing costs and latency for repeated or similar requests.
    
    The status view shows backend type, item count, hit rate percentage,
    and estimated cost savings from cache hits. Clearing the cache removes
    all stored responses, which may be necessary after configuration changes
    or to force fresh responses.

    Usage:
      umcp cache                # View cache statistics and status
      umcp cache --status       # Explicitly request status view
      umcp cache --clear        # Remove all cached entries (with confirmation)
      umcp cache --status --clear  # View stats before clearing

    Examples:
      umcp cache  # Check current cache performance and hit rate
    """
    should_clear = False
    if clear:
        if Confirm.ask("[bold yellow]Are you sure you want to clear the cache?[/bold yellow]"):
            should_clear = True
        else:
            console.print("[yellow]Cache clear aborted.[/yellow]")
            raise typer.Exit()

    # Only run the async part if needed
    if status or should_clear:
        with console.status("[bold green]Accessing cache..."):
            try:
                asyncio.run(check_cache(show_status=status, clear=should_clear))
            except Exception as e:
                console.print(Panel(f"Failed to access cache:\n{str(e)}", title="[bold red]Cache Error[/bold red]", border_style="red"))
                raise typer.Exit(code=1) from e
    elif not clear: # If only clear was specified but user aborted
        pass # Do nothing, already printed message
    else:
        console.print("Use --status to view status or --clear to clear the cache.")


@app.command(name="benchmark")
def benchmark(
    providers: List[str] = PROVIDERS_OPTION,
    models: Optional[List[str]] = BENCHMARK_MODELS_OPTION,
    prompt: Optional[str] = BENCHMARK_PROMPT_OPTION,
    runs: int = RUNS_OPTION,
):
    """
    [bold green]Benchmark Providers[/bold green]

    Compare performance metrics and costs across different LLM providers and models.
    The benchmark sends identical prompts to each selected provider/model combination
    and measures response time, token processing speed, and cost per request.
    
    Results are presented in a table format showing average metrics across
    multiple runs to ensure statistical validity. This helps identify the
    most performant or cost-effective options for your specific use cases.

    Usage:
      umcp benchmark                     # Test all configured providers
      umcp benchmark --providers openai,anthropic  # Test specific providers
      umcp benchmark --models gpt-4o,claude-3-5-haiku  # Test specific models
      umcp benchmark --prompt "Explain quantum computing" --runs 5  # Custom benchmark
      
    Examples:
      umcp benchmark --runs 3 --providers openai,gemini  # Compare top providers
    """
    asyncio.run(benchmark_providers(providers=providers, models=models, prompt=prompt, runs=runs))


@app.command(name="tools")
def tools(
    category: Optional[str] = CATEGORY_OPTION,
    show_examples: bool = SHOW_EXAMPLES_OPTION,
):
    """
    [bold green]List Available Tools[/bold green]

    Display the MCP tools registered in the server, organized by functional categories.
    These tools represent the server's capabilities that can be invoked by AI agents
    through the Model Context Protocol (MCP) interface.
    
    Tools are grouped into logical categories like completion, document processing,
    filesystem access, browser automation, and more. For each tool, you can
    optionally view associated example scripts that demonstrate its usage patterns.

    Usage:
      umcp tools                      # List all tools across all categories
      umcp tools --category document  # Show only document-related tools
      umcp tools --examples           # Show example scripts for each tool

    Examples:
      umcp tools --category filesystem --examples  # Learn filesystem tools with examples
    """
    # Manually list tools by category for demonstration
    tool_categories: Dict[str, List[str]] = {
        "completion": [
            "generate_completion",
            "stream_completion", 
            "chat_completion", 
            "multi_completion"
        ],
        "document": [
            "summarize_document",
            "extract_entities",
            "chunk_document",
            "process_document_batch"
        ],
        "extraction": [
            "extract_json",
            "extract_table",
            "extract_key_value_pairs",
            "extract_semantic_schema"
        ],
        "rag": [
            "create_knowledge_base",
            "add_documents",
            "retrieve_context",
            "generate_with_rag"
        ],
        "filesystem": [
            "read_file",
            "write_file",
            "list_directory",
            "search_files"
        ],
        "browser": [
            "browser_init",
            "browser_navigate",
            "browser_click",
            "execute_web_workflow"
        ]
    }
    
    # Filter by category if specified
    if category and category in tool_categories:
        categories_to_show = {category: tool_categories[category]}
    else:
        categories_to_show = tool_categories
    
    # Create Rich table for display
    table = Table(title="Ultimate MCP Server Tools")
    table.add_column("Category", style="cyan")
    table.add_column("Tool", style="green")
    
    if show_examples:
        table.add_column("Example Script", style="yellow")
    
    # Add rows to table
    for module_name, tool_names in sorted(categories_to_show.items()):
        for tool_name in sorted(tool_names):
            example_script = TOOL_TO_EXAMPLE_MAP.get(tool_name, "")
            
            if show_examples:
                table.add_row(
                    module_name, 
                    tool_name,
                    example_script if example_script else "N/A"
                )
            else:
                table.add_row(module_name, tool_name)
    
    console.print(table)
    
    # Print help for running examples
    if show_examples:
        console.print("\n[bold]Tip:[/bold] Run examples using the command:")
        console.print("  [cyan]umcp examples <example_name>[/cyan]")


@app.command(name="examples")
def examples(
    example_name: Optional[str] = typer.Argument(None, help="Name of the example to run"),
    category: Optional[str] = CATEGORY_FILTER_OPTION,
    list_examples: bool = LIST_OPTION,
):
    """
    [bold green]Run or List Example Scripts[/bold green]

    Browse and execute the demonstration Python scripts included with Ultimate MCP Server.
    These examples showcase real-world usage patterns and integration techniques for
    different server capabilities, from basic completions to complex workflows.
    
    Examples are organized by functional category (text-generation, document-processing,
    browser-automation, etc.) and contain fully functional code that interacts with
    a running MCP server. They serve as both educational resources and starting
    points for your own implementations.

    Usage:
      umcp examples               # List all available example scripts
      umcp examples --list        # List-only mode (same as above)
      umcp examples --category browser-automation  # Filter by category
      umcp examples rag_example   # Run specific example (with or without .py extension)
      umcp examples rag_example.py  # Explicit extension version

    Examples:
      umcp examples simple_completion_demo  # Run the basic completion example
    """
    # Ensure we have the examples directory
    project_root = Path(__file__).parent.parent.parent
    examples_dir = project_root / "examples"
    
    if not examples_dir.exists() or not examples_dir.is_dir():
        console.print(f"[bold red]Error:[/bold red] Examples directory not found at: {examples_dir}")
        console.print(Panel(f"Examples directory not found at: {examples_dir}", title="[bold red]Error[/bold red]", border_style="red"))
        return 1
    
    # If just listing examples
    if list_examples or not example_name:
        # Create Rich table for display
        table = Table(title="Ultimate MCP Server Example Scripts")
        table.add_column("Category", style="cyan")
        table.add_column("Example Script", style="green")
        
        # List available examples by category
        for category_name, script_names in sorted(EXAMPLE_CATEGORIES.items()):
            for script_name in sorted(script_names):
                table.add_row(category_name, script_name)
        
        console.print(table)
        
        # Print help for running examples
        console.print("\n[bold]Run an example:[/bold]")
        console.print("  [cyan]umcp examples <example_name>[/cyan]")
        
        return 0
    
    # Run the specified example
    example_file = None
    
    # Check if .py extension was provided
    if example_name.endswith('.py'):
        example_path = examples_dir / example_name
        if example_path.exists():
            example_file = example_path
    else:
        # Try with .py extension
        example_path = examples_dir / f"{example_name}.py"
        if example_path.exists():
            example_file = example_path
        else:
            # Try finding in the tool map
            if example_name in TOOL_TO_EXAMPLE_MAP:
                example_script = TOOL_TO_EXAMPLE_MAP[example_name]
                example_path = examples_dir / example_script
                if example_path.exists():
                    example_file = example_path
    
    if not example_file:
        console.print(f"[bold red]Error:[/bold red] Example '{example_name}' not found")
        console.print(Panel(f"Example script '{example_name}' not found in {examples_dir}", title="[bold red]Error[/bold red]", border_style="red"))
        return 1
    
    # Run the example script
    console.print(f"[bold blue]Running example:[/bold blue] {example_file.name}")
    
    # Change to the project root directory to ensure imports work
    os.chdir(project_root)
    
    # Use subprocess to run the script
    import subprocess
    try:
        # Execute the Python script
        result = subprocess.run(
            [sys.executable, str(example_file)], 
            check=True
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error:[/bold red] Example script failed with exit code {e.returncode}")
        console.print(Panel(f"Example script '{example_file.name}' failed with exit code {e.returncode}", title="[bold red]Execution Error[/bold red]", border_style="red"))
        return e.returncode
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to run example: {str(e)}")
        console.print(Panel(f"Failed to run example '{example_file.name}':\n{str(e)}", title="[bold red]Execution Error[/bold red]", border_style="red"))
        return 1


@app.callback()
def main(
    version: bool = VERSION_OPTION,
):
    """Ultimate MCP Server - A comprehensive AI agent operating system.
    
    The Ultimate MCP Server provides a unified interface to manage LLM providers,
    tools, and capabilities through the Model Context Protocol (MCP). It enables
    AI agents to access dozens of powerful capabilities including file operations,
    browser automation, document processing, database access, and much more.

    This CLI provides commands to:
    • Start and configure the server
    • Manage and test LLM providers
    • Generate text completions directly
    • View and clear the response cache
    • Benchmark provider performance
    • List available tools and capabilities
    • Run example scripts demonstrating usage patterns
    """
    # This function will be called before any command
    pass


def cli():
    """Entry point for CLI package installation.
    
    This function serves as the main entry point when the package is installed
    and the 'umcp' command is invoked. It's referenced in pyproject.toml's
    [project.scripts] section to create the command-line executable.
    """
    app()


if __name__ == "__main__":
    app() 