#!/usr/bin/env python
"""Workflow delegation example using Ultimate MCP Server."""
import asyncio
import json
import sys
import time
from collections import namedtuple  # Import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import FastMCP
from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import get_provider
from ultimate_mcp_server.core.server import Gateway
from ultimate_mcp_server.exceptions import ToolExecutionError
from ultimate_mcp_server.utils import get_logger, process_mcp_result

# --- Add Display Utils Import ---
from ultimate_mcp_server.utils.display import CostTracker, _display_stats  # Import CostTracker

# --- Add Rich Imports ---
from ultimate_mcp_server.utils.logging.console import console

# --- Import Tools Needed ---
# Import tool functions directly if not registering them all
# from ultimate_mcp_server.tools.optimization import recommend_model, execute_optimized_workflow # No, call via MCP
# from ultimate_mcp_server.tools.completion import generate_completion # Call via MCP
# -------------------------

# Initialize logger
logger = get_logger("example.workflow_delegation")

# Create a simple structure for cost tracking from dict
TrackableResult = namedtuple("TrackableResult", ["cost", "input_tokens", "output_tokens", "provider", "model", "processing_time"])

# Initialize FastMCP server
mcp = FastMCP("Workflow Delegation Demo")

# Mock provider initialization function (replace with actual if needed)
async def initialize_providers():
    logger.info("Initializing required providers...", emoji_key="provider")
    
    # Initialize gateway to let it handle provider initialization
    gateway = Gateway("workflow-delegation-demo", register_tools=False)
    await gateway._initialize_providers()
    
    # Check if we have the necessary providers initialized
    required_providers = ["openai", "anthropic", "gemini"]
    missing_providers = []
    
    for provider_name in required_providers:
        try:
            provider = await get_provider(provider_name)
            if provider:
                logger.info(f"Provider {provider_name} is available", emoji_key="success")
            else:
                missing_providers.append(provider_name)
        except Exception:
            missing_providers.append(provider_name)
    
    if missing_providers:
        logger.warning(f"Missing providers: {', '.join(missing_providers)}. Some demos might fail.", emoji_key="warning")
        console.print(f"[yellow]Warning:[/yellow] Missing providers: {', '.join(missing_providers)}")
    else:
        logger.info("All required providers are available", emoji_key="success")

# Keep execute_workflow as a locally defined tool demonstrating the concept
@mcp.tool()
async def execute_workflow(
    workflow_steps: List[Dict[str, Any]],
    initial_input: Optional[str] = None, # Make initial_input optional
    max_concurrency: int = 1, # Keep concurrency, though sequential for demo
    ctx = None # Keep ctx for potential use by called tools
) -> Dict[str, Any]:
    """Execute a multi-step workflow by calling registered project tools."""
    start_time = time.time()
    total_cost = 0.0
    step_results: Dict[str, Any] = {} # Store results keyed by step_id

    # Mapping from simple operation names to actual tool names
    operation_to_tool_map = {
        "summarize": "summarize_document",
        "extract_entities": "extract_entities",
        "generate_questions": "generate_qa_pairs", # Correct tool name
        "chunk": "chunk_document",
        # Add mappings for other tools as needed
        "completion": "generate_completion", 
        "chat": "chat_completion",
        "retrieve": "retrieve_context",
        "rag_generate": "generate_with_rag",
    }

    current_input_value = initial_input
    logger.info(f"Starting workflow execution with {len(workflow_steps)} steps.")

    for i, step in enumerate(workflow_steps):
        step_id = step.get("id")
        operation = step.get("operation")
        tool_name = operation_to_tool_map.get(operation)
        parameters = step.get("parameters", {}).copy() # Get parameters
        input_from_step = step.get("input_from") # ID of previous step for input
        output_as = step.get("output_as", step_id) # Key to store output under

        if not step_id:
             raise ValueError(f"Workflow step {i} is missing required 'id' key.")
        if not tool_name:
            raise ValueError(f"Unsupported operation '{operation}' in workflow step '{step_id}'. Mapped tool name not found.")

        logger.info(f"Executing workflow step {i+1}/{len(workflow_steps)}: ID='{step_id}', Tool='{tool_name}'")

        # Resolve input: Use previous step output or initial input
        step_input_data = None
        if input_from_step:
            if input_from_step not in step_results:
                 raise ValueError(f"Input for step '{step_id}' requires output from '{input_from_step}', which has not run or failed.")
            # Decide which part of the previous result to use
            # This needs a more robust mechanism (e.g., specifying the key)
            # For now, assume the primary output is needed (e.g., 'text', 'summary', 'chunks', etc.)
            prev_result = step_results[input_from_step]
            # Simple logic: look for common output keys
            if isinstance(prev_result, dict):
                 if 'summary' in prev_result: 
                     step_input_data = prev_result['summary']
                 elif 'text' in prev_result: 
                     step_input_data = prev_result['text']
                 elif 'chunks' in prev_result: 
                     step_input_data = prev_result['chunks'] # May need specific handling
                 elif 'result' in prev_result: 
                     step_input_data = prev_result['result'] # From DocumentResponse
                 else: 
                     step_input_data = prev_result # Pass the whole dict?
            else:
                 step_input_data = prev_result # Pass raw output
            logger.debug(f"Using output from step '{input_from_step}' as input.")
        else:
            step_input_data = current_input_value # Use input from previous step or initial
            logger.debug("Using input from previous step/initial input.")

        # --- Construct parameters for the target tool --- 
        # This needs mapping based on the target tool's expected signature
        # Example: If tool is 'summarize_document', map step_input_data to 'document' param
        if tool_name == "summarize_document" and isinstance(step_input_data, str):
            parameters["document"] = step_input_data
        elif tool_name == "extract_entities" and isinstance(step_input_data, str):
             parameters["document"] = step_input_data
             # Ensure entity_types is a list
             if "entity_types" not in parameters or not isinstance(parameters["entity_types"], list):
                  parameters["entity_types"] = ["organization", "person", "concept"] # Default
        elif tool_name == "generate_qa_pairs" and isinstance(step_input_data, str):
             parameters["document"] = step_input_data
             parameters["num_pairs"] = parameters.get("num_questions") or 5 # Map parameter name
        elif tool_name in ["generate_completion", "chat_completion"] and isinstance(step_input_data, str):
            if "prompt" not in parameters:
                 parameters["prompt"] = step_input_data # Assume input is the prompt if not specified
        # Add more mappings as needed for other tools...
        else:
             # Fallback: pass the input data under a generic key if not handled?
             # Or maybe the tool parameter should explicitly name the input field?
             # For now, we assume the tool can handle the input directly if not mapped.
             # This requires careful workflow definition. 
             # Maybe add 'input_arg_name' to workflow step definition?
             logger.warning(f"Input mapping for tool '{tool_name}' not explicitly defined. Passing raw input.")
             # Decide how to pass step_input_data if no specific mapping exists
             # Example: parameters['input_data'] = step_input_data

        # --- Call the actual tool via MCP --- 
        try:
            logger.debug(f"Calling tool '{tool_name}' with params: {parameters}")
            tool_result = await mcp.call_tool(tool_name, parameters)
            # Process result to handle potential list format from MCP
            step_output = process_mcp_result(tool_result)
            logger.debug(f"Tool '{tool_name}' returned: {step_output}")
            
            if isinstance(step_output, dict) and step_output.get("error"):
                 raise ToolExecutionError(f"Tool '{tool_name}' failed: {step_output['error']}")
                 
            # Store the successful result
            step_results[output_as] = step_output
            # Update current_input_value for the next step (assuming primary output is desired)
            # This logic might need refinement based on tool outputs
            if isinstance(step_output, dict):
                 current_input_value = step_output.get("text") or step_output.get("summary") or step_output.get("result") or step_output
            else:
                 current_input_value = step_output
                 
            # Accumulate cost if available
            if isinstance(step_output, dict) and "cost" in step_output:
                total_cost += float(step_output["cost"])
                
        except Exception as e:
            logger.error(f"Error executing step '{step_id}' (Tool: {tool_name}): {e}", exc_info=True)
            # Propagate exception to fail the workflow
            raise ToolExecutionError(f"Workflow failed at step '{step_id}': {e}") from e

    # Workflow completed successfully
    processing_time = time.time() - start_time
    logger.success(f"Workflow completed successfully in {processing_time:.2f}s")
    return {
        "outputs": step_results,
        "processing_time": processing_time,
        "total_cost": total_cost,
        "success": True # Indicate overall success
    }

# Enhanced display function for workflow demos
def display_workflow_result(title: str, result: Any):
    """Display workflow result with consistent formatting."""
    console.print(Rule(f"[bold blue]{escape(title)}[/bold blue]"))
    
    # Process result to handle list or dict format
    result = process_mcp_result(result)
    
    # Display outputs if present
    if "outputs" in result and result["outputs"]:
        for output_name, output_text in result["outputs"].items():
            console.print(Panel(
                escape(str(output_text).strip()),
                title=f"[bold magenta]Output: {escape(output_name)}[/bold magenta]",
                border_style="magenta",
                expand=False
            ))
    elif "text" in result:
        # Display single text output if there's no outputs dictionary
        console.print(Panel(
            escape(result["text"].strip()),
            title="[bold magenta]Result[/bold magenta]",
            border_style="magenta",
            expand=False
        ))
    
    # Display execution stats
    _display_stats(result, console)

# Enhanced display function for task analysis
def display_task_analysis(title: str, result: Any):
    """Display task analysis result with consistent formatting."""
    console.print(Rule(f"[bold blue]{escape(title)}[/bold blue]"))
    
    # Process result to handle list or dict format
    result = process_mcp_result(result)
    
    # Display task type and features
    analysis_table = Table(box=box.SIMPLE, show_header=False)
    analysis_table.add_column("Metric", style="cyan")
    analysis_table.add_column("Value", style="white")
    analysis_table.add_row("Task Type", escape(result.get("task_type", "N/A")))
    analysis_table.add_row("Required Features", escape(str(result.get("required_features", []))))
    console.print(analysis_table)
    
    # Display features explanation
    if "features_explanation" in result:
        console.print(Panel(
            escape(result["features_explanation"]),
            title="[bold]Features Explanation[/bold]",
            border_style="dim blue",
            expand=False
        ))
    
    # Display recommendations
    if "recommendations" in result and result["recommendations"]:
        rec_table = Table(title="[bold]Model Recommendations[/bold]", box=box.ROUNDED, show_header=True)
        rec_table.add_column("Provider", style="magenta")
        rec_table.add_column("Model", style="blue")
        rec_table.add_column("Explanation", style="white")
        for rec in result["recommendations"]:
            rec_table.add_row(
                escape(rec.get("provider", "N/A")),
                escape(rec.get("model", "N/A")),
                escape(rec.get("explanation", "N/A"))
            )
        console.print(rec_table)
    
    # Display execution stats
    _display_stats(result, console)

# Move _get_provider_for_model above run_delegate_task_demo
def _get_provider_for_model(model_name: str) -> str:
    """Helper to determine provider from model name."""
    # Accept both 'provider/model' and legacy short names
    model_lower = model_name.lower()
    if '/' in model_lower:
        # e.g., 'gemini/gemini-2.0-flash' or 'anthropic/claude-3-7-sonnet-20250219'
        return model_lower.split('/')[0]
    elif ':' in model_lower:
        return model_lower.split(':')[0]
    elif model_lower.startswith("gpt-"):
        return Provider.OPENAI.value
    elif model_lower.startswith("claude-"):
        return Provider.ANTHROPIC.value
    elif model_lower.startswith("gemini-"):
        return Provider.GEMINI.value
    elif model_lower.startswith("deepseek-"):
        return "deepseek"
    elif model_lower.startswith("grok-"):
        return "grok"
    elif model_lower.startswith("o1-") or model_lower.startswith("o3-"):
        return Provider.OPENAI.value
    else:
        raise ValueError(f"Unknown model prefix for model: {model_name}")

# --- Demo Functions ---

async def run_analyze_task_demo():
    """Demonstrate the analyze_task tool."""
    console.print(Rule("[bold blue]Analyze Task Demo[/bold blue]"))
    logger.info("Running analyze_task demo...", emoji_key="start")
    
    task_description = "Summarize the provided technical document about AI advancements and extract key entities."
    console.print(f"[cyan]Task Description:[/cyan] {escape(task_description)}")
    
    try:
        # Call the real recommend_model tool
        # Need to estimate input/output length for recommend_model
        # Rough estimate for demo purposes
        input_len_chars = len(task_description) * 10 # Assume task needs more context
        output_len_chars = 200 # Estimate output size

        result = await mcp.call_tool("recommend_model", {
            "task_type": "summarization",  # Added to match required argument
            "expected_input_length": input_len_chars,
            "expected_output_length": output_len_chars,
            # Can add other recommend_model params like required_capabilities, max_cost
        })
        
        # Use enhanced display function
        display_task_analysis("Analysis Results", result)
        
    except Exception as e:
        logger.error(f"Error in analyze_task demo: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {escape(str(e))}")
    console.print()


async def run_delegate_task_demo(tracker: CostTracker): # Add tracker
    """Demonstrate the delegate_task tool."""
    console.print(Rule("[bold blue]Delegate Task Demo[/bold blue]"))
    logger.info("Running task delegation demo (using recommend_model + completion)...", emoji_key="start")
    
    task_description = "Generate a short marketing blurb for a new AI-powered writing assistant."
    prompt = "Write a catchy, 2-sentence marketing blurb for 'AI Writer Pro', a tool that helps users write faster and better."
    console.print(f"[cyan]Task Description:[/cyan] {escape(task_description)}")
    console.print(f"[cyan]Prompt:[/cyan] {escape(prompt)}")

    priorities = ["balanced", "cost", "quality"]
    
    for priority in priorities:
        console.print(Rule(f"[yellow]Delegating with Priority: {priority}[/yellow]"))
        logger.info(f"Delegating task with priority: {priority}", emoji_key="processing")
        try:
            # 1. Get recommendation
            recommendation_result_raw = await mcp.call_tool("recommend_model", {
                 "task_type": "creative_writing", # Infer task type
                 "expected_input_length": len(prompt),
                 "expected_output_length": 100, # Estimate blurb length
                 "priority": priority
            })
            recommendation_result = process_mcp_result(recommendation_result_raw)

            if "error" in recommendation_result or not recommendation_result.get("recommendations"):
                 logger.error(f"Could not get recommendation for priority '{priority}'.")
                 console.print(f"[red]Error getting recommendation for '{priority}'.[/red]")
                 continue

            # 2. Execute with recommended model
            top_rec = recommendation_result["recommendations"][0]
            rec_provider = _get_provider_for_model(top_rec["model"])
            rec_model = top_rec["model"]
            logger.info(f"Recommendation for '{priority}': Use {rec_provider}/{rec_model}")

            # Call generate_completion tool
            completion_result_raw = await mcp.call_tool("generate_completion", {
                 "prompt": prompt,
                 "provider": rec_provider,
                 "model": rec_model,
                 "max_tokens": 100
            })

            # Track cost if possible
            completion_result = process_mcp_result(completion_result_raw)
            if isinstance(completion_result, dict) and all(k in completion_result for k in ["cost", "provider", "model"]) and "tokens" in completion_result:
                try:
                    trackable = TrackableResult(
                        cost=completion_result.get("cost", 0.0),
                        input_tokens=completion_result.get("tokens", {}).get("input", 0),
                        output_tokens=completion_result.get("tokens", {}).get("output", 0),
                        provider=completion_result.get("provider", rec_provider), # Use known provider as fallback
                        model=completion_result.get("model", rec_model), # Use known model as fallback
                        processing_time=completion_result.get("processing_time", 0.0)
                    )
                    tracker.add_call(trackable)
                except Exception as track_err:
                    logger.warning(f"Could not track cost for delegated task ({priority}): {track_err}", exc_info=False)

            # Display result
            if "error" in completion_result:
                 logger.error(f"Completion failed for recommended model {rec_model}: {completion_result['error']}")
                 console.print(f"[red]Completion failed for {rec_model}: {completion_result['error']}[/red]")
            else:
                 console.print(Panel(
                     escape(completion_result.get("text", "").strip()),
                     title=f"[bold green]Delegated Result ({escape(priority)} -> {escape(rec_model)})[/bold green]",
                     border_style="green",
                     expand=False
                 ))
                 _display_stats(completion_result, console) # Display stats from completion

        except Exception as e:
            logger.error(f"Error delegating task with priority {priority}: {e}", emoji_key="error", exc_info=True)
            console.print(f"[bold red]Error ({escape(priority)}):[/bold red] {escape(str(e))}")
        console.print()


async def run_workflow_demo():
    """Demonstrate the execute_workflow tool."""
    console.print(Rule("[bold blue]Execute Workflow Demo[/bold blue]"))
    logger.info("Running execute_workflow demo...", emoji_key="start")

    initial_text = """
    Artificial intelligence (AI) is rapidly transforming various sectors. 
    In healthcare, AI algorithms analyze medical images with remarkable accuracy, 
    aiding radiologists like Dr. Evelyn Reed. Pharmaceutical companies, such as InnovatePharma, 
    use AI to accelerate drug discovery. Meanwhile, financial institutions leverage AI 
    for fraud detection and algorithmic trading. The field continues to evolve, 
    driven by researchers like Kenji Tanaka and advancements in machine learning.
    """
    
    workflow = [
        {
            "id": "step1_summarize",
            "operation": "summarize",
            "provider": Provider.ANTHROPIC.value,
            "model": "claude-3-5-haiku-20241022",
            "parameters": {"format": "Provide a 2-sentence summary"},
            "output_as": "summary"
        },
        {
            "id": "step2_extract",
            "operation": "extract_entities",
            "provider": Provider.OPENAI.value,
            "model": "gpt-4.1-mini",
            "parameters": {"entity_types": ["person", "organization", "field"]},
            "input_from": None, # Use initial_input
            "output_as": "entities"
        },
        {
            "id": "step3_questions",
            "operation": "generate_questions",
            "provider": Provider.GEMINI.value,
            "model": "gemini-2.0-flash-lite",
            "parameters": {"question_count": 2, "question_type": "insightful"},
            "input_from": "summary", # Use output from step 1
            "output_as": "questions"
        }
    ]
    
    console.print("[cyan]Initial Input Text:[/cyan]")
    console.print(Panel(escape(initial_text.strip()), border_style="dim blue", expand=False))
    console.print("[cyan]Workflow Definition:[/cyan]")
    try:
        workflow_json = json.dumps(workflow, indent=2, default=lambda o: o.value if isinstance(o, Provider) else str(o)) # Handle enum serialization
        console.print(Panel(
            Syntax(workflow_json, "json", theme="default", line_numbers=True, word_wrap=True),
            title="[bold]Workflow Steps[/bold]",
            border_style="blue",
            expand=False
        ))
    except Exception as json_err:
         console.print(f"[red]Could not display workflow definition: {escape(str(json_err))}[/red]")
    
    logger.info(f"Executing workflow with {len(workflow)} steps...", emoji_key="processing")
    try:
        result = await mcp.call_tool("execute_workflow", {
            "workflow_steps": workflow, 
            "initial_input": initial_text
        })
        
        # Use enhanced display function
        display_workflow_result("Workflow Results", result)

    except Exception as e:
        logger.error(f"Error executing workflow: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Workflow Execution Error:[/bold red] {escape(str(e))}")
    console.print()


async def run_prompt_optimization_demo():
    """Demonstrate the optimize_prompt tool."""
    console.print(Rule("[bold blue]Prompt Optimization Demo[/bold blue]"))
    logger.info("Running optimize_prompt demo...", emoji_key="start")

    original_prompt = "Tell me about Large Language Models."
    target_model = "claude-3-opus-20240229"
    optimization_type = "detailed_response" # e.g., conciseness, detailed_response, specific_format
    
    console.print(f"[cyan]Original Prompt:[/cyan] {escape(original_prompt)}")
    console.print(f"[cyan]Target Model:[/cyan] {escape(target_model)}")
    console.print(f"[cyan]Optimization Type:[/cyan] {escape(optimization_type)}")
    
    logger.info(f"Optimizing prompt for {target_model}...", emoji_key="processing")
    try:
        result = await mcp.call_tool("optimize_prompt", {
            "prompt": original_prompt,
            "target_model": target_model,
            "optimization_type": optimization_type,
            "provider": Provider.OPENAI.value # Using OpenAI to optimize for Claude
        })
        
        # Process result to handle list or dict format
        result = process_mcp_result(result)
        
        # Get optimized prompt text
        optimized_prompt = result.get("optimized_prompt", "")
        if not optimized_prompt and hasattr(result, 'text'):
            optimized_prompt = result.text
        
        console.print(Panel(
            escape(optimized_prompt.strip() if optimized_prompt else "[red]Optimization failed[/red]"),
            title="[bold green]Optimized Prompt[/bold green]",
            border_style="green",
            expand=False
        ))
        
        # Display execution stats
        _display_stats(result, console)
        
    except Exception as e:
        logger.error(f"Error optimizing prompt: {e}", emoji_key="error", exc_info=True)
        console.print(f"[bold red]Prompt Optimization Error:[/bold red] {escape(str(e))}")
    console.print()


async def main():
    """Run workflow delegation examples."""
    console.print(Rule("[bold magenta]Workflow Delegation Demo Suite[/bold magenta]"))
    tracker = CostTracker() # Instantiate tracker
    
    try:
        # Setup providers first
        await initialize_providers() # Ensure keys are checked/providers ready
        console.print(Rule("[bold magenta]Workflow & Delegation Demos Starting[/bold magenta]"))
        
        # --- Register Necessary Tools --- 
        # Ensure tools called by demos are registered on the MCP instance
        from ultimate_mcp_server.tools.completion import generate_completion
        from ultimate_mcp_server.tools.document import (
            extract_entities,
            generate_qa_pairs,
            summarize_document,
        )
        from ultimate_mcp_server.tools.optimization import recommend_model
        
        mcp.tool()(recommend_model)
        mcp.tool()(generate_completion)
        mcp.tool()(summarize_document)
        mcp.tool()(extract_entities)
        mcp.tool()(generate_qa_pairs)
        logger.info("Manually registered recommend_model, completion, and document tools.")
        # --------------------------------

        await run_analyze_task_demo()
        
        # Pass tracker only to delegate demo
        await run_delegate_task_demo(tracker)
        
        await run_workflow_demo()
        # await run_prompt_optimization_demo() # Add back if needed
        
        # Display final cost summary
        tracker.display_summary(console)

        logger.success("Workflow Delegation Demo Finished Successfully!", emoji_key="complete")
        console.print(Rule("[bold magenta]Workflow Delegation Demos Complete[/bold magenta]"))
        return 0

    except Exception as e:
        logger.critical(f"Workflow demo failed: {str(e)}", emoji_key="critical", exc_info=True)
        console.print(f"[bold red]Critical Demo Error:[/bold red] {escape(str(e))}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 