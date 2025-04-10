#!/usr/bin/env python3
"""
Test script for the LLM-based code extraction function.

This script loads the tournament state from a previous run and tests
the new code extraction function against the raw response texts.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.panel import Panel
from rich.table import Table

from llm_gateway.core.server import Gateway

# Import the extraction function from the library
from llm_gateway.tools import extract_code_from_response
from llm_gateway.utils import get_logger
from llm_gateway.utils.logging.console import console

# Initialize logger
logger = get_logger("example.test_extraction")

# Initialize global gateway
gateway = None

# Path to the tournament state file from the last run
TOURNAMENT_STATE_PATH = "/home/ubuntu/llm_gateway_mcp_server/storage/tournaments/2025-04-01_03-24-37_tournament_76009a9a/tournament_state.json"

async def setup_gateway():
    """Set up the gateway for testing."""
    global gateway
    
    # Create gateway instance
    logger.info("Initializing gateway for testing", emoji_key="start")
    gateway = Gateway("test-extraction")
    
    # Initialize the server with all providers and built-in tools
    await gateway._initialize_providers()
    
    logger.info("Gateway initialized", emoji_key="success")

async def load_tournament_state() -> Dict:
    """Load the tournament state from the previous run."""
    try:
        with open(TOURNAMENT_STATE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading tournament state: {str(e)}", emoji_key="error")
        return {}

async def test_extraction():
    """Test the LLM-based code extraction function."""
    # Load the tournament state
    tournament_state = await load_tournament_state()
    
    if not tournament_state:
        logger.error("Failed to load tournament state", emoji_key="error")
        return 1
    
    # Check if we have rounds_results
    rounds_results = tournament_state.get('rounds_results', [])
    if not rounds_results:
        logger.error("No round results found in tournament state", emoji_key="error")
        return 1
    
    # Create a table to display the results
    console.print("\n[bold]Testing LLM-based Code Extraction Function[/bold]\n")
    
    # Create a table for extraction results
    extraction_table = Table(box=box.MINIMAL, show_header=True, expand=False)
    extraction_table.add_column("Round", style="cyan")
    extraction_table.add_column("Model", style="magenta")
    extraction_table.add_column("Code Extracted", style="green")
    extraction_table.add_column("Line Count", style="yellow", justify="right")
    
    # Process each round
    for round_idx, round_data in enumerate(rounds_results):
        responses = round_data.get('responses', {})
        
        for model_id, response in responses.items():
            display_model = model_id.split(':')[-1] if ':' in model_id else model_id
            response_text = response.get('response_text', '')
            
            if response_text:
                # Extract code using our new function
                extracted_code = await extract_code_from_response(response_text)
                
                # Calculate line count
                line_count = len(extracted_code.split('\n')) if extracted_code else 0
                
                # Add to the table
                extraction_table.add_row(
                    str(round_idx),
                    display_model,
                    "✅" if extracted_code else "❌",
                    str(line_count)
                )
                
                # Print detailed results
                if extracted_code:
                    console.print(Panel(
                        f"[bold]Round {round_idx} - {display_model}[/bold]\n\n"
                        f"[green]Successfully extracted {line_count} lines of code[/green]\n",
                        title="Extraction Result",
                        expand=False
                    ))
                    
                    # Print first 10 lines of code as a preview
                    code_preview = "\n".join(extracted_code.split('\n')[:10])
                    if len(extracted_code.split('\n')) > 10:
                        code_preview += "\n..."
                    
                    console.print(Panel(
                        code_preview,
                        title="Code Preview",
                        expand=False
                    ))
                else:
                    console.print(Panel(
                        f"[bold]Round {round_idx} - {display_model}[/bold]\n\n"
                        f"[red]Failed to extract code[/red]\n",
                        title="Extraction Result",
                        expand=False
                    ))
    
    # Display the summary table
    console.print("\n[bold]Extraction Summary:[/bold]")
    console.print(extraction_table)
    
    return 0

async def main():
    """Run the test script."""
    try:
        # Set up gateway
        await setup_gateway()
        
        # Run the extraction test
        return await test_extraction()
    except Exception as e:
        logger.critical(f"Test failed: {str(e)}", emoji_key="critical", exc_info=True)
        return 1
    finally:
        # Clean up
        if gateway:
            pass  # No cleanup needed for Gateway instance

if __name__ == "__main__":
    # Run the script
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 