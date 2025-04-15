"""
Utility functions for tournament functionality.
"""
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from llm_gateway.core.models.tournament import TournamentData

# Add imports for harmonized tools
from llm_gateway.tools.extraction import extract_code_from_response
from llm_gateway.tools.filesystem import write_file

logger = logging.getLogger(__name__)

def create_round_prompt(tournament: TournamentData, round_num: int) -> str:
    """
    Creates the prompt for a specific round based on the tournament state and type.
    """
    if round_num == 0:
        # Initial round always uses the original prompt
        return tournament.config.prompt

    prev_round_num = round_num - 1
    if prev_round_num < 0 or prev_round_num >= len(tournament.rounds_results):
        raise ValueError(f"Invalid round number {round_num} or missing previous round data.")

    prev_round_result = tournament.rounds_results[prev_round_num]
    if not prev_round_result.responses:
         return f"Error: No responses found for previous round {prev_round_num}. Using original prompt:\n\n{tournament.config.prompt}"

    # --- Base prompt structure --- 
    base_prompt = f"""Here is the original request/problem:

---
{tournament.config.prompt}
---

Here are the responses generated by different LLMs in the previous round ({prev_round_num}):

"""
    # --- Add previous round responses --- 
    response_texts = []
    for model_id, response_data in prev_round_result.responses.items():
        # Use raw response text for both code and text tournaments in the prompt
        content_to_include = response_data.response_text or "[No content available]"
        display_model_id = model_id.split(':')[-1] if ':' in model_id else model_id
        
        response_texts.append(f"-- Model: {display_model_id} --\n{content_to_include.strip()}\n")
        
    base_prompt += "\n".join(response_texts)
    base_prompt += "\n" # Add separator
    
    # --- Add type-specific instructions --- 
    if tournament.config.tournament_type == "code":
        combined_prompt = base_prompt + """Analyze each code solution carefully. Consider:
1. Correctness - Does the code work as intended? Handle edge cases?
2. Efficiency - Is the code optimized?
3. Readability - Is it clear and maintainable?
4. Robustness - Does it handle errors gracefully?
5. Innovation - Does it incorporate the best ideas from others?

Choose the best ideas and elements from ALL solutions to the extent they are complementary. 
Create a NEW, complete Python implementation based on your analysis. 

Provide only the code for your improved solution, enclosed in triple backticks (```python ... ```).
Do not include explanations before or after the code block unless specifically part of the code comments.
Improved Python Code:
"""
    elif tournament.config.tournament_type == "text":
        combined_prompt = base_prompt + """Analyze each response carefully based on the original request. Consider:
1. Relevance - Does the response directly address the original request?
2. Accuracy - Is the information provided correct?
3. Completeness - Does the response cover the key aspects requested?
4. Clarity - Is the response easy to understand?
5. Conciseness - Is the response to the point?
6. Style/Tone - Is the style appropriate for the request?

Synthesize the best aspects of ALL responses into a single, improved response.
Aim to create a definitive answer that is better than any individual response from the previous round.

Start your response with a brief (1-2 sentence) explanation of the key improvements or synthesis choices you made, enclosed in <thinking> tags. 
Then, provide only the improved text response itself.

Example:
<thinking>I combined the detailed explanation from Model A with the concise summary from Model B for better clarity.</thinking>
The refined answer goes here...

Improved Response (including <thinking> block):
"""
    else:
        # Default or fallback if type is unknown (or add more types)
        logger.warning(f"Unknown tournament type: {tournament.config.tournament_type}. Using generic refinement prompt.")
        combined_prompt = base_prompt + """Analyze the previous responses based on the original request. 
Synthesize the best aspects into an improved response. Provide only the improved response.
Improved Response:
"""

    return combined_prompt.strip()

async def extract_thinking(response_text: str) -> Optional[str]:
    """Extract the thinking/reasoning process from a model response if present.
    
    Uses the standardized code extraction tool with fallbacks to simpler pattern matching.
    
    Args:
        response_text: The model's response text.
        
    Returns:
        The extracted thinking process, or None if not found.
    """
    # First try using the standard extraction tool - handles many formats
    try:
        extracted = await extract_code_from_response(
            response_text=response_text,
            language_hint="thinking", # Signal we're looking for reasoning/thinking blocks
            timeout=10  # Shorter timeout for thinking extraction
        )
        if extracted and extracted.strip():
            return extracted.strip()
    except Exception as e:
        # Log but don't fail - fall back to regex patterns
        logger.debug(f"Error using standard extraction tool for thinking: {e}. Falling back to regex.")
    
    # Fall back to regex patterns for compatibility
    thinking_patterns = [
        (r'(?i)# *(thinking|reasoning).*?\n(.*?)(?=\n# |$)', 2),  # Markdown-style heading
        (r'(?i)<(?:thinking|reasoning)>\s*(.*?)\s*</(?:thinking|reasoning)>', 1),  # XML-style
        (r'(?i)(?:thinking|reasoning)(?:: | process:)\s*(.*?)(?:\n\n|$)', 1),  # Simple prefixed line
    ]
    
    for pattern, group in thinking_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            return match.group(group).strip()
    
    return None

def calculate_code_metrics(code: Optional[str]) -> dict:
    """
    Calculates basic metrics about a code string.
    """
    if not code:
        return {
            "code_lines": 0,
            "code_size_kb": 0.0,
            "function_count": 0,
            "class_count": 0,
            "import_count": 0,
        }

    code_lines = code.count('\n') + 1
    code_size_bytes = len(code.encode('utf-8'))
    code_size_kb = round(code_size_bytes / 1024, 2)
    function_count = len(re.findall(r'\bdef\s+\w+', code))
    class_count = len(re.findall(r'\bclass\s+\w+', code))
    import_count = len(re.findall(r'^import\s+|\bfrom\s+', code, re.MULTILINE))

    return {
        "code_lines": code_lines,
        "code_size_kb": code_size_kb,
        "function_count": function_count,
        "class_count": class_count,
        "import_count": import_count,
    }

def generate_comparison_file(tournament: TournamentData, round_num: int) -> Optional[str]:
    """Generate a markdown comparison file for the given round.
    
    Args:
        tournament: The tournament data.
        round_num: The round number to generate the comparison for.
        
    Returns:
        The markdown content string, or None if data is missing.
    """
    if round_num < 0 or round_num >= len(tournament.rounds_results):
        logger.warning(f"Cannot generate comparison for invalid round {round_num}")
        return None

    round_result = tournament.rounds_results[round_num]
    if not round_result.responses:
        logger.warning(f"Cannot generate comparison for round {round_num}, no responses found.")
        return None
        
    previous_round = tournament.rounds_results[round_num - 1] if round_num > 0 else None
    is_code_tournament = tournament.config.tournament_type == "code"

    # Start with a comprehensive header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comparison_content = f"# Tournament Comparison - Round {round_num}\n\n"
    comparison_content += f"**Generated:** {timestamp}\n"
    comparison_content += f"**Tournament ID:** {tournament.tournament_id}\n"
    comparison_content += f"**Tournament Name:** {tournament.config.name}\n"
    comparison_content += f"**Type:** {tournament.config.tournament_type}\n"
    comparison_content += f"**Current Round:** {round_num} of {tournament.config.rounds}\n"
    comparison_content += f"**Models:** {', '.join(model.model_id for model in tournament.config.models)}\n\n"
    
    # Add original prompt section
    if round_num == 0:
        comparison_content += f"## Original Prompt\n\n```\n{tournament.config.prompt}\n```\n\n"
    else:
        # For later rounds, show what was provided to the models
        comparison_content += f"## Round {round_num} Prompt\n\n"
        # Get a sample prompt - all models get the same prompt in a round
        sample_prompt = create_round_prompt(tournament, round_num)
        comparison_content += f"```\n{sample_prompt[:500]}...\n```\n\n"
    
    # Summarize overall metrics
    comparison_content += "## Summary Metrics\n\n"
    comparison_content += "| Model | Tokens In | Tokens Out | Cost | Latency (ms) |\n"
    comparison_content += "|-------|-----------|------------|------|-------------|\n"
    
    for model_id, response_data in sorted(round_result.responses.items()):
        metrics = response_data.metrics
        tokens_in = metrics.get("input_tokens", "N/A")
        tokens_out = metrics.get("output_tokens", "N/A")
        cost = metrics.get("cost", "N/A")
        latency = metrics.get("latency_ms", "N/A")
        
        display_model_id = model_id.split(':')[-1] if ':' in model_id else model_id
        cost_display = f"${cost:.6f}" if isinstance(cost, (int, float)) else cost
        
        comparison_content += f"| {display_model_id} | {tokens_in} | {tokens_out} | {cost_display} | {latency} |\n"
    
    comparison_content += "\n## Detailed Model Responses\n\n"

    for model_id, response_data in sorted(round_result.responses.items()):
        metrics = response_data.metrics
        display_model_id = model_id.split(':')[-1] if ':' in model_id else model_id
        
        comparison_content += f"### {display_model_id}\n\n"
        
        # Display detailed metrics as a subsection
        comparison_content += "#### Metrics\n\n"
        tokens_in = metrics.get("input_tokens", "N/A")
        tokens_out = metrics.get("output_tokens", "N/A")
        total_tokens = metrics.get("total_tokens", "N/A")
        cost = metrics.get("cost", "N/A")
        latency = metrics.get("latency_ms", "N/A")
        
        comparison_content += f"- **Tokens:** {tokens_in} in, {tokens_out} out, {total_tokens} total\n"
        if isinstance(cost, (int, float)):
            comparison_content += f"- **Cost:** ${cost:.6f}\n"
        else:
            comparison_content += f"- **Cost:** {cost}\n"
        comparison_content += f"- **Latency:** {latency}ms\n"
        
        # Code-specific metrics
        if is_code_tournament:
            code_lines = metrics.get("code_lines", "N/A")
            code_size = metrics.get("code_size_kb", "N/A")
            comparison_content += f"- **Code Stats:** {code_lines} lines, {code_size} KB\n"
        
        comparison_content += "\n"
        
        # Display thinking process if available
        if response_data.thinking_process:
            comparison_content += "#### Thinking Process\n\n"
            comparison_content += f"```\n{response_data.thinking_process}\n```\n\n"

        # Display response content
        if is_code_tournament:
            comparison_content += "#### Extracted Code\n\n"
            comparison_content += "```python\n"
            comparison_content += response_data.extracted_code or "# No code extracted"
            comparison_content += "\n```\n\n"
        else:
            # For text tournaments, display the raw response
            comparison_content += "#### Response Text\n\n"
            comparison_content += "```\n"
            comparison_content += response_data.response_text or "[No response text]"
            comparison_content += "\n```\n\n"
            
        # Add link to the full response file
        if response_data.response_file_path:
            comparison_content += f"[View full response file]({response_data.response_file_path})\n\n"
            
    # Add a section comparing changes from previous round if this isn't round 0
    if previous_round and previous_round.responses:
        comparison_content += "## Changes from Previous Round\n\n"
        for model_id, response_data in sorted(round_result.responses.items()):
            if model_id in previous_round.responses:
                display_model_id = model_id.split(':')[-1] if ':' in model_id else model_id
                comparison_content += f"### {display_model_id}\n\n"
                
                # Compare metrics
                current_metrics = response_data.metrics
                previous_metrics = previous_round.responses[model_id].metrics
                
                current_tokens_out = current_metrics.get("output_tokens", 0)
                previous_tokens_out = previous_metrics.get("output_tokens", 0)
                token_change = current_tokens_out - previous_tokens_out if isinstance(current_tokens_out, (int, float)) and isinstance(previous_tokens_out, (int, float)) else "N/A"
                
                comparison_content += f"- **Token Change:** {token_change} tokens\n"
                
                # Note: Here you could add more sophisticated text comparison/diff
                comparison_content += "- Review the full responses to see detailed changes\n\n"
    
    return comparison_content.strip() 

async def save_model_response(
    tournament: TournamentData, 
    round_num: int, 
    model_id: str, 
    response_text: str,
    thinking: Optional[str] = None,
    timestamp: Optional[str] = None
) -> str:
    """Save model response to a file using standardized filesystem tools.
    
    Args:
        tournament: Tournament data
        round_num: Round number
        model_id: Model ID that generated this response
        response_text: The text response to save
        thinking: Optional thinking process from the model
        timestamp: Optional timestamp (defaults to current time if not provided)
        
    Returns:
        Path to saved response file
    """
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get path to tournament storage directory
    storage_dir = Path(tournament.storage_path)
    round_dir = storage_dir / f"round_{round_num}"
    round_dir.mkdir(exist_ok=True)
    
    # Create a safe filename from model ID
    safe_model_id = model_id.replace(":", "_").replace("/", "_")
    response_file = round_dir / f"{safe_model_id}_response.md"
    
    # Construct the markdown file with basic metadata header
    content = f"""# Response from {model_id}

## Metadata
- Tournament: {tournament.name}
- Round: {round_num}
- Model: {model_id}
- Timestamp: {timestamp}

## Response:

{response_text}
"""

    # Add thinking process if available
    if thinking:
        content += f"\n\n## Thinking Process:\n\n{thinking}\n"
    
    # Use the standard filesystem write tool
    try:
        # Properly use the async write_file tool
        result = await write_file(
            path=str(response_file),
            content=content
        )
        
        if not result.get("success", False):
            logger.warning(f"Standard write_file tool reported failure: {result.get('error')}")
            # Fall back to direct write
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(content)
    except Exception as e:
        logger.error(f"Error using standardized file writer: {e}. Using direct file write.")
        # Fall back to direct write in case of errors
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return str(response_file)

def get_round_dir(tournament: TournamentData, round_num: int) -> Path:
    """Get the directory path for a specific tournament round.
    
    Args:
        tournament: The tournament data.
        round_num: The round number.
        
    Returns:
        Path to the round directory.
    """
    tournament_dir = Path(tournament.storage_path)
    round_dir = tournament_dir / f"round_{round_num}"
    return round_dir 

def get_word_count(text: str) -> int:
    """Get the word count of a text string.
    
    Args:
        text: The text to count words in.
        
    Returns:
        The number of words.
    """
    if not text:
        return 0
    return len(text.split())

def generate_synthesis_prompt(tournament: TournamentData, previous_responses: Dict[str, str]) -> str:
    """Generate the prompt for the synthesis round.
    
    Args:
        tournament: The tournament data
        previous_responses: A dictionary mapping model IDs to their responses
        
    Returns:
        The synthesis prompt for the next round.
    """
    # Letter used for referring to models to avoid bias
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    
    # Start with a base prompt instructing the model what to do
    prompt = f"""# {tournament.name} - Synthesis Round

Your task is to create an improved version based on the responses from multiple models.

Original task:
{tournament.config.prompt}

Below are responses from different models. Review them and create a superior response 
that combines the strengths of each model's approach while addressing any weaknesses.

"""
    
    # Add each model's response
    for i, (model_id, response) in enumerate(previous_responses.items()):
        if i < len(letters):
            letter = letters[i]
            model_name = model_id.split(":")[-1] if ":" in model_id else model_id
            
            prompt += f"""
## Model {letter} ({model_name}) Response:

{response}

"""
    
    # Add synthesis instructions
    prompt += """
# Your Task

Based on the responses above:

1. Create a single, unified response that represents the best synthesis of the information
2. Incorporate the strengths of each model's approach
3. Improve upon any weaknesses or omissions
4. Your response should be more comprehensive, accurate, and well-structured than any individual response

## Thinking Process
Start by briefly analyzing the strengths and weaknesses of each model's response, then explain your synthesis approach.

Example: "I synthesized the structured approach of Model A with the comprehensive detail from Model B, ensuring..."

"""
    
    return prompt 