# --- tools/single_shot_synthesis.py (NEW) ---
import asyncio
import json
import random
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from ultimate_mcp_server.exceptions import ToolError
from pydantic import ValidationError

from ultimate_mcp_server.core.models.tournament import (  # Reusing models from tournament where appropriate
    SingleShotGeneratorModelConfig,
    SingleShotIndividualResponse,
    SingleShotSynthesisInput,
    SingleShotSynthesisOutput,
)
from ultimate_mcp_server.core.tournaments.utils import extract_thinking
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.tools.completion import generate_completion
from ultimate_mcp_server.tools.extraction import extract_code_from_response
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.single_shot_synthesis")

STORAGE_DIR_BASE = Path(__file__).resolve().parent.parent.parent / "storage" / "single_shot_synthesis"
STORAGE_DIR_BASE.mkdir(parents=True, exist_ok=True)

def _get_single_shot_storage_path(name: str, request_id: str) -> Path:
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
    safe_name = re.sub(r'[-\s]+', '-', safe_name)[:50]
    uuid_suffix = request_id.split('-')[0]
    folder_name = f"{timestamp_str}_{safe_name}_{uuid_suffix}"
    path = STORAGE_DIR_BASE / folder_name
    path.mkdir(parents=True, exist_ok=True)
    return path

async def _generate_expert_response(
    prompt: str,
    config: SingleShotGeneratorModelConfig,
    global_retry_config: Optional[Dict] = None # e.g. {"max_retries": 2, "backoff_base": 1.0}
) -> SingleShotIndividualResponse:
    
    start_time = time.monotonic()
    response_data = SingleShotIndividualResponse(model_id=config.model_id)
    
    # Simple retry logic here, could be more sophisticated or rely on generate_completion
    max_retries = global_retry_config.get("max_retries", 1) if global_retry_config else 1
    backoff_base = global_retry_config.get("backoff_base", 1.0) if global_retry_config else 1.0
    
    # Determine provider and apply specific logic
    derived_provider = None
    if '/' in config.model_id:
        provider_prefix = config.model_id.split('/')[0].lower()
        if provider_prefix == "google":
            derived_provider = "gemini"  # Map 'google' to 'gemini'
        elif provider_prefix == "anthropic":
            derived_provider = "anthropic"
        # Add other explicit mappings here if needed in the future
        else:
            derived_provider = provider_prefix # Default to the prefix as is
    
    current_max_tokens = config.max_tokens
    if derived_provider == "anthropic" and current_max_tokens is None:
        logger.info(f"Anthropic model {config.model_id} called without max_tokens, defaulting to 2048 for timeout calculation.")
        current_max_tokens = 2048 # Default for Anthropic if not specified to prevent TypeError

    for attempt in range(max_retries + 1):
        try:
            # provider was originally derived inside the loop, now passed explicitly
            completion_result = await generate_completion(
                prompt=prompt,
                model=config.model_id,
                provider=derived_provider, # Use the determined/mapped provider
                temperature=config.temperature,
                max_tokens=current_max_tokens, # Use the potentially defaulted max_tokens
                # TODO: Add seed if SingleShotGeneratorModelConfig includes it
            )

            response_data.metrics["cost"] = completion_result.get("cost", 0.0)
            response_data.metrics["input_tokens"] = completion_result.get("tokens", {}).get("input")
            response_data.metrics["output_tokens"] = completion_result.get("tokens", {}).get("output")
            response_data.metrics["api_latency_ms"] = int(completion_result.get("processing_time", 0) * 1000)
            response_data.metrics["api_model_id_used"] = completion_result.get("model", config.model_id)

            if completion_result.get("success"):
                response_data.response_text = completion_result.get("text")
                break # Success
            else:
                response_data.error = completion_result.get("error", f"Generation failed on attempt {attempt+1}")
                if attempt == max_retries: # Last attempt
                    logger.error(f"Expert {config.model_id} failed after {max_retries+1} attempts: {response_data.error}")

        except Exception as e:
            logger.error(f"Exception during expert call {config.model_id} (attempt {attempt+1}): {e}", exc_info=True)
            response_data.error = str(e)
            if attempt == max_retries:
                logger.error(f"Expert {config.model_id} failed with exception after {max_retries+1} attempts.")
        
        if attempt < max_retries:
            sleep_duration = random.uniform(backoff_base, backoff_base * 1.5) * (2 ** attempt)
            sleep_duration = min(sleep_duration, 15.0) # Max 15s sleep
            logger.info(f"Expert {config.model_id} attempt {attempt+1} failed. Retrying in {sleep_duration:.2f}s...")
            await asyncio.sleep(sleep_duration)
            
    response_data.metrics["total_task_time_ms"] = int((time.monotonic() - start_time) * 1000)
    return response_data


@with_tool_metrics # This decorator will add its own overall metrics
@with_error_handling
async def single_shot_synthesis(
    name: str,
    prompt: str,
    expert_models: List[Dict[str, Any]], # CHANGED from expert_model_configs
    synthesizer_model: Dict[str, Any],   # CHANGED from synthesizer_model_config
    tournament_type: Literal["code", "text"] = "text",
    synthesis_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs a single-shot, multi-model synthesis:
    1. Sends the prompt to multiple "expert" LLMs in parallel.
    2. Collects their responses.
    3. Feeds the original prompt and all expert responses to a "synthesizer" LLM.
    4. The synthesizer LLM produces a final, fused response.
    Useful for quick brainstorming and merging diverse perspectives.

    Args:
        name: A descriptive name for this synthesis task (e.g., "Product Description Brainstorm").
        prompt: The initial challenge prompt or question for all expert models.
        expert_models: A list of configurations for the "expert" models. Each config is a dict:
            - model_id (str, required): e.g., 'openai/gpt-3.5-turbo'.
            - temperature (float, optional): Model-specific temperature.
            - max_tokens (int, optional): Model-specific max tokens.
        synthesizer_model: Configuration for the "synthesizer" model. Dict fields:
            - model_id (str, required, default 'anthropic/claude-3-7-sonnet-20250219'): e.g., 'google/gemini-1.5-pro-latest'.
            - temperature (float, optional): Synthesizer-specific temperature.
            - max_tokens (int, optional): Synthesizer-specific max tokens.
            - system_prompt (str, optional): System prompt for the synthesizer.
        tournament_type: 'code' or 'text'. Influences synthesis instructions and output processing (default 'text').
        synthesis_instructions: Custom instructions for the synthesizer. If None, default instructions are used.

    Returns:
        A dictionary containing the request_id, status, individual expert responses,
        the synthesized response, metrics, and storage path for artifacts.
    """
    task_start_time = time.monotonic()
    request_id = str(uuid.uuid4())
    storage_path = _get_single_shot_storage_path(name, request_id)
    
    output_data = SingleShotSynthesisOutput(
        request_id=request_id,
        name=name,
        status="FAILED", # Default to FAILED, update on success
        expert_responses=[],
        storage_path=str(storage_path)
    )

    try:
        # Validate inputs using Pydantic model with aliases
        validated_input = SingleShotSynthesisInput(
            name=name,
            prompt=prompt,
            expert_models=expert_models, # Pass using alias
            synthesizer_model=synthesizer_model, # Pass using alias
            tournament_type=tournament_type,
            synthesis_instructions=synthesis_instructions
        )
        parsed_expert_configs = validated_input.expert_model_configs
        parsed_synthesizer_config = validated_input.synthesizer_model_config
        retry_cfg_experts = {"max_retries": 1, "backoff_base": 1.0} 

    except ValidationError as e:
        output_data.error_message = f"Input validation error: {e.errors()}"
        output_data.total_metrics["total_task_time_ms"] = int((time.monotonic() - task_start_time) * 1000)
        logger.error(f"SingleShotSynthesis input validation error for '{name}': {e.json(indent=2)}")
        raise ToolError(f"Invalid input for single_shot_synthesis: {e.errors()}", status_code=400) from e


    # 1. Parallel fan-out to expert models
    logger.info(f"[{request_id}] Starting expert model responses for '{name}'. Count: {len(parsed_expert_configs)}")
    expert_tasks = [
        _generate_expert_response(prompt, config, retry_cfg_experts) for config in parsed_expert_configs
    ]
    output_data.expert_responses = await asyncio.gather(*expert_tasks, return_exceptions=False) # Exceptions handled in _generate_expert_response

    # Persist expert responses
    for i, resp in enumerate(output_data.expert_responses):
        expert_file_name = f"expert_{i+1}_{re.sub(r'[^a-zA-Z0-9_-]', '_', resp.model_id)}.md"
        expert_file_path = storage_path / expert_file_name
        content = f"# Expert Response: {resp.model_id}\n\n"
        content += f"## Metrics\nCost: ${resp.metrics.get('cost',0):.6f}\nLatency: {resp.metrics.get('api_latency_ms','N/A')}ms\n\n"
        if resp.error:
            content += f"## Error\n```\n{resp.error}\n```\n"
        if resp.response_text:
            content += f"## Response Text\n```\n{resp.response_text}\n```\n"
        expert_file_path.write_text(content, encoding='utf-8')

    # 2. Aggregation prompt builder
    agg_prompt_parts = [f"You are an advanced AI tasked with synthesizing a definitive answer based on multiple expert inputs.\n\nOriginal Problem/Prompt:\n---\n{prompt}\n---\n"]
    agg_prompt_parts.append("Below are responses from several expert models. Review them critically:\n")

    for i, resp in enumerate(output_data.expert_responses):
        agg_prompt_parts.append(f"-- Response from Expert Model {i+1} ({resp.model_id}) --")
        if resp.response_text and not resp.error:
            agg_prompt_parts.append(resp.response_text)
        elif resp.error:
            agg_prompt_parts.append(f"[This model encountered an error: {resp.error}]")
        else:
            agg_prompt_parts.append("[This model provided no content.]")
        agg_prompt_parts.append("-------------------------------------\n")
    
    # Synthesis Instructions
    if synthesis_instructions:
        agg_prompt_parts.append(f"\nSynthesis Instructions:\n{synthesis_instructions}\n")
    else: # Default instructions
        default_instr = "Your Task:\n1. Identify unique insights, complementary information, and key arguments from each expert response.\n"
        default_instr += "2. Resolve any contradictions or discrepancies, prioritizing verifiable facts and logical consistency.\n"
        default_instr += "3. Produce a single, coherent, and comprehensive response that is strictly superior to any individual expert response.\n"
        if tournament_type == "code":
            default_instr += "4. If the task involves code, provide ONLY the complete, final code block (e.g., ```python ... ```). Do not include explanations outside of code comments.\n"
        else: # Text
            default_instr += "4. You MAY optionally begin your output with a brief <thinking>...</thinking> block explaining your synthesis strategy, then provide the final synthesized text.\n"
        default_instr += "\n### Final Synthesized Answer:\n"
        agg_prompt_parts.append(default_instr)
        
    aggregation_prompt = "\n".join(agg_prompt_parts)
    (storage_path / "synthesis_prompt.md").write_text(aggregation_prompt, encoding='utf-8')


    # 3. Fan-in call to synthesizer_id
    logger.info(f"[{request_id}] Requesting synthesis from {parsed_synthesizer_config.model_id} for '{name}'.")
    synthesizer_success = False
    try:
        # Simple retry for synthesizer
        retry_cfg_synth = {"max_retries": 1, "backoff_base": 2.0} 
        synth_response_raw = await _generate_expert_response( # Reuse for basic call structure
            aggregation_prompt,
            SingleShotGeneratorModelConfig( # Adapt to expected input
                model_id=parsed_synthesizer_config.model_id,
                temperature=parsed_synthesizer_config.temperature,
                max_tokens=parsed_synthesizer_config.max_tokens
            ),
            retry_cfg_synth
        )
        output_data.synthesizer_metrics = synth_response_raw.metrics
        
        if synth_response_raw.response_text and not synth_response_raw.error:
            output_data.synthesized_response_text = synth_response_raw.response_text
            output_data.synthesizer_thinking_process = await extract_thinking(output_data.synthesized_response_text)
            if output_data.synthesizer_thinking_process and output_data.synthesized_response_text:
                 # Remove thinking block from main response if present
                 output_data.synthesized_response_text = output_data.synthesized_response_text.replace(f"<thinking>{output_data.synthesizer_thinking_process}</thinking>", "").strip()

            if tournament_type == "code":
                # Extraction model can be fixed or configurable for this tool
                extraction_model_id = "anthropic/claude-3-5-haiku-20241022" # Example default
                code_extraction_result = await extract_code_from_response(
                    response_text=output_data.synthesized_response_text,
                    language_hint="python", # Assuming Python for now
                    extraction_model_id=extraction_model_id
                )
                if code_extraction_result.get("success"):
                    output_data.synthesized_extracted_code = code_extraction_result.get("code_block")
                else: # Log if extraction failed but don't mark whole thing as failure
                    logger.warning(f"[{request_id}] Code extraction from synthesizer output failed: {code_extraction_result.get('error')}")
            synthesizer_success = True
        else:
            output_data.error_message = f"Synthesizer ({parsed_synthesizer_config.model_id}) failed: {synth_response_raw.error or 'No response text'}"
            logger.error(f"[{request_id}] {output_data.error_message}")
            
    except Exception as e:
        output_data.error_message = f"Exception during synthesis call: {str(e)}"
        logger.error(f"[{request_id}] {output_data.error_message}", exc_info=True)

    # 4. Persist synthesized artifact
    synth_content = f"# Synthesized Response\n\n**Synthesizer Model:** {parsed_synthesizer_config.model_id}\n"
    synth_content += f"## Metrics\nCost: ${output_data.synthesizer_metrics.get('cost',0):.6f}\nLatency: {output_data.synthesizer_metrics.get('api_latency_ms','N/A')}ms\n\n"
    if output_data.synthesizer_thinking_process:
        synth_content += f"## Thinking Process\n```\n{output_data.synthesizer_thinking_process}\n```\n"
    if tournament_type == "code" and output_data.synthesized_extracted_code:
        synth_content += f"## Extracted Code\n```python\n{output_data.synthesized_extracted_code}\n```\n"
    elif output_data.synthesized_response_text: # Fallback to full text if code not extracted or text type
        synth_content += f"## Response Text\n```\n{output_data.synthesized_response_text}\n```\n"
    if output_data.error_message and not synthesizer_success : # If synthesizer specifically failed
         synth_content += f"\n## Synthesizer Error\n```\n{output_data.error_message}\n```"

    (storage_path / "synthesized_response.md").write_text(synth_content, encoding='utf-8')
    if tournament_type == "code" and output_data.synthesized_extracted_code:
        (storage_path / "synthesized_code.py").write_text(output_data.synthesized_extracted_code, encoding='utf-8')


    # 5. Finalize output
    output_data.status = "SUCCESS" if synthesizer_success else ("PARTIAL_SUCCESS" if any(er.response_text for er in output_data.expert_responses) else "FAILED")
    if not synthesizer_success and not output_data.error_message: # General failure if no specific error
        output_data.error_message = output_data.error_message or "Synthesis process encountered an issue."

    # Aggregate total metrics
    total_cost_agg = output_data.synthesizer_metrics.get("cost", 0.0)
    total_input_tokens_agg = output_data.synthesizer_metrics.get("input_tokens", 0) or 0
    total_output_tokens_agg = output_data.synthesizer_metrics.get("output_tokens", 0) or 0
    
    for resp in output_data.expert_responses:
        total_cost_agg += resp.metrics.get("cost", 0.0)
        total_input_tokens_agg += resp.metrics.get("input_tokens", 0) or 0
        total_output_tokens_agg += resp.metrics.get("output_tokens", 0) or 0
        
    output_data.total_metrics = {
        "total_cost": total_cost_agg,
        "total_input_tokens": total_input_tokens_agg,
        "total_output_tokens": total_output_tokens_agg,
        "overall_task_time_ms": int((time.monotonic() - task_start_time) * 1000)
    }
    
    # Save overall metrics
    (storage_path / "overall_metrics.json").write_text(json.dumps(output_data.total_metrics, indent=2), encoding='utf-8')
    
    logger.info(f"[{request_id}] Single-shot synthesis '{name}' completed. Status: {output_data.status}. Cost: ${total_cost_agg:.4f}")
    return output_data.model_dump()
