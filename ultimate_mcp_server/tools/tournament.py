"""Tournament tools for Ultimate MCP Server."""
from typing import Any, Dict, List, Optional

from ultimate_mcp_server.exceptions import ToolError

from ultimate_mcp_server.core.models.tournament import (
    CancelTournamentInput,
    CancelTournamentOutput,
    CreateTournamentInput,
    CreateTournamentOutput,
    GetTournamentResultsInput,
    GetTournamentStatusInput,
    GetTournamentStatusOutput,
    TournamentBasicInfo,
    TournamentData,
    TournamentStatus,
)
from ultimate_mcp_server.core.models.tournament import (
    EvaluatorConfig as InputEvaluatorConfig,
)
from ultimate_mcp_server.core.models.tournament import (
    ModelConfig as InputModelConfig,
)
from ultimate_mcp_server.core.tournaments.manager import tournament_manager
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.tournament")

# --- Standalone Tool Functions ---

@with_tool_metrics
@with_error_handling
async def create_tournament(
    name: str,
    prompt: str,
    models: List[Dict[str, Any]],
    rounds: int = 3,
    tournament_type: str = "code",
    extraction_model_id: Optional[str] = "anthropic/claude-3-5-haiku-20241022",
    evaluators: Optional[List[Dict[str, Any]]] = None,
    max_retries_per_model_call: int = 3,
    retry_backoff_base_seconds: float = 1.0,
    max_concurrent_model_calls: int = 5
) -> Dict[str, Any]:
    """
    Creates and starts a new LLM competition (tournament) based on a prompt and model configurations.

    Args:
        name: Human-readable name for the tournament (e.g., "Essay Refinement Contest", "Python Sorting Challenge").
        prompt: The task prompt provided to all participating LLM models.
        models: List of model configurations (external key is "models"). Each config is a dictionary specifying:
            - model_id (str, required): e.g., 'openai/gpt-4o'.
            - diversity_count (int, optional, default 1): Number of variants per model.
            # ... (rest of ModelConfig fields) ...
        rounds: Number of tournament rounds. Each round allows models to refine their previous output (if applicable to the tournament type). Default is 3.
        tournament_type: The type of tournament defining the task and evaluation method. Supported types include:
                         - "code": For evaluating code generation based on correctness and potentially style/efficiency.
                         - "text": For general text generation, improvement, or refinement tasks.
                         Default is "code".
        extraction_model_id: (Optional, primarily for 'code' type) Specific LLM model to use for extracting and evaluating results like code blocks. If None, a default is used.
        evaluators: (Optional) List of evaluator configurations as dicts.
        max_retries_per_model_call: Maximum retries per model call.
        retry_backoff_base_seconds: Base seconds for retry backoff.
        max_concurrent_model_calls: Maximum concurrent model calls.

    Returns:
        Dictionary with tournament creation status containing:
        - tournament_id: Unique identifier for the created tournament.
        - status: Initial tournament status (usually 'PENDING' or 'RUNNING').
        - storage_path: Filesystem path where tournament data will be stored.

    Example:
        {
            "tournament_id": "tour_abc123xyz789",
            "status": "PENDING",
            "storage_path": "/path/to/storage/tour_abc123xyz789"
        }

    Raises:
        ToolError: If input is invalid, tournament creation fails, or scheduling fails.
    """
    logger.info(f"Tool 'create_tournament' invoked for: {name}")
    try:
        parsed_model_configs = [InputModelConfig(**mc) for mc in models]
        parsed_evaluators = [InputEvaluatorConfig(**ev) for ev in (evaluators or [])]
        input_data = CreateTournamentInput(
            name=name,
            prompt=prompt,
            models=parsed_model_configs,
            rounds=rounds,
            tournament_type=tournament_type,
            extraction_model_id=extraction_model_id,
            evaluators=parsed_evaluators,
            max_retries_per_model_call=max_retries_per_model_call,
            retry_backoff_base_seconds=retry_backoff_base_seconds,
            max_concurrent_model_calls=max_concurrent_model_calls
        )

        tournament = tournament_manager.create_tournament(input_data)
        if not tournament:
            raise ToolError("Failed to create tournament entry.")

        logger.info("Calling start_tournament_execution (using asyncio)")
        success = tournament_manager.start_tournament_execution(
            tournament_id=tournament.tournament_id
        )

        if not success:
            logger.error(f"Failed to schedule background execution for tournament {tournament.tournament_id}")
            updated_tournament = tournament_manager.get_tournament(tournament.tournament_id)
            error_msg = updated_tournament.error_message if updated_tournament else "Failed to schedule execution."
            raise ToolError(f"Failed to start tournament execution: {error_msg}")

        logger.info(f"Tournament {tournament.tournament_id} ({tournament.name}) created and background execution started.")
        # Include storage_path in the return value
        output = CreateTournamentOutput(
            tournament_id=tournament.tournament_id,
            status=tournament.status,
            storage_path=tournament.storage_path,
            message=f"Tournament '{tournament.name}' created successfully and execution started."
        )
        return output.dict()

    except ValueError as ve:
        logger.warning(f"Validation error creating tournament: {ve}")
        raise ToolError(f"Invalid input: {ve}") from ve
    except Exception as e:
        logger.error(f"Error creating tournament: {e}", exc_info=True)
        raise ToolError(f"An unexpected error occurred: {e}") from e

@with_tool_metrics
@with_error_handling
async def get_tournament_status(
    tournament_id: str
) -> Dict[str, Any]:
    """Retrieves the current status and progress of a specific tournament.

    Use this tool to monitor an ongoing tournament (PENDING, RUNNING) or check the final
    state (COMPLETED, FAILED, CANCELLED) of a past tournament.

    Args:
        tournament_id: Unique identifier of the tournament to check.

    Returns:
        Dictionary containing tournament status information:
        - tournament_id: Unique identifier for the tournament.
        - name: Human-readable name of the tournament.
        - tournament_type: Type of tournament (e.g., "code", "text").
        - status: Current status (e.g., "PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED").
        - current_round: Current round number (1-based) if RUNNING, else the last active round.
        - total_rounds: Total number of rounds configured for this tournament.
        - created_at: ISO timestamp when the tournament was created.
        - updated_at: ISO timestamp when the tournament status was last updated.
        - error_message: Error message if the tournament FAILED (null otherwise).

    Error Handling:
        - Raises ToolError (400) if tournament_id format is invalid.
        - Raises ToolError (404) if the tournament ID is not found.
        - Raises ToolError (500) for internal server errors.

    Example:
        {
            "tournament_id": "tour_abc123xyz789",
            "name": "Essay Refinement Contest",
            "tournament_type": "text",
            "status": "RUNNING",
            "current_round": 2,
            "total_rounds": 3,
            "created_at": "2023-04-15T14:32:17.123456",
            "updated_at": "2023-04-15T14:45:22.123456",
            "error_message": null
        }
    """
    logger.debug(f"Getting status for tournament: {tournament_id}")
    try:
        if not tournament_id or not isinstance(tournament_id, str):
            raise ToolError(
                status_code=400,
                detail="Invalid tournament ID format. Tournament ID must be a non-empty string."
            )

        try:
            input_data = GetTournamentStatusInput(tournament_id=tournament_id)
        except ValueError as ve:
            raise ToolError(
                status_code=400,
                detail=f"Invalid tournament ID: {str(ve)}"
            ) from ve

        tournament = tournament_manager.get_tournament(input_data.tournament_id, force_reload=True)
        if not tournament:
            raise ToolError(
                status_code=404,
                detail=f"Tournament not found: {tournament_id}. Check if the tournament ID is correct or use list_tournaments to see all available tournaments."
            )

        try:
            output = GetTournamentStatusOutput(
                tournament_id=tournament.tournament_id,
                name=tournament.name,
                tournament_type=tournament.config.tournament_type,
                status=tournament.status,
                current_round=tournament.current_round,
                total_rounds=tournament.config.rounds,
                created_at=tournament.created_at,
                updated_at=tournament.updated_at,
                error_message=tournament.error_message
            )
            return output.dict()
        except Exception as e:
            logger.error(f"Error converting tournament data to output format: {e}", exc_info=True)
            raise ToolError(
                status_code=500,
                detail=f"Error processing tournament data: {str(e)}. The tournament data may be corrupted."
            ) from e
    except ToolError:
        raise
    except Exception as e:
        logger.error(f"Error getting tournament status for {tournament_id}: {e}", exc_info=True)
        raise ToolError(
            status_code=500,
            detail=f"Internal server error retrieving tournament status: {str(e)}. Please try again or check the server logs."
        ) from e

@with_tool_metrics
@with_error_handling
async def list_tournaments(
) -> List[Dict[str, Any]]:
    """Lists all created tournaments with basic identifying information and status.

    Useful for discovering existing tournaments and their current states without fetching full results.

    Returns:
        List of dictionaries, each containing basic tournament info:
        - tournament_id: Unique identifier for the tournament.
        - name: Human-readable name of the tournament.
        - tournament_type: Type of tournament (e.g., "code", "text").
        - status: Current status (e.g., "PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED").
        - created_at: ISO timestamp when the tournament was created.
        - updated_at: ISO timestamp when the tournament was last updated.

    Example:
        [
            {
                "tournament_id": "tour_abc123",
                "name": "Tournament A",
                "tournament_type": "code",
                "status": "COMPLETED",
                "created_at": "2023-04-10T10:00:00",
                "updated_at": "2023-04-10T12:30:00"
            },
            ...
        ]
    """
    logger.debug("Listing all tournaments")
    try:
        tournaments = tournament_manager.list_tournaments()
        output_list = []
        for tournament in tournaments:
            try:
                # Ensure tournament object has necessary attributes before accessing
                if not hasattr(tournament, 'tournament_id') or \
                   not hasattr(tournament, 'name') or \
                   not hasattr(tournament, 'config') or \
                   not hasattr(tournament.config, 'tournament_type') or \
                   not hasattr(tournament, 'status') or \
                   not hasattr(tournament, 'created_at') or \
                   not hasattr(tournament, 'updated_at'):
                    logger.warning(f"Skipping tournament due to missing attributes: {getattr(tournament, 'tournament_id', 'UNKNOWN ID')}")
                    continue

                basic_info = TournamentBasicInfo(
                    tournament_id=tournament.tournament_id,
                    name=tournament.name,
                    tournament_type=tournament.config.tournament_type,
                    status=tournament.status,
                    created_at=tournament.created_at,
                    updated_at=tournament.updated_at,
                )
                output_list.append(basic_info.dict())
            except Exception as e:
                logger.warning(f"Skipping tournament {getattr(tournament, 'tournament_id', 'UNKNOWN')} due to data error during processing: {e}")
        return output_list
    except Exception as e:
        logger.error(f"Error listing tournaments: {e}", exc_info=True)
        raise ToolError(
            status_code=500,
            detail=f"Internal server error listing tournaments: {str(e)}"
        ) from e

@with_tool_metrics
@with_error_handling
async def get_tournament_results(
    tournament_id: str
) -> List[Dict[str, str]]:
    """Retrieves the complete results and configuration for a specific tournament.

    Provides comprehensive details including configuration, final scores (if applicable),
    detailed round-by-round results, model outputs, and any errors encountered.
    Use this *after* a tournament has finished (COMPLETED or FAILED) for full analysis.

    Args:
        tournament_id: Unique identifier for the tournament.

    Returns:
        Dictionary containing the full tournament data (structure depends on the tournament manager's implementation, but generally includes config, status, results, timestamps, etc.).

    Example (Conceptual - actual structure may vary):
        {
            "tournament_id": "tour_abc123",
            "name": "Sorting Algo Test",
            "status": "COMPLETED",
            "config": { ... },
            "results": { "scores": { ... }, "round_results": [ { ... }, ... ] },
            "created_at": "...",
            "updated_at": "...",
            "error_message": null
        }

    Raises:
        ToolError: If the tournament ID is invalid, not found, results are not ready (still PENDING/RUNNING), or an internal error occurs.
    """
    logger.debug(f"Getting results for tournament: {tournament_id}")
    try:
        if not tournament_id or not isinstance(tournament_id, str):
            raise ToolError(
                status_code=400,
                detail="Invalid tournament ID format. Tournament ID must be a non-empty string."
            )

        try:
            input_data = GetTournamentResultsInput(tournament_id=tournament_id)
        except ValueError as ve:
             raise ToolError(
                status_code=400,
                detail=f"Invalid tournament ID: {str(ve)}"
            ) from ve

        # Make sure to request TournamentData which should contain results
        tournament_data: Optional[TournamentData] = tournament_manager.get_tournament(input_data.tournament_id, force_reload=True)

        if not tournament_data:
            # Check if the tournament exists but just has no results yet (e.g., PENDING)
            tournament_status_info = tournament_manager.get_tournament(tournament_id) # Gets basic info
            if tournament_status_info:
                current_status = tournament_status_info.status
                if current_status in [TournamentStatus.PENDING, TournamentStatus.RUNNING]:
                     raise ToolError(
                         status_code=404, # Use 404 to indicate results not ready
                         detail=f"Tournament '{tournament_id}' is currently {current_status}. Results are not yet available."
                     )
                else: # Should have results if COMPLETED or ERROR, maybe data issue?
                     logger.error(f"Tournament {tournament_id} status is {current_status} but get_tournament_results returned None.")
                     raise ToolError(
                         status_code=500,
                         detail=f"Could not retrieve results for tournament '{tournament_id}' despite status being {current_status}. There might be an internal data issue."
                     )
            else:
                raise ToolError(
                    status_code=404,
                    detail=f"Tournament not found: {tournament_id}. Cannot retrieve results."
                )

        # NEW: Return a structure that FastMCP might recognize as a pre-formatted content list
        json_string = tournament_data.json()
        logger.info(f"[DEBUG_GET_RESULTS] Returning pre-formatted TextContent list. JSON Snippet: {json_string[:150]}")
        return [{ "type": "text", "text": json_string }]

    except ToolError:
        raise
    except Exception as e:
        logger.error(f"Error getting tournament results for {tournament_id}: {e}", exc_info=True)
        raise ToolError(
            f"Internal server error retrieving tournament results: {str(e)}",
            500 # status_code
        ) from e

@with_tool_metrics
@with_error_handling
async def cancel_tournament(
    tournament_id: str
) -> Dict[str, Any]:
    """Attempts to cancel a running (RUNNING) or pending (PENDING) tournament.

    Signals the tournament manager to stop processing. Cancellation is not guaranteed
    to be immediate. Check status afterwards using `get_tournament_status`.
    Cannot cancel tournaments that are already COMPLETED, FAILED, or CANCELLED.

    Args:
        tournament_id: Unique identifier for the tournament to cancel.

    Returns:
        Dictionary confirming the cancellation attempt:
        - tournament_id: The ID of the tournament targeted for cancellation.
        - status: The status *after* the cancellation attempt (e.g., "CANCELLED", or the previous state like "COMPLETED" if cancellation was not possible).
        - message: A message indicating the outcome (e.g., "Tournament cancellation requested successfully.", "Cancellation failed: Tournament is already COMPLETED.").

    Raises:
        ToolError: If the tournament ID is invalid, not found, or an internal error occurs.
    """
    logger.info(f"Received request to cancel tournament: {tournament_id}")
    try:
        if not tournament_id or not isinstance(tournament_id, str):
            raise ToolError(status_code=400, detail="Invalid tournament ID format.")

        try:
            input_data = CancelTournamentInput(tournament_id=tournament_id)
        except ValueError as ve:
            raise ToolError(status_code=400, detail=f"Invalid tournament ID: {str(ve)}") from ve

        # Call the manager's cancel function
        success, message, final_status = await tournament_manager.cancel_tournament(input_data.tournament_id)

        # Prepare output using the Pydantic model
        output = CancelTournamentOutput(
            tournament_id=tournament_id,
            status=final_status, # Return the actual status after attempt
            message=message
        )

        if not success:
            # Log the failure but return the status/message from the manager
            logger.warning(f"Cancellation attempt for tournament {tournament_id} reported failure: {message}")
            # Raise ToolError if the status implies a client error (e.g., not found)
            if "not found" in message.lower():
                raise ToolError(status_code=404, detail=message)
            elif final_status in [TournamentStatus.COMPLETED, TournamentStatus.FAILED, TournamentStatus.CANCELLED] and "already" in message.lower():
                raise ToolError(status_code=409, detail=message)
            # Optionally handle other errors as 500
            # else:
            #     raise ToolError(status_code=500, detail=f"Cancellation failed: {message}")
        else:
            logger.info(f"Cancellation attempt for tournament {tournament_id} successful. Final status: {final_status}")

        return output.dict()

    except ToolError:
        raise
    except Exception as e:
        logger.error(f"Error cancelling tournament {tournament_id}: {e}", exc_info=True)
        raise ToolError(status_code=500, detail=f"Internal server error during cancellation: {str(e)}") from e