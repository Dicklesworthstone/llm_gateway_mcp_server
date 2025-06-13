"""API Meta-Tool for dynamically exposing FastAPI endpoints via MCP.

This module provides a tool for automatically discovering and integrating
FastAPI-compatible REST APIs into the MCP server by pointing it at the
FastAPI server's OpenAPI specification (e.g., /openapi.json).

Usage Examples:

1. Register an API:
   ```python
   result = await client.tools.register_api(
       api_name="petstore",
       openapi_url="https://petstore.swagger.io/v2/swagger.json"
   )
   print(f"Registered {result['tools_count']} tools for the Petstore API")
   ```

2. List all registered APIs:
   ```python
   apis = await client.tools.list_registered_apis()
   for api_name, api_info in apis["apis"].items():
       print(f"{api_name}: {api_info['tools_count']} tools")
   ```

3. Call a dynamically registered tool:
   ```python
   # Get a pet by ID
   pet = await client.tools.call_dynamic_tool(
       tool_name="petstore_getPetById",
       inputs={"petId": 123}
   )
   print(f"Pet name: {pet['name']}")

   # Add a new pet
   new_pet = await client.tools.call_dynamic_tool(
       tool_name="petstore_addPet",
       inputs={
           "body": {
               "id": 0,
               "name": "Fluffy",
               "status": "available"
           }
       }
   )
   print(f"Added pet with ID: {new_pet['id']}")
   ```

4. Unregister an API:
   ```python
   result = await client.tools.unregister_api(api_name="petstore")
   print(f"Unregistered {result['tools_count']} tools")
   ```
"""

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from ultimate_mcp_server.exceptions import ToolError, ToolInputError
from ultimate_mcp_server.services.cache import with_cache
from ultimate_mcp_server.tools.base import (
    with_error_handling,
    with_state_management,
    with_tool_metrics,
)
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tools.meta_api")


async def fetch_openapi_spec(
    url: str, timeout: float = 30.0, headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Fetches the OpenAPI spec from the given URL.

    Args:
        url: URL of the OpenAPI spec (typically ending in /openapi.json)
        timeout: Timeout for the HTTP request in seconds
        headers: Optional headers to include in the request (e.g., for authentication)

    Returns:
        Parsed OpenAPI spec as a dictionary

    Raises:
        ToolError: If the fetch or parsing fails
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise ToolError(
            f"Failed to fetch OpenAPI spec: HTTP {e.response.status_code}",
            details={"url": url, "status_code": e.response.status_code},
        ) from e
    except httpx.RequestError as e:
        raise ToolError(
            f"Failed to fetch OpenAPI spec: {str(e)}", details={"url": url, "error": str(e)}
        ) from e
    except json.JSONDecodeError as e:
        raise ToolError(
            f"Failed to parse OpenAPI spec as JSON: {str(e)}", details={"url": url, "error": str(e)}
        ) from e


def extract_endpoint_info(openapi_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extracts endpoint information from an OpenAPI spec.

    Args:
        openapi_spec: Parsed OpenAPI spec as a dictionary

    Returns:
        List of dictionaries containing endpoint information, each with keys:
        - path: The endpoint path
        - method: The HTTP method (GET, POST, etc.)
        - operation_id: The operationId from the spec (used as tool name)
        - parameters: List of parameter objects
        - request_body: Request body schema (if applicable)
        - responses: Response schemas
        - summary: Endpoint summary
        - description: Endpoint description
    """
    endpoints = []

    paths = openapi_spec.get("paths", {})
    for path, path_item in paths.items():
        for method, operation in path_item.items():
            if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                continue  # Skip non-HTTP methods like "parameters"

            # Extract operation ID (fall back to generating one if not provided)
            operation_id = operation.get("operationId")
            if not operation_id:
                # Generate operation ID from path and method
                path_parts = [p for p in path.split("/") if p and not p.startswith("{")]
                if path_parts:
                    operation_id = f"{method.lower()}_{path_parts[-1]}"
                else:
                    operation_id = f"{method.lower()}_root"

                # Ensure operation_id is a valid Python identifier
                operation_id = re.sub(r"[^a-zA-Z0-9_]", "_", operation_id)
                if operation_id[0].isdigit():
                    operation_id = f"op_{operation_id}"

            # Extract parameters
            parameters = []
            # Include parameters from the path item
            if "parameters" in path_item:
                parameters.extend(path_item["parameters"])
            # Include parameters from the operation (overriding path item parameters if same name)
            if "parameters" in operation:
                # Remove any path item parameters with the same name
                path_param_names = {
                    p["name"] for p in path_item.get("parameters", []) if "name" in p
                }
                op_params = []
                for p in operation["parameters"]:
                    if p.get("name") in path_param_names:
                        # This parameter overrides a path item parameter
                        parameters = [
                            param for param in parameters if param.get("name") != p.get("name")
                        ]
                    op_params.append(p)
                parameters.extend(op_params)

            # Extract request body schema
            request_body = None
            if "requestBody" in operation:
                request_body = operation["requestBody"]

            # Extract response schemas
            responses = operation.get("responses", {})

            endpoints.append(
                {
                    "path": path,
                    "method": method.lower(),
                    "operation_id": operation_id,
                    "parameters": parameters,
                    "request_body": request_body,
                    "responses": responses,
                    "summary": operation.get("summary", ""),
                    "description": operation.get("description", ""),
                    "tags": operation.get("tags", []),
                }
            )

    return endpoints


def generate_tool_function_code(
    endpoint_info: Dict[str, Any],
    base_url: str,
    api_name: str,
    cache_ttl: Optional[int] = None,
    auth_header: Optional[str] = None,
) -> str:
    """Generates Python code for a tool function based on endpoint info.

    Args:
        endpoint_info: Dictionary containing endpoint information
        base_url: Base URL of the API
        api_name: Name of the API (used for function documentation)
        cache_ttl: Optional TTL for caching tool results in seconds
        auth_header: Optional authentication header name to include in requests

    Returns:
        String containing Python code for the tool function
    """
    operation_id = endpoint_info["operation_id"]
    path = endpoint_info["path"]
    method = endpoint_info["method"]
    summary = endpoint_info["summary"]
    description = endpoint_info["description"]
    tags = ", ".join(endpoint_info.get("tags", []))

    # Generate a clean function name (no API prefix, will be added during registration)
    function_name = operation_id

    # Generate docstring
    docstring = (
        f'"""{summary}\n\n'
        if summary
        else f'"""Calls the {method.upper()} {path} endpoint of the {api_name} API.\n\n'
    )
    if description:
        docstring += f"{description}\n\n"
    if tags:
        docstring += f"Tags: {tags}\n\n"

    docstring += "Args:\n"

    # Generate function parameters
    params = []
    path_params = []
    query_params = []
    header_params = []
    body_param = None

    for param in endpoint_info.get("parameters", []):
        param_name = param["name"]
        # Clean the parameter name to be a valid Python identifier
        clean_param_name = re.sub(r"[^a-zA-Z0-9_]", "_", param_name)
        if clean_param_name[0].isdigit():
            clean_param_name = f"p_{clean_param_name}"

        param_type = param.get("schema", {}).get("type", "string")
        required = param.get("required", False)
        param_in = param.get("in", "query")
        param_description = param.get("description", "")

        python_type = "str"
        if param_type == "integer":
            python_type = "int"
        elif param_type == "number":
            python_type = "float"
        elif param_type == "boolean":
            python_type = "bool"
        elif param_type == "array":
            python_type = "List[Any]"
        elif param_type == "object":
            python_type = "Dict[str, Any]"

        if required:
            params.append(f"{clean_param_name}: {python_type}")
            docstring += f"    {clean_param_name}: {param_description} (in: {param_in})\n"
        else:
            params.append(f"{clean_param_name}: Optional[{python_type}] = None")
            docstring += (
                f"    {clean_param_name}: (Optional) {param_description} (in: {param_in})\n"
            )

        # Store parameter location for request building
        if param_in == "path":
            path_params.append((param_name, clean_param_name))
        elif param_in == "query":
            query_params.append((param_name, clean_param_name))
        elif (
            param_in == "header" and param_name.lower() != auth_header.lower()
            if auth_header
            else True
        ):
            header_params.append((param_name, clean_param_name))

    # Handle request body
    if endpoint_info.get("request_body"):
        content = endpoint_info["request_body"].get("content", {})
        if "application/json" in content:
            body_param = "body"
            schema_desc = "Request body"
            # Try to get schema description from the content schema
            schema = content.get("application/json", {}).get("schema", {})
            if "description" in schema:
                schema_desc = schema["description"]
            params.append("body: Dict[str, Any]")
            docstring += f"    body: {schema_desc}\n"

    # Add timeout and auth_token params if needed
    params.append("timeout: float = 30.0")
    docstring += "    timeout: Timeout for the HTTP request in seconds\n"

    if auth_header:
        params.append("auth_token: Optional[str] = None")
        docstring += f"    auth_token: Optional authentication token to include in the '{auth_header}' header\n"

    docstring += '\n    Returns:\n        API response data as a dictionary\n    """'

    # Generate function body
    function_body = []
    function_body.append("    async with httpx.AsyncClient() as client:")

    # Format URL with path params
    if path_params:
        # For path params, replace {param} with {clean_param_name}
        url_format = path
        for param_name, clean_name in path_params:
            url_format = url_format.replace(f"{{{param_name}}}", f"{{{clean_name}}}")
        function_body.append(f'        url = f"{base_url}{url_format}"')
    else:
        function_body.append(f'        url = "{base_url}{path}"')

    # Prepare query params
    if query_params:
        function_body.append("        params = {}")
        for param_name, clean_name in query_params:
            function_body.append(f"        if {clean_name} is not None:")
            function_body.append(f'            params["{param_name}"] = {clean_name}')
    else:
        function_body.append("        params = None")

    # Prepare headers
    function_body.append("        headers = {}")
    if auth_header:
        function_body.append("        if auth_token is not None:")
        function_body.append(f'            headers["{auth_header}"] = auth_token')

    if header_params:
        for param_name, clean_name in header_params:
            function_body.append(f"        if {clean_name} is not None:")
            function_body.append(f'            headers["{param_name}"] = {clean_name}')

    # Prepare request
    request_args = ["url"]
    if query_params:
        request_args.append("params=params")
    if header_params or auth_header:
        request_args.append("headers=headers")
    if body_param:
        request_args.append(f"json={body_param}")
    request_args.append("timeout=timeout")

    function_body.append("        try:")
    function_body.append("            response = await client.{method}({', '.join(request_args)})")
    function_body.append("            response.raise_for_status()")
    function_body.append(
        "            if response.headers.get('content-type', '').startswith('application/json'):"
    )
    function_body.append("                return response.json()")
    function_body.append("            else:")
    function_body.append("                return {{'text': response.text}}")
    function_body.append("        except httpx.HTTPStatusError as e:")
    function_body.append("            error_detail = e.response.text")
    function_body.append("            try:")
    function_body.append("                error_json = e.response.json()")
    function_body.append("                if isinstance(error_json, dict):")
    function_body.append("                    error_detail = error_json")
    function_body.append("            except Exception:")
    function_body.append("                pass  # Couldn't parse JSON error")
    function_body.append("            raise ToolError(")
    function_body.append('                f"API request failed: HTTP {{e.response.status_code}}",')
    function_body.append(
        '                details={{"status_code": e.response.status_code, "response": error_detail}}'
    )
    function_body.append("            )")
    function_body.append("        except httpx.RequestError as e:")
    function_body.append("            raise ToolError(")
    function_body.append('                f"API request failed: {{str(e)}}",')
    function_body.append('                details={{"error": str(e)}}')
    function_body.append("            )")

    # Generate the full function
    param_str = ", ".join(params)
    if param_str:
        param_str = f", {param_str}"

    # Add decorators based on configuration
    decorators = ["@with_tool_metrics", "@with_error_handling"]

    if cache_ttl is not None:
        decorators.insert(0, f"@with_cache(ttl={cache_ttl})")

    function_code = [
        *decorators,
        f"async def {function_name}(self{param_str}):",
        f"{docstring}",
        *function_body,
    ]

    return "\n".join(function_code)


# After the generate_tool_function_code function and before register_api_meta_tools
@with_tool_metrics
@with_error_handling
@with_state_management(namespace="meta_api")
async def register_api(
    api_name: str,
    openapi_url: str,
    base_url: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    auth_header: Optional[str] = None,
    auth_token: Optional[str] = None,
    tool_name_prefix: Optional[str] = None,
    timeout: float = 30.0,
    ctx: Optional[Dict[str, Any]] = None,
    get_state=None,
    set_state=None,
    delete_state=None
) -> Dict[str, Any]:
    """Registers an API with the MCP server by fetching its OpenAPI spec.

    Dynamically generates MCP tools for each endpoint in the API and registers
    them with the MCP server. The tools are prefixed with the API name by default,
    resulting in tool names like "api_name_operation_id".

    Args:
        api_name: A unique name for the API (used as a prefix for tool names)
        openapi_url: URL of the OpenAPI spec (typically ending in /openapi.json)
        base_url: Base URL of the API (if different from the OpenAPI URL)
        cache_ttl: Optional TTL for caching tool results in seconds
        auth_header: Optional name of the header to use for authentication
        auth_token: Optional token to use when fetching the OpenAPI spec
        tool_name_prefix: Optional prefix for tool names (default: api_name)
        timeout: Timeout for the HTTP request in seconds
        ctx: MCP context
        get_state: Function to get state from store (injected by decorator)
        set_state: Function to set state in store (injected by decorator)
        delete_state: Function to delete state from store (injected by decorator)

    Returns:
        A dictionary containing the registration results:
        {
            "success": true,
            "api_name": "example_api",
            "base_url": "https://api.example.com",
            "tools_registered": ["example_api_get_users", "example_api_create_user", ...],
            "tools_count": 5,
            "processing_time": 1.23
        }
    """
    # Validate inputs
    if not api_name:
        raise ToolInputError("api_name cannot be empty")

    # Check if API name has invalid characters
    if not re.match(r"^[a-zA-Z0-9_]+$", api_name):
        raise ToolInputError(
            "api_name must contain only alphanumeric characters and underscores"
        )

    if not openapi_url:
        raise ToolInputError("openapi_url cannot be empty")

    # Get registered APIs from state store
    registered_apis = await get_state("registered_apis", {})
    generated_tools = await get_state("generated_tools", {})

    # Check if API is already registered
    if api_name in registered_apis:
        raise ToolInputError(
            f"API {api_name} is already registered. Use a different name or unregister it first."
        )

    # Set tool name prefix
    tool_name_prefix = tool_name_prefix or api_name

    # Determine base URL if not provided
    if not base_url:
        # Extract base URL from OpenAPI URL
        try:
            parsed_url = urlparse(openapi_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            logger.info(f"Using base_url: {base_url} (derived from openapi_url)")
        except Exception as e:
            raise ToolInputError(f"Could not determine base_url from openapi_url: {str(e)}") from e

    # Prepare headers for fetching OpenAPI spec
    headers = None
    if auth_token and auth_header:
        headers = {auth_header: auth_token}

    # Fetch OpenAPI spec
    logger.info(f"Fetching OpenAPI spec from {openapi_url}")
    start_time = time.time()
    openapi_spec = await fetch_openapi_spec(openapi_url, timeout, headers)

    # Extract endpoint information
    endpoints = extract_endpoint_info(openapi_spec)
    logger.info(f"Extracted {len(endpoints)} endpoints from OpenAPI spec")

    # Get MCP server from context
    mcp = ctx.get('mcp')
    if not mcp:
        raise ToolError("MCP server context not available")

    # Generate and register tools for each endpoint
    registered_tools = []
    generated_code = {}

    for endpoint in endpoints:
        operation_id = endpoint["operation_id"]
        tool_name = f"{tool_name_prefix}_{operation_id}"

        # Skip if this tool is already registered
        if tool_name in generated_tools:
            logger.warning(f"Tool {tool_name} already registered, skipping")
            continue

        # Generate tool function code
        tool_code = generate_tool_function_code(
            endpoint, base_url, api_name, cache_ttl, auth_header
        )

        # Store the generated code for debugging
        generated_code[tool_name] = tool_code

        # Create and register the tool function
        try:
            # Create a namespace for the exec
            namespace = {}
            # Add required imports to the namespace
            namespace.update(
                {
                    "httpx": httpx,
                    "ToolError": ToolError,
                    "Dict": Dict,
                    "Any": Any,
                    "Optional": Optional,
                    "List": List,
                    "with_tool_metrics": with_tool_metrics,
                    "with_error_handling": with_error_handling,
                    "with_cache": with_cache,
                }
            )

            # Execute the generated code
            exec(tool_code, namespace)

            # Get the generated function from the namespace
            generated_func = namespace[operation_id]

            # Register with MCP server
            mcp.tool(name=tool_name)(generated_func)

            # Store the generated tool in state
            generated_tools[tool_name] = tool_code
            registered_tools.append(tool_name)

            logger.info(
                f"Registered tool {tool_name} for endpoint {endpoint['method'].upper()} {endpoint['path']}"
            )
        except Exception as e:
            logger.error(f"Failed to register tool {tool_name}: {str(e)}", exc_info=True)
            if "tool_code" in locals():
                logger.error(f"Generated code that failed:\n{tool_code}")

    # Store API information in state store
    registered_apis[api_name] = {
        "base_url": base_url,
        "openapi_url": openapi_url,
        "spec": openapi_spec,
        "tools": registered_tools,
        "tool_name_prefix": tool_name_prefix,
        "generated_code": generated_code,
        "auth_header": auth_header,
    }

    # Update state store
    await set_state("registered_apis", registered_apis)
    await set_state("generated_tools", generated_tools)

    processing_time = time.time() - start_time
    logger.success(
        f"API {api_name} registered with {len(registered_tools)} tools in {processing_time:.2f}s"
    )

    return {
        "success": True,
        "api_name": api_name,
        "base_url": base_url,
        "tools_registered": registered_tools,
        "tools_count": len(registered_tools),
        "processing_time": processing_time,
    }

@with_tool_metrics
@with_error_handling
@with_state_management(namespace="meta_api")
async def list_registered_apis(
    ctx: Optional[Dict[str, Any]] = None,
    get_state=None,
    set_state=None,
    delete_state=None
) -> Dict[str, Any]:
    """Lists all registered APIs and their endpoints.

    Args:
        ctx: MCP context
        get_state: Function to get state from store (injected by decorator)
        set_state: Function to set state in store (injected by decorator)
        delete_state: Function to delete state from store (injected by decorator)

    Returns:
        A dictionary containing the registered APIs:
        {
            "success": true,
            "apis": {
                "example_api": {
                    "base_url": "https://api.example.com",
                    "openapi_url": "https://api.example.com/openapi.json",
                    "tools_count": 5,
                    "tools": ["example_api_get_users", "example_api_create_user", ...]
                },
                ...
            },
            "total_apis": 2,
            "total_tools": 12
        }
    """
    # Get state data
    registered_apis = await get_state("registered_apis", {})
    generated_tools = await get_state("generated_tools", {})

    result = {
        "success": True,
        "apis": {},
        "total_apis": len(registered_apis),
        "total_tools": len(generated_tools),
    }

    for api_name, api_info in registered_apis.items():
        result["apis"][api_name] = {
            "base_url": api_info["base_url"],
            "openapi_url": api_info["openapi_url"],
            "tools_count": len(api_info["tools"]),
            "tools": api_info["tools"],
            "tool_name_prefix": api_info["tool_name_prefix"],
        }

    return result

@with_tool_metrics
@with_error_handling
@with_state_management(namespace="meta_api")
async def get_api_details(
    api_name: str,
    ctx: Optional[Dict[str, Any]] = None,
    get_state=None,
    set_state=None,
    delete_state=None
) -> Dict[str, Any]:
    """Gets detailed information about a registered API.

    Args:
        api_name: The name of the API to get details for
        ctx: MCP context
        get_state: Function to get state from store (injected by decorator)
        set_state: Function to set state in store (injected by decorator)
        delete_state: Function to delete state from store (injected by decorator)

    Returns:
        A dictionary containing the API details:
        {
            "success": true,
            "api_name": "example_api",
            "base_url": "https://api.example.com",
            "openapi_url": "https://api.example.com/openapi.json",
            "tools": [
                {
                    "name": "example_api_get_users",
                    "method": "get",
                    "path": "/users",
                    "summary": "Get all users",
                    "description": "Returns a list of all users in the system",
                    "parameters": [...]
                },
                ...
            ],
            "endpoints_count": 5
        }
    """
    # Get registered APIs from state
    registered_apis = await get_state("registered_apis", {})

    if api_name not in registered_apis:
        raise ToolInputError(f"API {api_name} not found")

    api_info = registered_apis[api_name]

    # Extract endpoint details from the OpenAPI spec
    endpoints = []
    spec = api_info["spec"]

    for endpoint_info in extract_endpoint_info(spec):
        tool_name = f"{api_info['tool_name_prefix']}_{endpoint_info['operation_id']}"
        endpoints.append(
            {
                "name": tool_name,
                "method": endpoint_info["method"],
                "path": endpoint_info["path"],
                "summary": endpoint_info["summary"],
                "description": endpoint_info["description"],
                "parameters": endpoint_info["parameters"],
                "tags": endpoint_info.get("tags", []),
            }
        )

    return {
        "success": True,
        "api_name": api_name,
        "base_url": api_info["base_url"],
        "openapi_url": api_info["openapi_url"],
        "tools": endpoints,
        "endpoints_count": len(endpoints),
    }

@with_tool_metrics
@with_error_handling
@with_state_management(namespace="meta_api")
async def unregister_api(
    api_name: str,
    ctx: Optional[Dict[str, Any]] = None,
    get_state=None,
    set_state=None,
    delete_state=None
) -> Dict[str, Any]:
    """Unregisters an API and all its tools from the MCP server.

    Args:
        api_name: The name of the API to unregister
        ctx: MCP context
        get_state: Function to get state from store (injected by decorator)
        set_state: Function to set state in store (injected by decorator)
        delete_state: Function to delete state from store (injected by decorator)

    Returns:
        A dictionary indicating the result:
        {
            "success": true,
            "api_name": "example_api",
            "tools_unregistered": ["example_api_get_users", "example_api_create_user", ...],
            "tools_count": 5
        }
    """
    # Get state data
    registered_apis = await get_state("registered_apis", {})
    generated_tools = await get_state("generated_tools", {})

    if api_name not in registered_apis:
        raise ToolInputError(f"API {api_name} not found")

    api_info = registered_apis[api_name]
    tools = api_info["tools"]

    # Get MCP server from context
    mcp = ctx.get('mcp')
    if not mcp:
        raise ToolError("MCP server context not available")

    # Unregister tools from MCP server
    for tool_name in tools:
        try:
            # Check if the MCP server has a method for unregistering tools
            if hasattr(mcp, "unregister_tool"):
                mcp.unregister_tool(tool_name)
            # If not, try to remove from the tools dictionary
            elif hasattr(mcp, "tools"):
                if tool_name in mcp.tools:
                    del mcp.tools[tool_name]

            # Remove from our generated tools dictionary
            if tool_name in generated_tools:
                del generated_tools[tool_name]

            logger.info(f"Unregistered tool {tool_name}")
        except Exception as e:
            logger.error(f"Failed to unregister tool {tool_name}: {str(e)}", exc_info=True)

    # Remove API from registered APIs
    del registered_apis[api_name]

    # Update state
    await set_state("registered_apis", registered_apis)
    await set_state("generated_tools", generated_tools)

    logger.success(f"API {api_name} unregistered with {len(tools)} tools")

    return {
        "success": True,
        "api_name": api_name,
        "tools_unregistered": tools,
        "tools_count": len(tools),
    }

@with_tool_metrics
@with_error_handling
@with_state_management(namespace="meta_api")
async def call_dynamic_tool(
    tool_name: str,
    inputs: Optional[Dict[str, Any]] = None,
    ctx: Optional[Dict[str, Any]] = None,
    get_state=None,
    set_state=None,
    delete_state=None
) -> Dict[str, Any]:
    """Calls a dynamically registered tool by name.

    This is a convenience function for calling tools registered via register_api,
    allowing direct invocation of API endpoints.

    Args:
        tool_name: Name of the tool to call
        inputs: Inputs to pass to the tool (parameters for the API endpoint)
        ctx: MCP context
        get_state: Function to get state from store (injected by decorator)
        set_state: Function to set state in store (injected by decorator)
        delete_state: Function to delete state from store (injected by decorator)

    Returns:
        The result of the tool call
    """
    # Get MCP server from context
    mcp = ctx.get('mcp')
    if not mcp:
        raise ToolError("MCP server context not available")

    # Get registered APIs and generated tools from state
    registered_apis = await get_state("registered_apis", {})
    generated_tools = await get_state("generated_tools", {})

    if not tool_name:
        raise ToolInputError("tool_name cannot be empty")

    # Check if tool exists
    if tool_name not in generated_tools:
        valid_tools = list(generated_tools.keys())
        raise ToolInputError(
            f"Tool {tool_name} not found. Valid tools: {', '.join(valid_tools[:10])}..."
            if len(valid_tools) > 10
            else f"Tool {tool_name} not found. Valid tools: {', '.join(valid_tools)}"
        )

    # Initialize inputs
    if inputs is None:
        inputs = {}

    # Find which API this tool belongs to
    api_name = None
    for name, info in registered_apis.items():
        if tool_name in info["tools"]:
            api_name = name
            break

    if not api_name:
        logger.warning(f"Could not determine which API {tool_name} belongs to")

    # Add auth_token to inputs if specified and the API has an auth_header
    api_info = registered_apis.get(api_name, {})
    if api_info.get("auth_header") and "auth_token" in ctx:
        inputs["auth_token"] = ctx["auth_token"]

    # Call the tool directly through MCP
    logger.info(f"Calling dynamic tool {tool_name} with inputs: {inputs}")
    start_time = time.time()

    # MCP execute may be different from mcp.call_tool, handle appropriately
    if hasattr(mcp, "execute"):
        result = await mcp.execute(tool_name, inputs)
    else:
        result = await mcp.call_tool(tool_name, inputs)

    processing_time = time.time() - start_time

    # Add metadata to result
    if isinstance(result, dict):
        result["processing_time"] = processing_time
        result["success"] = True
    else:
        result = {"data": result, "processing_time": processing_time, "success": True}

    logger.info(f"Called dynamic tool {tool_name} in {processing_time:.4f}s")
    return result

@with_tool_metrics
@with_error_handling
@with_state_management(namespace="meta_api")
async def refresh_api(
    api_name: str,
    update_base_url: Optional[str] = None,
    timeout: float = 30.0,
    ctx: Optional[Dict[str, Any]] = None,
    get_state=None,
    set_state=None,
    delete_state=None
) -> Dict[str, Any]:
    """Refreshes an API by re-fetching its OpenAPI spec and updating tools.

    This is useful when the API has been updated with new endpoints or
    modifications to existing endpoints.

    Args:
        api_name: The name of the API to refresh
        update_base_url: Optional new base URL for the API
        timeout: Timeout for the HTTP request in seconds
        ctx: MCP context
        get_state: Function to get state from store (injected by decorator)
        set_state: Function to set state in store (injected by decorator)
        delete_state: Function to delete state from store (injected by decorator)

    Returns:
        A dictionary indicating the result:
        {
            "success": true,
            "api_name": "example_api",
            "tools_added": ["example_api_new_endpoint", ...],
            "tools_updated": ["example_api_modified_endpoint", ...],
            "tools_removed": ["example_api_deleted_endpoint", ...],
            "tools_count": 8
        }
    """
    # Get registered APIs from state
    registered_apis = await get_state("registered_apis", {})

    if api_name not in registered_apis:
        raise ToolInputError(f"API {api_name} not found")

    api_info = registered_apis[api_name]
    old_tools = set(api_info["tools"])

    # Determine if we need to update the base URL
    base_url = update_base_url or api_info["base_url"]

    # First, unregister the API
    await unregister_api(api_name, ctx=ctx, get_state=get_state, set_state=set_state, delete_state=delete_state)

    # Re-register with the same parameters but potentially updated base URL
    result = await register_api(
        api_name=api_name,
        openapi_url=api_info["openapi_url"],
        base_url=base_url,
        auth_header=api_info.get("auth_header"),
        tool_name_prefix=api_info["tool_name_prefix"],
        timeout=timeout,
        ctx=ctx,
        get_state=get_state,
        set_state=set_state,
        delete_state=delete_state
    )

    # Determine which tools were added, updated, or removed
    new_tools = set(result["tools_registered"])
    tools_added = list(new_tools - old_tools)
    tools_removed = list(old_tools - new_tools)
    tools_updated = list(new_tools.intersection(old_tools))

    logger.success(
        f"API {api_name} refreshed: "
        f"{len(tools_added)} added, {len(tools_removed)} removed, {len(tools_updated)} updated"
    )

    return {
        "success": True,
        "api_name": api_name,
        "tools_added": tools_added,
        "tools_updated": tools_updated,
        "tools_removed": tools_removed,
        "tools_count": len(new_tools),
    }

@with_tool_metrics
@with_error_handling
@with_state_management(namespace="meta_api")
async def get_tool_details(
    tool_name: str,
    ctx: Optional[Dict[str, Any]] = None,
    get_state=None,
    set_state=None,
    delete_state=None
) -> Dict[str, Any]:
    """Gets detailed information about a dynamically registered tool.

    Args:
        tool_name: Name of the tool to get details for
        ctx: MCP context
        get_state: Function to get state from store (injected by decorator)
        set_state: Function to set state in store (injected by decorator)
        delete_state: Function to delete state from store (injected by decorator)

    Returns:
        A dictionary containing the tool details:
        {
            "success": true,
            "tool_name": "example_api_get_users",
            "api_name": "example_api",
            "method": "get",
            "path": "/users",
            "summary": "Get all users",
            "description": "Returns a list of all users in the system",
            "parameters": [...],
            "source_code": "..."
        }
    """
    # Get registered APIs and generated tools from state
    registered_apis = await get_state("registered_apis", {})
    generated_tools = await get_state("generated_tools", {})

    if tool_name not in generated_tools:
        raise ToolInputError(f"Tool {tool_name} not found")

    # Find which API this tool belongs to
    api_name = None
    for name, info in registered_apis.items():
        if tool_name in info["tools"]:
            api_name = name
            break

    if not api_name:
        raise ToolError(f"Could not determine which API {tool_name} belongs to")

    api_info = registered_apis[api_name]

    # Find endpoint information in the API's endpoint list
    endpoint_info = None
    for endpoint in extract_endpoint_info(api_info["spec"]):
        if f"{api_info['tool_name_prefix']}_{endpoint['operation_id']}" == tool_name:
            endpoint_info = endpoint
            break

    if not endpoint_info:
        raise ToolError(f"Could not find endpoint information for tool {tool_name}")

    # Get the source code
    source_code = api_info.get("generated_code", {}).get(tool_name, "Source code not available")

    return {
        "success": True,
        "tool_name": tool_name,
        "api_name": api_name,
        "method": endpoint_info["method"],
        "path": endpoint_info["path"],
        "summary": endpoint_info["summary"],
        "description": endpoint_info["description"],
        "parameters": endpoint_info["parameters"],
        "tags": endpoint_info.get("tags", []),
        "source_code": source_code,
    }

@with_tool_metrics
@with_error_handling
@with_state_management(namespace="meta_api")
async def list_available_tools(
    include_source_code: bool = False,
    ctx: Optional[Dict[str, Any]] = None,
    get_state=None,
    set_state=None,
    delete_state=None
) -> Dict[str, Any]:
    """Lists all available tools registered via the API Meta-Tool.

    Args:
        include_source_code: Whether to include source code in the response
        ctx: MCP context
        get_state: Function to get state from store (injected by decorator)
        set_state: Function to set state in store (injected by decorator)
        delete_state: Function to delete state from store (injected by decorator)

    Returns:
        A dictionary containing the available tools:
        {
            "success": true,
            "tools": [
                {
                    "name": "example_api_get_users",
                    "api_name": "example_api",
                    "method": "get",
                    "path": "/users",
                    "summary": "Get all users",
                    "source_code": "..." # Only if include_source_code=True
                },
                ...
            ],
            "tools_count": 12
        }
    """
    # Get registered APIs from state
    registered_apis = await get_state("registered_apis", {})
    generated_tools = await get_state("generated_tools", {})

    tools = []

    for api_name, api_info in registered_apis.items():
        spec = api_info["spec"]
        endpoints = extract_endpoint_info(spec)

        for endpoint in endpoints:
            tool_name = f"{api_info['tool_name_prefix']}_{endpoint['operation_id']}"
            if tool_name in generated_tools:
                tool_info = {
                    "name": tool_name,
                    "api_name": api_name,
                    "method": endpoint["method"],
                    "path": endpoint["path"],
                    "summary": endpoint["summary"],
                }

                if include_source_code:
                    tool_info["source_code"] = api_info.get("generated_code", {}).get(
                        tool_name, "Source code not available"
                    )

                tools.append(tool_info)

    return {"success": True, "tools": tools, "tools_count": len(tools)}

# Now we have all our stateless functions defined:
# register_api, list_registered_apis, get_api_details, unregister_api
# call_dynamic_tool, refresh_api, get_tool_details, list_available_tools

def register_api_meta_tools(mcp_server):
    """Registers API Meta-Tool with the MCP server.

    Args:
        mcp_server: MCP server instance
    """
    # Register tools with MCP server
    mcp_server.tool(name="register_api")(register_api)
    mcp_server.tool(name="list_registered_apis")(list_registered_apis)
    mcp_server.tool(name="get_api_details")(get_api_details)
    mcp_server.tool(name="unregister_api")(unregister_api)
    mcp_server.tool(name="call_dynamic_tool")(call_dynamic_tool)
    mcp_server.tool(name="refresh_api")(refresh_api)
    mcp_server.tool(name="get_tool_details")(get_tool_details)
    mcp_server.tool(name="list_available_tools")(list_available_tools)

    logger.info("Registered API Meta-Tool functions")
    return None  # No need to return an instance anymore


# Example usage if this module is run directly
if __name__ == "__main__":
    import argparse
    import asyncio

    from ultimate_mcp_server import create_app

    async def main():
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="API Meta-Tool for Ultimate MCP Server")
        parser.add_argument("--register", help="Register an API with the given name")
        parser.add_argument("--url", help="OpenAPI spec URL")
        parser.add_argument("--list", action="store_true", help="List registered APIs")
        parser.add_argument("--details", help="Get details for the given API")
        parser.add_argument("--unregister", help="Unregister the given API")
        parser.add_argument("--refresh", help="Refresh the given API")
        parser.add_argument("--base-url", help="Base URL for API requests")
        args = parser.parse_args()

        # Create MCP server
        create_app()
        # In FastMCP 2.0+, access the MCP server directly from the Gateway instance
        # The create_app() should return the gateway instance or we need to get it differently
        from ultimate_mcp_server.core import _gateway_instance
        mcp_server = _gateway_instance.mcp if _gateway_instance else None
        if not mcp_server:
            raise RuntimeError("Gateway instance not initialized or MCP server not available")

        # Register API Meta-Tool
        register_api_meta_tools(mcp_server)

        # Create context for stateless functions
        ctx = {"mcp": mcp_server}

        # Process commands
        if args.register and args.url:
            result = await register_api(
                api_name=args.register, 
                openapi_url=args.url, 
                base_url=args.base_url,
                ctx=ctx
            )
            print(f"Registered API {args.register} with {result['tools_count']} tools")
            print(f"Tools: {', '.join(result['tools_registered'])}")
        elif args.list:
            result = await list_registered_apis(ctx=ctx)
            print(f"Registered APIs ({result['total_apis']}):")
            for api_name, api_info in result["apis"].items():
                print(
                    f"- {api_name}: {api_info['tools_count']} tools, Base URL: {api_info['base_url']}"
                )
        elif args.details:
            result = await get_api_details(args.details, ctx=ctx)
            print(f"API {args.details} ({result['endpoints_count']} endpoints):")
            print(f"Base URL: {result['base_url']}")
            print(f"OpenAPI URL: {result['openapi_url']}")
            print("Endpoints:")
            for endpoint in result["tools"]:
                print(f"- {endpoint['method'].upper()} {endpoint['path']} ({endpoint['name']})")
                if endpoint["summary"]:
                    print(f"  Summary: {endpoint['summary']}")
        elif args.unregister:
            result = await unregister_api(args.unregister, ctx=ctx)
            print(f"Unregistered API {args.unregister} with {result['tools_count']} tools")
        elif args.refresh:
            result = await refresh_api(
                api_name=args.refresh, 
                update_base_url=args.base_url,
                ctx=ctx
            )
            print(
                f"Refreshed API {args.refresh}: {len(result['tools_added'])} added, {len(result['tools_removed'])} removed, {len(result['tools_updated'])} updated"
            )
        else:
            print("No action specified. Use --help for usage information.")

    # Run the main function
    asyncio.run(main())
