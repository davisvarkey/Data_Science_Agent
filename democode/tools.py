"""
Tools for AI agent to interact with CSV data and execute pandas code.
Includes both CPU and GPU-accelerated execution options.
"""

import io
import sys
import contextlib
import time


# =============================================================================
# PERSISTENT EXECUTION ENVIRONMENT
# =============================================================================

# Flag to track if GPU acceleration has been initialized
_GPU_INITIALIZED = False

# Persistent namespace that carries over between executions
PERSISTENT_NAMESPACE = {}


def _initialize_gpu_pandas():
    """Initialize GPU-accelerated pandas once at module level."""
    global _GPU_INITIALIZED
    if not _GPU_INITIALIZED:
        try:
            import cudf.pandas
            cudf.pandas.install()
            _GPU_INITIALIZED = True
            print("[GPU Acceleration] cudf.pandas initialized successfully")
        except Exception as e:
            print(f"[GPU Acceleration] Warning: Failed to initialize cudf.pandas: {e}")
            print("[GPU Acceleration] Falling back to CPU mode")


def _initialize_namespace():
    """Initialize or reset the persistent namespace with pandas."""
    global PERSISTENT_NAMESPACE
    # Import pandas AFTER cudf.pandas.install() has been called
    import pandas as pd
    import matplotlib.pyplot as plt
    PERSISTENT_NAMESPACE = {'pd': pd, 'plt': plt}


def reset_execution_environment():
    """Reset the persistent execution environment, clearing all variables."""
    global _GPU_INITIALIZED
    _initialize_namespace()
    return {
        "success": True,
        "message": "Execution environment reset. All variables cleared.",
        "gpu_mode": _GPU_INITIALIZED
    }


# Initialize GPU acceleration FIRST, then import pandas
_initialize_gpu_pandas()
# Now initialize the namespace with GPU-accelerated pandas
_initialize_namespace()


# =============================================================================
# TOOL FUNCTIONS
# =============================================================================

def get_csv_headers(file_path: str) -> dict:
    """
    Get the column headers from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        Dictionary with columns list and count
    """
    try:
        import pandas as pd
        df_header = pd.read_csv(file_path, nrows=0)
        columns_list = df_header.columns.tolist()
        return {
            "success": True,
            "file_path": file_path,
            "columns": columns_list,
            "column_count": len(columns_list)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def execute_python_code(code: str, use_gpu: bool = True, verbose: bool = True) -> dict:
    """
    Execute Python code using pandas with GPU acceleration.

    Variables persist between executions in the PERSISTENT_NAMESPACE,
    allowing dataframes to be reused across multiple calls.

    Note: GPU acceleration via cudf.pandas is initialized at module load time
    (before pandas import) and applies to all executions if available.

    Args:
        code: Python code string to execute
        use_gpu: Kept for API compatibility. GPU is initialized at module load.
        verbose: If True, print execution details and output. Default True.

    Returns:
        Dictionary with execution results, timing, dataframe tracking, and any errors
    """
    global PERSISTENT_NAMESPACE, _GPU_INITIALIZED

    mode = "gpu_accelerated" if _GPU_INITIALIZED else "cpu"

    if verbose:
        print(f"\n[Executing Python Code - {'GPU Accelerated' if _GPU_INITIALIZED else 'CPU'} Mode]")
        print("-" * 60)
        print(code)
        print("-" * 60)

    try:
        start_time = time.time()

        # Track dataframes before execution
        dataframes_before = _get_dataframe_info(PERSISTENT_NAMESPACE)

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, PERSISTENT_NAMESPACE)

        end_time = time.time()
        execution_time = end_time - start_time

        # Track dataframes after execution
        dataframes_after = _get_dataframe_info(PERSISTENT_NAMESPACE)

        # Determine what changed
        dataframe_changes = _compute_dataframe_changes(dataframes_before, dataframes_after)

        # Get captured output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Print the output so user can see it (only if verbose)
        if verbose:
            if stdout_output:
                print(stdout_output, end='')
            if stderr_output:
                print(stderr_output, end='', file=sys.stderr)

        return {
            "success": True,
            "mode": mode,
            "execution_time_seconds": round(execution_time, 4),
            "stdout": stdout_output,
            "stderr": stderr_output,
            "dataframes": dataframes_after,
            "dataframe_changes": dataframe_changes,
            "message": f"Code executed successfully on {'GPU' if _GPU_INITIALIZED else 'CPU'} in {execution_time:.4f} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "mode": mode,
            "error": str(e),
            "error_type": type(e).__name__
        }


def _get_dataframe_info(namespace: dict) -> dict:
    """
    Extract information about pandas DataFrames in the namespace.

    Args:
        namespace: The execution namespace to inspect

    Returns:
        Dictionary mapping variable names to dataframe metadata
    """
    import pandas as pd
    dataframe_info = {}

    for var_name, var_value in namespace.items():
        # Skip built-in items and modules
        if var_name.startswith('_') or var_name == 'pd':
            continue

        # Check if it's a DataFrame
        if isinstance(var_value, pd.DataFrame):
            try:
                dataframe_info[var_name] = {
                    "shape": var_value.shape,
                    "columns": var_value.columns.tolist()[:10],  # Limit to first 10 columns
                    "dtypes": var_value.dtypes.astype(str).to_dict() if len(var_value.columns) <= 20 else "too_many_columns",
                    "memory_usage_mb": round(var_value.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }
            except Exception:
                # In case of any error getting metadata
                dataframe_info[var_name] = {
                    "shape": var_value.shape,
                    "columns": "error_getting_columns"
                }

    return dataframe_info


def _compute_dataframe_changes(before: dict, after: dict) -> dict:
    """
    Compute what dataframes were added, modified, or removed.

    Args:
        before: Dataframe info before execution
        after: Dataframe info after execution

    Returns:
        Dictionary with added, modified, and removed dataframe names
    """
    before_names = set(before.keys())
    after_names = set(after.keys())

    added = list(after_names - before_names)
    removed = list(before_names - after_names)

    # Check for modifications (shape change)
    modified = []
    for name in before_names & after_names:
        if before[name].get("shape") != after[name].get("shape"):
            modified.append(name)

    return {
        "added": added,
        "modified": modified,
        "removed": removed
    }


# =============================================================================
# TOOL DEFINITIONS FOR OPENAI API
# =============================================================================

get_csv_headers_tool = {
    "type": "function",
    "function": {
        "name": "get_csv_headers",
        "description": "Get the column headers from a CSV file. Use this to understand what columns are available in the CSV before writing pandas code.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the CSV file"
                }
            },
            "required": ["file_path"]
        }
    }
}

execute_python_code_tool = {
    "type": "function",
    "function": {
        "name": "execute_python_code",
        "description": "Execute Python pandas code with optional GPU acceleration. By default, uses GPU-accelerated execution via NVIDIA cudf.pandas for better performance. The pandas library (pd) is already imported. IMPORTANT: Variables persist between executions - if you've previously loaded data into a variable like 'df', it will still be available in subsequent calls. This allows you to build on previous work without re-reading data.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python pandas code to execute. Use print() to output results. Do not include import statements for pandas. You can reference variables created in previous executions."
                },
                "use_gpu": {
                    "type": "boolean",
                    "description": "If true (default), use GPU acceleration.",
                    "default": True
                }
            },
            "required": ["code"]
        }
    }
}


# =============================================================================
# TOOL REGISTRY
# =============================================================================

# Map of tool names to functions
TOOL_FUNCTIONS = {
    "get_csv_headers": get_csv_headers,
    "execute_python_code": execute_python_code,
    "reset_execution_environment": reset_execution_environment,
}

# List of all tool definitions
ALL_TOOLS = [
    #get_csv_headers_tool,
    execute_python_code_tool,
]


def call_tool(tool_name: str, **arguments):
    """
    Execute a tool by name with given arguments.

    Args:
        tool_name: Name of the tool to call
        **arguments: Arguments to pass to the tool

    Returns:
        Tool execution result as dictionary
    """
    if tool_name in TOOL_FUNCTIONS:
        return TOOL_FUNCTIONS[tool_name](**arguments)
    else:
        return {"error": f"Unknown tool: {tool_name}"}
