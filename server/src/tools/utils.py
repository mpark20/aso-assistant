import requests
from typing import TypedDict, Optional
from requests.exceptions import HTTPError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

MAX_RETRIES = 3

class APIResponse(TypedDict):
    success: bool
    error: Optional[str]
    query_info: Optional[dict[str, any]]
    result: Optional[dict[str, any]]

class QueryInfo(TypedDict):
    endpoint: str
    method: str
    description: str



def is_retriable_error(exception):
    """Check if the error should trigger a retry."""
    if isinstance(exception, HTTPError):
        return exception.response.status_code in [409, 429, 500, 502, 503, 504]
    return False

@retry(
    retry=retry_if_exception(is_retriable_error),
    wait=wait_exponential(multiplier=1, min=0.5, max=5),
    stop=stop_after_attempt(MAX_RETRIES),
)
def query_api_with_retry(endpoint, method="GET", params=None, headers=None, json_data=None, description=None):
    """General helper function to query REST APIs with consistent error handling.
    Credit: https://github.com/snap-stanford/Biomni/blob/c36e39f9202863bc7b0665563e74e97723862fa5/biomni/tool/database.py

    Parameters
    ----------
    endpoint (str): Full URL endpoint to query
    method (str): HTTP method ("GET" or "POST")
    params (dict, optional): Query parameters to include in the URL
    headers (dict, optional): HTTP headers for the request
    json_data (dict, optional): JSON data for POST requests
    description (str, optional): Description of this query for error messages

    Returns
    -------
    dict: Dictionary containing the result or error information

    """
    # Set default headers if not provided
    if headers is None:
        headers = {"Accept": "application/json"}

    # Set default description if not provided
    if description is None:
        description = f"{method} request to {endpoint}"

    url_error = None

    try:
        # Make the API request
        if method.upper() == "GET":
            response = requests.get(endpoint, params=params, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(endpoint, params=params, headers=headers, json=json_data)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}

        response.raise_for_status()

        # Try to parse JSON response
        try:
            result = response.json()
        except ValueError:
            # Return raw text if not JSON
            result = {"raw_text": response.text}

        return {
            "success": True,
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description,
            },
            "result": result,
        }

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        response_text = ""

        # Try to get more detailed error info from response
        if hasattr(e, "response") and e.response:
            try:
                error_json = e.response.json()
                if "messages" in error_json:
                    error_msg = "; ".join(error_json["messages"])
                elif "message" in error_json:
                    error_msg = error_json["message"]
                elif "error" in error_json:
                    error_msg = error_json["error"]
                elif "detail" in error_json:
                    error_msg = error_json["detail"]
            except Exception:
                response_text = e.response.text

        return {
            "success": False,
            "error": f"API error: {error_msg}. {response_text}",
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description,
            },
        }