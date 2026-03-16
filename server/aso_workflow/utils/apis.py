"""
Implements common search APIs needed for the ASO workflow.

NOTE: search_serper, search_serper_scholar, and browse_webpage
implementations are from [rlresearch/dr-tulu](https://github.com/rlresearch/dr-tulu/tree/c786700ca63d323ed32f8749eec7ef1562c8ee8c/agent/dr_agent/mcp_backend/apis).
"""
import json
import pdb
import os
import requests
import time
import pandas as pd
from io import StringIO
from functools import cache
from typing import TypedDict, Optional
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from requests.exceptions import HTTPError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from aso_workflow.utils.clinvar import clinvar_formatter
from aso_workflow.utils.pubmed import pubmed_formatter

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

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

class SearchResponse(TypedDict, total=False):
    searchParameters: dict[str, any]
    knowledgeGraph: Optional[dict[str, any]]
    organic: list[dict[str, any]]
    peopleAlsoAsk: Optional[list[dict[str, any]]]
    relatedSearches: Optional[list[dict[str, any]]]

class SearchResult(TypedDict):
    title: str
    link: str
    snippet: str
    position: int
    sitelinks: Optional[list[dict[str, any]]]
    attributes: Optional[dict[str, str]]
    date: Optional[str]

class SearchResponse(TypedDict, total=False):
    searchParameters: dict[str, any]
    knowledgeGraph: Optional[dict[str, any]]
    organic: list[SearchResult]
    peopleAlsoAsk: Optional[list[dict[str, any]]]
    relatedSearches: Optional[list[dict[str, any]]]


@cache
def search_serper(
    query: str,
    num_results: int = 10,
    gl: str = "us",
    hl: str = "en",
    search_type: str = "search",  # Can be "search", "places", "news", "images"
    api_key: str = None,
) -> SearchResponse:
    """
    Search using Serper.dev API for general web search.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)
        gl: Country code to boosts search results whose country of origin matches the parameter value (default: us)
        hl: Host language of user interface (default: en)
        search_type: Type of search to perform (default: "search")
                    Options: "search", "places", "news", "images"
        api_key: Serper API key (if not provided, will use SERPER_API_KEY env var)

    Returns:
        SearchResponse containing:
        - searchParameters: Dict with search metadata
        - knowledgeGraph: Optional knowledge graph information
        - organic: List of organic search results
        - peopleAlsoAsk: Optional list of related questions
        - relatedSearches: Optional list of related search queries
    """
    if not api_key:
        import os

        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError(
                "SERPER_API_KEY environment variable is not set or api_key parameter not provided"
            )

    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": query, "num": num_results, "gl": gl, "hl": hl, "type": search_type})

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=payload)

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        return response.json()

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error performing Serper search: {str(e)}")


class ScholarResult(TypedDict):
    title: str
    link: str
    publicationInfo: str
    snippet: str
    year: int | str
    citedBy: int

class ScholarResponse(TypedDict):
    searchParameters: dict[str, any]
    organic: list[ScholarResult]

@cache
def search_serper_scholar(
    query: str,
    num_results: int = 10,
    api_key: str = None,
) -> ScholarResponse:
    """
    Search academic papers using Serper.dev Scholar API.

    Args:
        query: Academic search query string
        num_results: Number of results to return (default: 10)
        api_key: Serper API key (if not provided, will use SERPER_API_KEY env var)

    Returns:
        ScholarResponse containing:
        - organic: List of academic paper results with:
            - title: Paper title
            - link: URL to the paper
            - publicationInfo: Author and publication details
            - snippet: Brief excerpt from the paper
            - year: Publication year
            - citedBy: Number of citations
    """
    if not api_key:
        import os

        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError(
                "SERPER_API_KEY environment variable is not set or api_key parameter not provided"
            )

    url = "https://google.serper.dev/scholar"

    payload = json.dumps({"q": query, "num": num_results})

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=payload)

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        return response.json()

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error performing Serper scholar search: {str(e)}")


class NCBIResult(TypedDict):
    database: str
    query: str
    results: list[any]
    error: Optional[str]

def search_ncbi(
    database: str,
    search_term: str,
    max_results: int = 3,
) -> NCBIResult:
    """Core function to query NCBI databases using LLM for query interpretation and NCBI eutils.

    Parameters
    ----------
    database (str): NCBI database to query (e.g., "clinvar", "gds", "geoprofiles")
    search_term (str): Search term to query the database with
    max_results (int): Maximum number of results to return

    Returns
    -------
    Dict: Dictionary containing both the structured query and the results

    """
    # Query NCBI API using the structured search term
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    esearch_url = f"{base_url}/esearch.fcgi"
    esearch_params = {
        "db": database,
        "term": search_term,
        "retmode": "json",
        "retmax": 100,
        "usehistory": "y",  # Use history server to store results
    }
    
    if os.getenv("NCBI_API_KEY"):
        esearch_params["api_key"] = os.getenv("NCBI_API_KEY")

    # Get IDs of matching entries
    query_kwargs = {
        "endpoint": esearch_url,
        "method": "GET",
        "params": esearch_params,
        "description": "NCBI ESearch API query"
    }
    search_response = query_api_with_retry(**query_kwargs)

    if not search_response["success"]:
        return NCBIResult(
            database=database,
            query=search_term,
            num_results=0,
            results=[],
            error=search_response["error"],
        )
    time.sleep(0.34)

    search_data = search_response["result"]

    # If we have results, fetch the details
    if "esearchresult" in search_data and int(search_data["esearchresult"].get("count", 0)) > 0:
        # Extract WebEnv and query_key from the search results
        webenv = search_data["esearchresult"].get("webenv", "")
        query_key = search_data["esearchresult"].get("querykey", "")

        # Use WebEnv and query_key if available
        if webenv and query_key:
            # Get details using eSummary
            esummary_url = f"{base_url}/esummary.fcgi"
            esummary_params = {
                "db": database,
                "query_key": query_key,
                "WebEnv": webenv,
                "retmode": "json",
                "retmax": max_results,
            }
            query_kwargs = {
                "endpoint": esummary_url,
                "method": "GET",
                "params": esummary_params,
                "description": "NCBI ESummary API query",
            }
            details_response = query_api_with_retry(**query_kwargs)
            time.sleep(0.34)

            if not details_response["success"]:
                return details_response

            results = details_response["result"]

        else:
            # Fall back to direct ID fetch
            id_list = search_data["esearchresult"]["idlist"][:max_results]

            # Get details for each ID
            esummary_url = f"{base_url}/esummary.fcgi"
            esummary_params = {
                "db": database,
                "id": ",".join(id_list),
                "retmode": "json",
            }
            query_kwargs = {
                "endpoint": esummary_url,
                "method": "GET",
                "params": esummary_params,
                "description": "NCBI ESummary API query",
            }
            details_response = query_api_with_retry(**query_kwargs)

            if not details_response["success"]:
                return details_response

            results = details_response["result"]

        # Format results
        formatted_results = []
        raw_results = results.get("result", {})
        for uid in raw_results.get("uids", []):
            res = raw_results.get(uid, {})
            if database == "clinvar":
                res = clinvar_formatter(res)
            elif database in ["pubmed", "pmc"]:
                res = pubmed_formatter(res, database)
            formatted_results.append(res)

        # Return the combined information
        return NCBIResult(
            database=database,
            query=search_term,
            results=formatted_results,
        )
    else:
        return NCBIResult(
            database=database,
            query=search_term,
            results=[],
            error="No results found for the search query.",
        )


class BrowseWebpageResult(TypedDict):
    url: str
    text: str
    success: bool

@cache
async def browse_webpage(url: str) -> BrowseWebpageResult:
    """
    Crawl a given webpage URL and return its contents as markdown.

    Uses crawl4ai to fetch and render the page, excluding social media links,
    forms, headers, footers, and navigation elements. Prefers fit_markdown
    over raw_markdown when available.
    Credit: https://github.com/rlresearch/dr-tulu/blob/main/agent/dr_agent/mcp_backend/local/crawl4ai_fetcher.py#L25

    Args:
        url: The URL of the webpage to crawl.

    Returns:
        A BrowseWebpageResult containing:
        - url: The original URL that was crawled.
        - text: The page content as markdown, or an error message on failure.
        - success: Whether the crawl succeeded.
    """
    #content_filter = None
    # if use_pruning:
    #     try:
    #         content_filter = PruningContentFilter(
    #             threshold=0.5, threshold_type="fixed", min_word_threshold=50
    #         )
    #     except Exception:
    #         content_filter = None

    run_config = CrawlerRunConfig(
        exclude_social_media_links=True,
        excluded_tags=["form", "header", "footer", "nav"],
        exclude_domains=["ads.com", "spammytrackers.net"],
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=run_config)
    
    if not getattr(result, "success", True):
        error = getattr(result, "error_message", "Unknown error")
        return BrowseWebpageResult(
            url=url,
            text=error,
            success=False,
        )
    
    md_value = ""
    fit_markdown_value = None
    md_obj = getattr(result, "markdown", None)
    if isinstance(md_obj, str):
        md_value = md_obj
    else:
        fit_markdown_value = getattr(md_obj, "fit_markdown", None)
        raw_markdown = getattr(md_obj, "raw_markdown", None)
        if fit_markdown_value:
            md_value = fit_markdown_value
        elif raw_markdown:
            md_value = raw_markdown
        else:
            md_value = str(md_obj) if md_obj is not None else ""
    
    return BrowseWebpageResult(
        url=url,
        text=md_value,
        success=True,
    )


class MutalyzerResult(TypedDict):
    normalized: str
    gene_id: str
    type: str
    deleted: Optional[str]
    inserted: Optional[str]
    offset: Optional[int]
    intronic: Optional[bool]
    equivalent_descriptions: Optional[list[str]]
    coding_position: Optional[int]
    nearest_exon: Optional[int]
    transcript_exons: Optional[int]
    exon_positions: Optional[list[tuple[int, int]]]
    cds: Optional[tuple[int, int]]
    error: Optional[str]

def search_mutalyzer(variant: str, return_exons: bool = False) -> MutalyzerResult:
    """
    Normalize a genetic variant using Mutalyzer and return structured variant metadata.

    Raises ValueError if Mutalyzer returns an error or an unexpected response.

    Args:
        variant: A genetic variant description in HGVS format
            (e.g. "NM_000059.3:c.7790G>A").
        return_exons: If True, includes per-exon boundary positions and CDS coordinates
            in the result when the variant falls within a coding exon.

    Returns:
        A MutalyzerResult containing:
        - normalized: Normalized HGVS description.
        - gene_id: Gene identifier.
        - type: Variant type (e.g. substitution, deletion).
        - deleted: Deleted sequence, if applicable.
        - inserted: Inserted sequence, if applicable.
        - equivalent_descriptions: List of equivalent HGVS representations,
            including the protein-level description if available.
        - coding_position: Position within the CDS, if applicable.
        - nearest_exon: Exon number containing the variant, if applicable.
        - transcript_exons: Total number of exons in the transcript, if applicable.
        - exon_positions: List of [start, end] exon boundaries (only if return_exons=True).
        - cds: Tuple of (cds_start, cds_end) coordinates (only if return_exons=True).
    """
    from urllib.parse import quote

    if not variant or not isinstance(variant, str):
        raise ValueError("Variant must be a non-empty string")

    # Mutalyzer normalize endpoint
    base_url = "https://mutalyzer.nl/api/normalize"
    url = f"{base_url}/{quote(variant, safe='')}"

    try:
        response = requests.get(url, timeout=15)
        data = response.json()
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Mutalyzer returned HTTP {response.status_code}: {response.text}"
        )
    except requests.RequestException as exc:
        raise ValueError(f"Failed to contact Mutalyzer: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Unexpected Mutalyzer error: {exc}") from exc

    location = "exonic"
    if data.get("errors") or data.get("custom", {}).get("errors"):
        errors = data.get("errors") if "errors" in data else data.get("custom", {}).get("errors")
        if len(errors) > 0 and any([e.get("code") == "EINTRONIC" for e in errors]):
            location = "intronic"
            data = data["custom"]
        elif len(errors) > 0:
            err_str = "\n".join([json.dumps(e) for e in errors])
            return MutalyzerResult(
                normalized=None,
                gene_id=None,
                type=None,
                error=err_str
            )

    normalized = (
        data.get("normalized_description")
        or data.get("corrected_description")
    )
    if not normalized:
        errors = data.get("errors") if "errors" in data else data.get("custom", {}).get("errors")
        err_str = "\n".join([json.dumps(e) for e in errors])
        return MutalyzerResult(
            normalized=None,
            gene_id=None,
            type=None,
            deleted=None,
            inserted=None,
            equivalent_descriptions=None,
            coding_position=None,
            error=err_str
        )
    
    # get extra metadata
    variants = []
    if "normalized_model" in data:
        variants = data.get("normalized_model", {}).get("variants", [])
    else:
        variants = data.get("input_model", {}).get("variants", [])
    
    loc_info = variants[0].get("location", {}) if len(variants) else None
    exon_num = None
    if not loc_info or not "type" in loc_info or loc_info.get("type") not in ["point", "range"]:
        exon_info, coding_pos, deleted, inserted, offset = None, None, None, None, None
    else:
        # identify changed bases
        deleted = variants[0].get("deleted", [{}])[0].get("sequence")
        inserted = variants[0].get("inserted", [{}])[0].get("sequence")

        # distance upstream/downstream of nearest coding base (None if exonic)
        offset = loc_info.get("offset", {}).get("value")

        # special case for coding exons: exon positions may be provided
        coding_pos = loc_info.get("position") if loc_info["type"] == "point" else loc_info.get("start", {}).get("position")
        exon_info = data.get("selector_short", {}).get("exon")
        exon_list = exon_info.get("c") if exon_info else None
        parsed_exon_list = []
        if exon_list and coding_pos:
            cds = data["selector_short"]["cds"]
            cds_start, cds_end = int(cds["c"][0][0]), int(cds["c"][0][1])
            for i, (start, end) in enumerate(exon_list):
                try:
                    start, end = start.strip(), end.strip()
                    s = int(start)
                    e = int(end.replace('*', ''))
                    if '*' in end and cds_end:
                        e += cds_end
                    parsed_exon_list.append([s, e])
                    if s <= coding_pos <= e:
                        exon_num = i + 1
                except ValueError:
                    break
    
    equivalent_descriptions = data.get("equivalent_descriptions", [])
    protein_desc = data.get("protein", {}).get("description")
    if protein_desc:
        equivalent_descriptions.append(protein_desc)

    result = MutalyzerResult(
        normalized=normalized,
        gene_id=data.get("gene_id"),
        type=data.get("corrected_model", {}).get("type"),
        deleted=deleted,
        inserted=inserted,
        location=location,
        equivalent_descriptions=equivalent_descriptions,
    )
    if location == "intronic":
        result["offset"] = offset
    if coding_pos:
        result["coding_position"] = coding_pos
    if exon_num:
        result["nearest_exon"] = exon_num
        result["transcript_exons"] = len(exon_list)
        if return_exons and len(parsed_exon_list) > 0:
            result["exon_positions"] = parsed_exon_list
            result["cds"] = (cds_start, cds_end)
    return result


class UniProtResult(TypedDict):
    uniprot_id: str
    uniprot_name: str
    organism: str
    entry_type: str
    comments: Optional[str]
    url: str

def search_uniprot(protein_name: str = None, gene_name: str = None, max_results: int = 10) -> list[UniProtResult]:
    """
    Search UniProt to find protein IDs for a given protein name.
    Args:
        protein_name: The name of the protein to search for.
        gene_name: Filters results to those that mention this gene name in gene_primary. If no protein name is provided, this is used as the search term.
        max_results: The maximum number of results to return.
    Returns:
        A list of UniProtResult objects.
    """
    if not protein_name and not gene_name:
        raise ValueError("Must specify a gene or protein name")
    if gene_name and protein_name is None:
        protein_name = f"{gene_name}_HUMAN"
    
    query_kwargs = {
        "endpoint": "https://rest.uniprot.org/uniprotkb/search",
        "method": "GET",
        "params": {"query": protein_name, "fields": "id,gene_primary,protein_name,cc_function,cc_domain,ft_domain"},
        "description": "UniProt search API query"
    }
    search_response = query_api_with_retry(**query_kwargs)
    if not search_response.get("success"):
        return search_response
    result = search_response["result"]
    if not "results" in result:
        return search_response

    parsed_results = []
    for res in result["results"]:
        if gene_name:
            # if a gene name was used, filter for results that mention the given gene
            genes = [g["geneName"]["value"] for g in res.get("genes", [])]
            if gene_name.upper() not in genes:
                continue
        comments = []
        for com in res.get("comments", []):
            if isinstance(com["texts"], dict):
                comments.append(com["texts"]["value"])
            elif isinstance(com["texts"], list):
                comments.extend([c["value"] for c in com["texts"]])
        
        comments = "\n".join(comments) if len(comments) else None
        parsed_results.append(UniProtResult(
            uniprot_id=res['primaryAccession'],
            uniprot_name=res.get("uniProtkbId", ""),
            organism=res.get("organism", {}).get("scientificName", ""),
            entry_type=res.get("entryType", ""),
            comments=comments,
            url=f"https://www.ebi.ac.uk/interpro/protein/UniProt/{res['primaryAccession']}"
        ))

    return parsed_results[:max_results]


class InterproCitation(TypedDict):
    citation_id: str
    pmid: str
    title: str
    journal: str
    year: str
    doi: str

class InterProFeature(TypedDict):
    interpro_id: str
    name: str
    type: str
    description: Optional[str]
    citations: Optional[list[InterproCitation]]
    overlapping_regions: Optional[list[dict[str, any]]]

class BrowseUniProtResult(TypedDict):
    uniprot_id: str
    protein_length: int
    source_database: str
    url: str
    num_results: int
    results: list[InterProFeature]

# TODO change this name
def browse_uniprot(uniprot_id: str = None) -> BrowseUniProtResult:
    """
    Browse a UniProt protein features (InterPro domains) in detail.
    Only a preview of each InterPro features is returned.
    Use `browse_interpro` to fetch the full details of an InterPro feature.

    Args:
        uniprot_id: The UniProt ID of the protein to browse.
    Returns:
        A dictionary containing the browse results.
    """
    # TODO: handle case where uniprot_id is not provided
    endpoint = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{uniprot_id}"
    query_kwargs = {
        "endpoint": endpoint,
        "method": "GET",
        "description": "UniProt browse API query"
    }
    browse_response = query_api_with_retry(**query_kwargs)
    if not browse_response.get("success") or not "result" in browse_response:
        return browse_response

    result = browse_response["result"]
    protein_length, source_db = None, None
    parsed_results = []
    for feature in result["results"]:
        # get parent metadata if not already set
        parent_protein = feature.get("proteins", [{}])[0]
        if not protein_length:
            protein_length = parent_protein.get("length")
            source_db = parent_protein.get("source_database")
        
        # parse overlapping fragments
        fragments = []
        for entry in parent_protein.get("entry_protein_locations", []):
            fragments.extend(entry.get("fragments", []))
        
        if not "metadata" in feature:
            continue

        metadata = feature["metadata"]
        parsed_results.append(InterProFeature(
            interpro_id=metadata["accession"],
            name=metadata["name"],
            type=metadata["type"],
            overlapping_regions=fragments,
        ))
        if not protein_length:
            protein_length = feature.get("proteins", [{}])[0].get("protein_length")
        if not source_db:
            source_db = feature.get("metadata", {}).get("source_database")
    
    return BrowseUniProtResult(
        uniprot_id=uniprot_id,
        protein_length=protein_length,
        source_database=source_db,
        url=f"https://www.ebi.ac.uk/interpro/protein/UniProt/{uniprot_id}",
        num_results=result.get("count", 0),
        results=parsed_results,
    )



def browse_interpro(interpro_id: str) -> InterProFeature:
    """
    Browse a single InterPro feature in detail.
    Args:
        interpro_id: The InterPro ID of the feature to browse.
    Returns:
        A dictionary containing the browse results.
    """
    endpoint = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/{interpro_id}"
    query_kwargs = {
        "endpoint": endpoint,
        "method": "GET",
        "description": "Interpro API query"
    }
    browse_response = query_api_with_retry(**query_kwargs)
    if not browse_response.get("success"):
        return browse_response
    
    # print(json.dumps(browse_response, indent=1))
    
    result = browse_response["result"]
    metadata = result["metadata"]
    description = "\n".join([d["text"] for d in metadata["description"]])

    citations = []
    if metadata.get("literature") and isinstance(metadata["literature"], dict):
        for pub_id, entry in metadata["literature"].items():
            citations.append(InterproCitation(
                citation_id=pub_id,
                pmid=entry.get("PMID"),
                title=entry.get("title"),
                journal=entry.get("ISO_journal"),
                year=entry.get("year"),
                doi=entry.get("DOI_URL")
            ))
    
    return InterProFeature(
        interpro_id=metadata["accession"],
        name=metadata["name"]["name"],
        type=metadata["type"],
        description=description,
        citations=citations
    )


class EnsemblVEPResult(TypedDict, total=False):
    hgvs: str
    transcript_id: str
    assembly_name: str
    seq_region_name: str
    start: int
    end: int
    strand: int
    refseq_id: str
    biotype: str
    exon: str
    cds_start: int
    cds_end: int
    ref_allele: str  # used_ref
    variant_allele: str
    amino_acids: str
    protein_start: int
    protein_end: int
    phaplo: float
    ptriplo: float
    consequence_terms: list[str]

def search_ensembl_vep(hgvs: str) -> EnsemblVEPResult:
    """
    Search Ensembl VEP to get variant information.
    Args:
        hgvs: The HGVS variant to search for.
    Returns:
        A dictionary containing the Ensembl VEP results.
    """
    endpoint = f"https://rest.ensembl.org/vep/human/hgvs/{hgvs}"
    params = {
        "DosageSensitivity": 1,
        "mane": 1,
        "numbers": 1,
        "refseq": 1,
        "canonical": 1,
        "RiboseqORFs": 1,
        "SpliceAI": 1,
        "MaveDB": 1,
        "domains": 1,
        "LOEUF": 1,
        "UTRAnnotator": 1,
    }
    query_kwargs = {
        "endpoint": endpoint,
        "params": params,
        "method": "GET",
        "description": "Ensembl VEP API query",
    }
    vep_response = query_api_with_retry(**query_kwargs)
    if not vep_response.get("success") or not "result" in vep_response or not len(vep_response["result"]):
        return vep_response
    result = vep_response["result"][0]

    data = EnsemblVEPResult(
        hgvs=hgvs,
        assembly_name=result["assembly_name"],
        seq_region_name=result["seq_region_name"],
        start=result["start"],
        end=result["end"],
        strand=result["strand"],
        most_severe_consequence=result.get("most_severe_consequence"),
    )

    refseq_id = hgvs.split(":")[0]
    for tc in result.get("transcript_consequences", []):
        if tc["transcript_id"] == refseq_id:
            #for k in ["biotype", "gene_symbol", "cds_start", "cds_end", "protein_start", "protein_end", "used_ref", "variant_allele", "amino_acids", "consequence_terms"]:
            for k in tc.keys():
                data[k] = tc.get(k)
            break
    return data
    


class GencodeResult(TypedDict):
    name: str
    strand: str
    exon_frames: list[int]
    exon_starts: list[int]
    exon_ends: list[int]
    cds_start: int
    cds_end: int

def search_gencode(chrom: str, start: int, end: int) -> GencodeResult:
    """
    Search GENCODE (via UCSC Genome Browser) to get transcript exon information.

    Queries the UCSC Genome Browser API for GENCODE Basic V48 transcript data
    overlapping the specified genomic region on the GRCh38/hg38 assembly.

    Args:
        chrom: The chromosome name (e.g. "chr17").
        start: The start position of the genomic region (0-based).
        end: The end position of the genomic region.

    Returns:
        A GencodeResult containing the transcript name, strand, exon frames,
        exon start/end positions, and CDS start/end coordinates for the first
        overlapping GENCODE transcript. Returns None if no transcript is found
        or the API call fails.
    """
    params = {
        "chrom": chrom,
        "start": start,
        "end": end,
        "track": "wgEncodeGencodeBasicV48",
        "genome": "hg38",
    }

    endpoint = f"https://api.genome.ucsc.edu/getData/track"
    query_kwargs = {
        "endpoint": endpoint,
        "params": params,
        "method": "GET",
        "description": "UCSC Genome Browser API query",
    }
    ucsc_response = query_api_with_retry(**query_kwargs)
    if not ucsc_response.get("success") or not "result" in ucsc_response:
        print(f"Error fetching GENCODE track for {chrom}:{start}-{end}: {ucsc_response['error']}")
        return None
    
    result = ucsc_response["result"]
    if not "wgEncodeGencodeBasicV48" in result or not len(result["wgEncodeGencodeBasicV48"]):
        print(f"No GENCODE track found for {chrom}:{start}-{end}")
        return None
    track_result = result["wgEncodeGencodeBasicV48"][0]

    exon_frames = [int(i) for i in track_result["exonFrames"].split(",") if len(i)]
    exon_starts = [int(i) for i in track_result["exonStarts"].split(",") if len(i)]
    exon_ends = [int(i) for i in track_result["exonEnds"].split(",") if len(i)]

    return GencodeResult(
        name=track_result["name"],
        strand=track_result["strand"],
        exon_frames=exon_frames,
        exon_starts=exon_starts,
        exon_ends=exon_ends,
        cds_start=track_result["cdsStart"],
        cds_end=track_result["cdsEnd"],
    )


def fetch_dna_sequence(chrom: str, start: int, end: int, rev_comp=False) -> str | None:
    """
    Search UCSC Genome Browser to get dna sequence for a given genomic region.
    Args:
        chrom: The chromosome to search for.
        start: The start position of the region.
        end: The end position of the region.
        rev_comp: If true, returns the reverse complement of the sequence.
    Returns:
        The DNA sequence.
    """
    params = {
        "chrom": chrom,
        "start": start,
        "end": end,
        "genome": "hg38",
    }
    endpoint = f"https://api.genome.ucsc.edu/getData/sequence"
    query_kwargs = {
        "endpoint": endpoint,
        "params": params,
        "method": "GET",
        "description": "UCSC Genome Browser API query",
    }
    if rev_comp:
        params["revComp"] = 1
    ucsc_response = query_api_with_retry(**query_kwargs)
    if not ucsc_response.get("success") or not "result" in ucsc_response:
        print(f"Error fetching DNA sequence for region {chrom}:{start}-{end}: {ucsc_response['error']}")
        return None
    
    result = ucsc_response["result"]
    if not "dna" in result or len(result["dna"]) == 0:
        print(f"No DNA sequence found for region {chrom}:{start}-{end}")
        return None

    return result["dna"].upper()


class GnomadResult(TypedDict):
    transcript_id: str
    hgvs: str
    hgvsc: str
    consequence: str
    pos: int
    exome: dict[str, float]
    error: Optional[str]

def search_gnomad(gene_symbol: str, hgvsc: str = None) -> list[GnomadResult] | GnomadResult:
    """
    Query gnomAD v4 for variant allele frequency data for a given gene.

    Args:
        gene_symbol: The HGNC gene symbol to query (e.g. "BRCA1").
        hgvsc: Optional HGVSc notation to filter to a single variant
            (e.g. "NM_007294.4:c.5266dupC"). Only the coding change portion
            after the colon is used for matching against gnomAD records.

    Returns:
        If hgvsc is provided: a single GnomadResult for the matching variant,
        or a GnomadResult with an error field if the variant is not found.
        If hgvsc is not provided: a list of all GnomadResult entries for variants
        in the gene, each containing transcript_id, hgvs, hgvsc, consequence,
        chromosomal position, and exome allele count/frequency data.
    """
    query_tmpl = """
    query VariantsInGene {
        gene(gene_symbol: "BRCA1", reference_genome: GRCh38) {
            name
            reference_genome
            gnomad_constraint {
                pli
                oe_lof
                oe_mis
                oe_syn
            }
            variants(dataset: gnomad_r4) {
                transcript_id
                pos
                hgvs
                hgvsc
                hgvsp
                consequence
                joint {
                    ac
                    an
                    hemizygote_count
                    homozygote_count
                }
            }
        }
    }
    """
    query = query_tmpl.replace("\"BRCA1\"", f"\"{gene_symbol}\"")
    query_kwargs = {
        "endpoint": "https://gnomad.broadinstitute.org/api",
        "method": "POST",
        "json_data": {"query": query},
        "headers": {"Content-Type": "application/json"},
    }
    result = query_api_with_retry(**query_kwargs)
    if not result.get("success") or not "result" in result:
        return result
    result = result["result"]

    var_filter = None
    if hgvsc is not None:
        var_filter = hgvsc.split(":")[-1]
    
    variants = result["data"]["gene"]["variants"]
    ret_variants = []
    for entry in variants:
        if var_filter is None or entry["hgvsc"] == var_filter:
            ret_variants.append(GnomadResult(**entry))
    
    if var_filter is not None:
        if len(ret_variants) == 0:
            return GnomadResult(
                gene_symbol=gene_symbol,
                hgvs=hgvsc,
                hgvsc=hgvsc,
                consequence=None,
                pos=None,
                exome=None,
                error="No variants found for this gene and variant.",
            )
        return ret_variants[0]
    
    return ret_variants



class ClinGenDosageKBResult(TypedDict, total=False):
    gene_symbol: str
    hgnc_id: str
    haploinsufficiency: str
    triplosensitivity: str
    online_report: str
    date: str
    message: str

def search_clingen_dosage_kb(gene_symbol: str) -> list[ClinGenDosageKBResult]:
    """
    Search ClinGen to find dosage sensitivity evidence for a given gene symbol.

    Reads from a locally cached CSV of ClinGen gene dosage curations, downloading
    fresh data from ClinGen if the cache file does not yet exist.

    Args:
        gene_symbol: The HGNC gene symbol to look up (e.g. "BRCA1").

    Returns:
        A ClinGenDosageKBResult containing haploinsufficiency score,
        triplosensitivity score, and a link to the online curation report.
        If no curation is found for the gene, returns a result with only
        the gene_symbol and a message field indicating no data was found.
    """
    # NOTE: there's no API for ClinGen, so we need to download the data manually
    dosage_file = os.path.join(DATA_DIR, "clingen_gene_dosage.csv")
    if not os.path.exists(dosage_file):
        # download fresh data
        download_url = "https://search.clinicalgenome.org/kb/gene-dosage/download"
        response = requests.get(download_url)
        lines = response.text.split("\n")
        lines = [lines[4]] + lines[6:]
        payload = "\n".join(lines)
        payload = StringIO(payload)
        data = pd.read_csv(payload)
        col_renames = {k:"_".join(k.split(" ")).lower() for k in data.columns}
        data.rename(columns=col_renames, inplace=True)
        data.to_csv(dosage_file, index=False)  # cache to disk
    else:
        data = pd.read_csv(dosage_file)
    
    filt_data = data.loc[data["gene_symbol"] == gene_symbol]
    if len(filt_data):
        clingen_data = filt_data.to_dict(orient="records")[0]
        return ClinGenDosageKBResult(**clingen_data)
    
    return ClinGenDosageKBResult(
        gene_symbol=gene_symbol,
        message="No ClinGen dosage sensitivity curation found for this gene yet.",
    )


def search_omim(mim_number: str) -> dict[str, any]:
    """
    Fetch full OMIM entry data for a given MIM number.

    Parameters
    ----------
    mim_number : int
        The OMIM MIM ID (e.g., 605543).

    Returns
    -------
    dict
        Full JSON response from the OMIM API for the given entry.
    """

    base_url = "https://api.omim.org/api/entry"
    
    params = {
        "mimNumber": mim_number,
        "include": "all",      # request all available sections
        "format": "json",
        "apiKey": os.getenv("OMIM_API_KEY")
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()

    return response.json()


def search_alt_splicing_events(gene_symbol: str) -> list[dict[str, any]]:
    """
    Searches supplementary files from Felker et al. (2022), Lim et al. (2020), and Mittal et al. (2022)
    for alt splicing events (poison exons, uORF, NAT, NMD events).
    """
    hits = {}
    # gene_symbol, alt_gene_list, PE_*
    sourceA = pd.read_excel("data/felker_2022_PE.xlsx")
    record = sourceA.loc[
        (sourceA['gene_symbol'] == gene_symbol) | 
        (sourceA['alt_gene_list'].str.contains(gene_symbol))
    ].dropna(axis=1, how='all').to_dict(orient="records")
    hits["Felker et al. (2022)"] = record

    # Gene, AS type, Coordinates
    sourceB = pd.read_excel("data/lim_2020_nmd_events.xlsx")
    record = sourceB.loc[sourceB['Gene'] == gene_symbol].dropna(axis=1, how='all').to_dict(orient="records")
    hits["Lim et al. (2020)"] = record

    # Gene, uORF, NAT, PE, Haploinsufficient (ClinGen), Pathogenic (ClinVar), Contains Known Splice Variants, pLI
    sourceC = pd.read_csv("data/mittal_2022_uORF_NAT_PE.csv", index_col=0)
    record = sourceC.loc[sourceC['Gene'] == gene_symbol].dropna(axis=1, how='all').to_dict(orient="records")
    hits["Mittal et al. (2022)"] = record

    return hits


if __name__ == "__main__":
    # # ensembl vep
    # result = search_ensembl_vep("NM_024312.5:c.3503_3504del")
    # print(json.dumps(result, indent=2))

    # # fetch gene name from clinvar
    # var_info = search_mutalyzer("NM_024312.5:c.3503_3504del")
    # print(json.dumps(var_info, indent=2))

    # gene = var_info["gene_id"]

    # fetch gnomad variants
    results = search_gnomad(gene_symbol="ABCA4")
    print(json.dumps(results, indent=2))

    # # fetch domains from uniprot
    # results = search_uniprot(gene_name=gene)
    # print(json.dumps(results[0], indent=2))
    # uniprot_id = results[0]["uniprot_id"]
    # browse_result = browse_uniprot(uniprot_id)
    # print(json.dumps(browse_result, indent=2))

    # # select an interpro feature to browse in depth
    # result = browse_interpro("IPR060126")
    # print(json.dumps(result, indent=2))