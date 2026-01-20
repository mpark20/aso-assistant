"""
From https://github.com/rlresearch/dr-tulu/tree/c786700ca63d323ed32f8749eec7ef1562c8ee8c/agent/dr_agent/mcp_backend/apis
"""
import asyncio
import json
import os
import requests
import time
from functools import cache
from typing import TypedDict, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from src.tools.utils import query_api_with_retry

load_dotenv()

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
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    esearch_url = f"{base_url}/esearch.fcgi"
    esearch_params = {
        "db": database,
        "term": search_term,
        "retmode": "json",
        "retmax": 100,
        "usehistory": "y",  # Use history server to store results
    }

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

async def search_gpt(query: str) -> str:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.responses.create(
        model="gpt-5",
        tools=[{"type": "web_search"}],
        input=query,
        timeout=300,
        reasoning={"effort": "low"},
    )
    # access citations via response.message.content[0].annotations
    return response.to_dict()


class BrowseWebpageResult(TypedDict):
    url: str
    text: str
    success: bool

async def browse_webpage(url: str) -> BrowseWebpageResult:
    """
    Crawl a given webpage url and return its contents.
    Credit: https://github.com/rlresearch/dr-tulu/blob/main/agent/dr_agent/mcp_backend/local/crawl4ai_fetcher.py#L25
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
    type: str
    equivalent_descriptions: Optional[list[str]]
    coding_position: Optional[int]
    affected_exon: Optional[int]
    trasnscript_exons: Optional[int]

def search_mutalyzer(variant: str) -> MutalyzerResult:
    """
    Normalize a genetic variant using Mutalyzer and return the normalized HGVS.
    Raises ValueError if Mutalyzer returns an error or an unexpected response.

    Parameters:
    variant : str
        A genetic variant description (e.g. "NM_000059.3:c.7790G>A")
    """
    import requests
    from urllib.parse import quote

    if not variant or not isinstance(variant, str):
        raise ValueError("variant must be a non-empty string")

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

    # TODO: skip EINTRONIC error codes
    if data.get("errors") or data.get("custom", {}).get("errors"):
        errors = data.get("errors") if "errors" in data else data.get("custom", {}).get("errors")
        raise ValueError(f"Mutalyzer error(s), code {response.status_code}: {errors}")

    normalized = (
        data.get("normalized_description")
        or data.get("corrected_description")
    )

    if not normalized:
        raise ValueError(f"Unexpected Mutalyzer response: {data}")
    
    # get extra metadata
    variants = data.get("normalized_model", {}).get("variants", [])
    loc_info = variants[0].get("location", {}) if len(variants) else None
    if not loc_info or not "type" in loc_info or loc_info.get("type") not in ["point", "range"]:
        exon_info, coding_pos = None, None
    else:
        # special case for variants in c. notation: exon positions may be provided
        coding_pos = loc_info.get("position") if loc_info["type"] == "point" else loc_info.get("start", {}).get("position")
        exon_info = data.get("selector_short", {}).get("exon")
        exon_list = exon_info.get("c") if exon_info else None
        exon_num = None
        if exon_list and coding_pos:
            for i, (start, end) in enumerate(exon_list):
                try:
                    start, end = start.strip(), end.strip()
                    s, e = int(start), int(end)
                    if s <= coding_pos <= e:
                        exon_num = i + 1
                        break
                except ValueError:
                    break
    
    equivalent_descriptions = data.get("equivalent_descriptions", [])
    protein_desc = data.get("protein", {}).get("description")
    if protein_desc:
        equivalent_descriptions.append(protein_desc)

    result = MutalyzerResult(
        normalized=normalized,
        type=data.get("corrected_model", {}).get("type"),
        equivalent_descriptions=equivalent_descriptions,
    )
    if coding_pos:
        result["coding_position"] = coding_pos
    if exon_num:
        result["affected_exon"] = exon_num
        result["transcript_exons"] = len(exon_list)
    return result


class UniProtResult(TypedDict):
    uniprot_id: str
    uniprot_name: str
    organism: str
    entry_type: str
    url: str

def search_uniprot(protein_name: str = None, max_results: int = 10) -> list[UniProtResult]:
    """
    Search UniProt to find protein IDs for a given protein name.
    Args:
        protein_name: The name of the protein to search for.
        max_results: The maximum number of results to return.
    Returns:
        A list of UniProtResult objects.
    """
    query_kwargs = {
        "endpoint": "https://rest.uniprot.org/uniprotkb/search",
        "method": "GET",
        "params": {"query": protein_name},
        "description": "UniProt search API query"
    }
    search_response = query_api_with_retry(**query_kwargs)
    if not search_response.get("success"):
        return search_response
    result = search_response["result"]

    parsed_results = []
    for res in result["results"]:
        parsed_results.append(UniProtResult(
            uniprot_id=res['primaryAccession'],
            uniprot_name=res.get("uniProtkbId", ""),
            organism=res.get("organism", {}).get("scientificName", ""),
            entry_type=res.get("entryType", ""),
            url=f"https://www.ebi.ac.uk/interpro/protein/UniProt/{res['primaryAccession']}"
        ))
    return parsed_results[:max_results]


class InterProFeature(TypedDict):
    metadata: dict[str, any]
    overlapping_regions: list[dict[str, any]]

class BrowseUniProtResult(TypedDict):
    uniprot_id: str
    protein_length: int
    source_database: str
    url: str
    num_results: int
    results: list[InterProFeature]

def browse_uniprot(uniprot_id: str = None) -> BrowseUniProtResult:
    f"""
    Browse a UniProt protein's overlapping InterPro features in detail.
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
    if not browse_response.get("success"):
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
        parsed_results.append(InterProFeature(
            metadata=feature.get("metadata", {}),
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



# special formatting functions for ClinVar results
def clinvar_formatter(data: dict[str, any]) -> dict[str, any]:
    # Extract location (prefer GRCh38 if available)
    locs = data.get("variation_set", [{}])[0].get("variation_loc", [])
    primary_loc = next((l for l in locs if l.get("assembly_name") == "GRCh38"), locs[0] if locs else {})
    chr_ = primary_loc.get("chr")
    assembly = primary_loc.get("assembly_name")
    start = primary_loc.get("start")
    stop = primary_loc.get("stop")

    # Germline classification (pathogenicity)
    germline = data.get("germline_classification", {})
    pathogenicity = germline.get("description")
    review_status = germline.get("review_status")
    last_evaluated = germline.get("last_evaluated")

    # Associated traits
    trait_set = germline.get("trait_set", [])
    associated_traits = [t.get("trait_name") for t in trait_set if t.get("trait_name")]

    # Molecular consequences
    molecular_consequences = data.get("molecular_consequence_list", [])

    # Build final flattened dict
    url = "https://www.ncbi.nlm.nih.gov/clinvar/variation/" + data.get("uid")
    formatted_result = {
        "uid": data.get("uid"),
        "accession": data.get("accession"),
        "title": data.get("title"),
        "protein_change": data.get("protein_change"),
        "chromosome": chr_,
        "assembly": assembly,
        "start": start,
        "stop": stop,
        "molecular_consequences": molecular_consequences,
        "pathogenicity": pathogenicity,
        "review_status": review_status,
        "last_evaluated": last_evaluated,
        "associated_traits": associated_traits,
        "url": url,
    }
    return formatted_result


if __name__ == "__main__":
    # for variant in ['NM_000392.5:c.3399_3400delTT', 'NM_000548.3:c.3598C>T', 'NM_145239.2:c.649dupC', 'NM_001037.4:c.363C>G']:
    #     response = search_ncbi("clinvar", variant)
    #     if not len(response["results"]):
    #         ref, change = variant.split(":")
    #         variant = f"{ref.split('.')[0]}:{change}"
    #         response = search_ncbi("clinvar", variant)
    #     print(response)
    #     print()
    results = search_uniprot("abca4_human")
    uniprot_id = results[0]["uniprot_id"]
    browse_result = browse_uniprot(uniprot_id)
    print(json.dumps(browse_result, indent=2))