"""
Module for fetching PubMed abstracts and full text articles.
"""
import json
import re
import requests
import time
import xml.etree.ElementTree as ET
from pymed.article import PubMedArticle
from typing import Literal, Optional, TypedDict

NCBI_RPS = 3


class PubMedResult(TypedDict, total=False):
    pmid: str
    pmcid: str
    title: str
    pubdate: str
    journal: Optional[str]


def pubmed_formatter(data: dict[str, any], database: Literal["pubmed", "pmc"] = "pubmed") -> dict[str, any]:
    """
    PMC
    {'uid': '12975024',
   'pubdate': '2026 Apr 1',
   'epubdate': '2026 Feb 23',
   'printpubdate': '2026 Apr 1',
   'source': 'Curr Opin Neurol',
   'authors': [{'name': 'Ryvlin P', 'authtype': 'Author'},
    {'name': 'Rheims S', 'authtype': 'Author'}],
   'title': 'SUDEP update 2026: recent advances in experimental and clinical research.',
   'volume': '39',
   'issue': '2',
   'pages': '123-130',
   'articleids': [{'idtype': 'pmid', 'value': '41735792'},
    {'idtype': 'doi', 'value': '10.1097/WCO.0000000000001463'},
    {'idtype': 'pii', 'value': '00019052-990000000-00319'},
    {'idtype': 'pmcid', 'value': 'PMC12975024'}],
   'fulljournalname': 'Current opinion in neurology'}

    PUBMED
   'uid': '41714109',
   'pubdate': '2026 Feb 19',
   'epubdate': '2026 Feb 19',
   'source': 'RNA',
   'title': 'Ancestral intronic splicing regulatory elements in the SCNα gene family.',
   'articleids': [{'idtype': 'pubmed', 'idtypen': 1, 'value': '41714109'},
    {'idtype': 'doi', 'idtypen': 3, 'value': '10.1261/rna.080730.125'},
    {'idtype': 'pii', 'idtypen': 4, 'value': 'rna.080730.125'}],
   'fulljournalname': 'RNA (New York, N.Y.)'},
    """
    formatted_result = PubMedResult(
        title=data.get("title"),
        pubdate=data.get("pubdate"),
        journal=data.get("fulljournalname"),
    )
    if database == "pubmed":
        formatted_result["pmid"] = data.get("uid")
    elif database == "pmc":
        formatted_result["pmcid"] = data.get("uid")
    
    for entry in data.get("articleids", []):
        id_type = entry["idtype"]
        if id_type == "pubmed":
            id_type = "pmid"
        elif id_type == "pmc":
            id_type = "pmcid"
        formatted_result[id_type] = entry["value"]
    return formatted_result


class PubMedPaper(TypedDict, total=False):
    pmid: str
    title: str
    journal: str
    date: str
    abstract: str
    pmcid: str
    full_text: list[str]

def fetch_pubmed(pmids: list[str]) -> list[PubMedPaper]:
    try:
        pmid_to_pmcid = _pmid_to_pmcid(pmids)
    except Exception as e:
        pmid_to_pmcid = {}
    papers = []
    for pmid in pmids:
        # get overview from pubmed
        paper_snippet = fetch_pubmed_abstracts([pmid])[0]
        paper_sections = []
        # if available, get the full text from pmc
        if pmid in pmid_to_pmcid:
            pmcid = pmid_to_pmcid[pmid]
            paper_data = fetch_pmc_fulltext([pmcid])
            if pmcid in paper_data:
                paper_sections = paper_data[pmcid]["full_text"]
                paper_snippet["pmcid"] = pmcid
                paper_snippet["full_text"] = paper_sections
            
            if len(pmids) > 1:
                time.sleep(1 / NCBI_RPS)

        papers.append(paper_snippet)
    return papers

def fetch_pubmed_abstracts(pmids: list[str], batch_size: int = 200) -> list[dict[str, any]]:
    """
    Fetch PubMed abstracts for a list of PMIDs.

    Args:
        pmids (list[str]): List of PMIDs to fetch abstracts for.
        batch_size (int): Batch size for fetching abstracts.

    Returns:
        list[PubMedPaper]: List of papers with PubMed abstracts.
    """
    papers = []
    for i in range(0, len(pmids), batch_size):
        batch_pmids = pmids[i:i+batch_size]
        # Fetch the XML article
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": batch_pmids}
        res = requests.get(base_url, params=params)
        res.raise_for_status()
        res_xml = res.text
        for c in ['<i>', '</i>', '<b>', '</b>', '<u>', '</u>']:
            res_xml = res_xml.replace(c, "")
        
        # Parse as XML
        root = ET.fromstring(res_xml)

        # Loop over the articles and construct article objects
        batch_papers = []
        for article in root.iter("PubmedArticle"):
            article = PubMedArticle(xml_element=article)
            article_dict = article.toDict()
            article_dict.pop("xml")
            curr_id = article_dict.pop("pubmed_id", "")
            curr_id = curr_id.split('\n')[0]
            batch_papers.append({
                "pmid": curr_id,
                "title": article_dict.pop("title", curr_id),
                "journal": article_dict.pop("journal", ""),
                "date" : str(article_dict.pop("publication_date", "")),
                "abstract": article_dict.pop("abstract", ""),
                # "metadata": article_dict,
            })
        papers.extend(batch_papers)
        time.sleep(1 / NCBI_RPS)
    return papers



def fetch_pmc_fulltext(pmc_ids: list[str], ignore_sections=["ref", "comp_int", "auth_cont"]) -> dict[str, str]:
    """
    Fetch full text articles from PMC OA. Articles passed that are not in the PMC OA will be ignored.

    Args:
        pmc_ids (list[str]): List of PMCIDs to try fetching articles for.

    Returns:
        dict[str, str]: Dictionary of PMCIDs and their full text content.
    """
    paper_contents = {}
    # pmids that were not oa will get skipped
    for pmc_id in pmc_ids:
        id_str = str(pmc_id)
        id_str = "PMC" + id_str if not id_str.startswith("PMC") else id_str
        endpoint = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{id_str}/unicode"
        skip_paper = False
        while True:
            response = requests.get(endpoint)
            if response.status_code == 429:
                # if rate limited, just wait and try again
                time.sleep(1)
                continue
            if response.status_code != 200:
                skip_paper = True
                break
            try:
                data = response.json()
            except:
                skip_paper = True
                break
            if len(data) != 1 or not data[0].get('documents'):
                # biocjson only handles 1 document at a time, event though the response is formatted as a list of documents
                skip_paper = True
            break

        if skip_paper:
            continue

        data = data[0]
        sections, _ = _parse_biocjson(pmc_id, data, ignore_sections=ignore_sections)
        if len(sections) > 0:
            paper_contents[pmc_id] = {
                "pmid": pmc_id,
                "full_text": sections
            }
        time.sleep(1 / NCBI_RPS)
    return paper_contents


def url_to_pmid(url: str) -> str | None:
    """Extract PMID from PubMed URL."""
    match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=(\d+)", url)
    if match:
        return match.group(1)
    return None

def url_to_pmcid(url: str) -> str | None:
    """Extract PMCID from PMC URL."""
    match = re.search(r"pmc\.ncbi\.nlm\.nih\.gov/articles/PMC(\d+)", url)
    if match:
        return match.group(1)
    return None


def _pmid_to_pmcid(pmids: list[str]) -> dict[str, str]:
    """
    Convert a list of PMIDs to a PMCIDs using NCBI E-utilities.
    Returns a dictionary mapping PMIDs to PMCIDs, excluding PMIDs without a corresponding PMCID.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": ','.join(pmids),
        "retmode": "json"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    # If the PMID has a corresponding PMCID, it will appear in result
    result = data.get("result", {})
    pmids = result["uids"]
    _pmid_to_pmcid_map = {}
    for pmid in pmids:
        for article_id in result[pmid].get("articleids", []):
            if article_id["idtype"] == "pmc":
                _pmid_to_pmcid_map[pmid] = article_id["value"]
                break
    return _pmid_to_pmcid_map


def _parse_biocjson(
    pmid: str, data: dict[str, any],
    ignore_sections: list[str] = ["ref", "comp_int", "auth_cont"]
) -> tuple[list[str], dict[str, any]]:
    """
    Parse a BiocJSON file into a list of paragraphs and metadata.
    """
    payload = data['documents'][0]  # bioc json will be a list of 1 document
    passages = payload['passages']
    metadata = {
        "PMID": pmid,
        "access_type": "full text",
        "source": "pubmed",
    }
    section2title = {}  # {section_type: section_title}
    section2text = {}  # {section_title: section_text}
    for p in passages:
        infons = p['infons']
        section_type = infons.get('section_type').lower()
        if not len(section_type):
            continue
        type_ = infons.get('type', '')
        if section_type in ignore_sections:
            continue
        if section_type == 'title' and type_ == 'front':
            # add title and journal to top-level metadata (these will be at the beginning of the document
            # so it will be populated by the time we get to the first paragraph)
            metadata[section_type] = metadata.get(section_type, "") + p['text']
            if infons.get('source'):
                metadata['journal'] = infons['source']
        elif 'title' in type_:
            # # the section title will be the first entry with the given section_type, so it will be the start of the text
            section2title[section_type] = p['text']
        elif type_ == 'paragraph' or type_ == 'abstract':
            if type_ == "abstract":
                header = "Abstract"
            else:
                header = section2title.get(section_type, f"Section {len(section2text)+1}")
            p['text'] = f"### {header}:\n{p['text']}"
            # merge paragraphs if they are in the same section
            section2text[header] = section2text.get(header, "") + f"\n{p['text']}"
    
    return list(section2text.values()), metadata



if __name__ == "__main__":
    pmids = ["30254639", "41160699", "34126256"]
    _pmid_to_pmcid = _pmid_to_pmcid(pmids)
    for pmid in pmids:
        # get overview from pubmed
        paper_snippet = fetch_pubmed_abstracts([pmid])[0]
        paper_sections = []
        # if available, get the full text from pmc
        if pmid in _pmid_to_pmcid:
            pmcid = _pmid_to_pmcid[pmid]
            paper_data = fetch_pmc_fulltext([pmcid])
            paper_sections = paper_data[pmcid]["full_text"]
            time.sleep(1 / NCBI_RPS)
            paper_snippet["full_text"] = paper_sections

        with open(f"data/{pmid}.json", 'w') as f:
            json.dump(paper_snippet, f, indent=2)
