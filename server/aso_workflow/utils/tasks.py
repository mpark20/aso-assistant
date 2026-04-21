"""
Core workflows for the ASO Variant Assessment.
Composed of various combinations of database searches and postprocessing.
"""
import json
import os
import pandas as pd
from typing import Optional, TypedDict, Dict, List, Any
from aso_workflow.utils.apis import *
from aso_workflow.utils.clinvar import fetch_clinvar_rcv, ClinVarResult
from aso_workflow.utils.pubmed import fetch_pubmed, PubMedPaper, fetch_pubmed_abstracts


__all__ = ['fetch_clinical_context', 'fetch_protein_context', 'fetch_transcript_context']


# HELPER FUNCTIONS

class ClinicalContext(TypedDict):
    mutalyzer: MutalyzerResult
    clinvar: ClinVarResult | NCBIResult  # if there were no results, return the placeholder NCBIResult
    clingen: ClinGenDosageKBResult

def fetch_clinical_context(hgvs: str) -> ClinicalContext:
    """Fetch variant clinical overview, containing basic information from Mutalyzer, ClinVar, and PubMed."""

    # normalize variant name using Mutalyzer
    # TODO: add some central caching mechanism for normalization, as it'll be used frequently
    mutalyzer_result = search_mutalyzer(hgvs)
    norm_hgvs = mutalyzer_result.get("normalized")
    gene_id = mutalyzer_result.get("gene_id")

    # fetch clinical context from ClinVar
    if not norm_hgvs or ":" not in norm_hgvs:
        return ClinicalContext(
            mutalyzer=mutalyzer_result,
            clinvar={"error": "Could not normalize variant"},
            clingen=ClinGenDosageKBResult(gene_symbol=gene_id or "", message="Could not normalize variant"),
        )
    refseq, coding_change = norm_hgvs.split(":", 1)
    refseq_id = refseq.split(".")[0]
    ncbi_result = search_ncbi(database="clinvar", search_term=f"{refseq_id} AND {coding_change}[VARNAME] AND {gene_id}[GENE]")
    if not "results" in ncbi_result or len(ncbi_result["results"]) == 0:
        return {
            "mutalyzer": mutalyzer_result,
            "clinvar": ncbi_result,
        }
    clinvar_result = ncbi_result["results"][0]

    # get curated PubMed citations from ClinVar RCV comments
    supporting_submissions = clinvar_result.get("supporting_submissions") or {}
    rcv_id_list = supporting_submissions.get("rcv", []) if isinstance(supporting_submissions, dict) else []
    rcv_id_list = rcv_id_list or []
    rcv_results = []
    for rcv_id in rcv_id_list:
        rcv_records = fetch_clinvar_rcv(rcv_id, add_citations=True)
        rcv_records = [{"rcv": rcv_id, **r} for r in (rcv_records or []) if r.get("classification") and r["classification"].get("comment") is not None]
        if len(rcv_records):
            rcv_results.extend(rcv_records)
    if isinstance(clinvar_result.get("supporting_submissions"), dict):
        clinvar_result["supporting_submissions"]["comments"] = rcv_results
        clinvar_result["supporting_submissions"].pop("rcv", None)
        clinvar_result["supporting_submissions"].pop("scv", None)

    # get dosage sensitivity evidence from ClinGen. MUST run `download_data.py` to download resource
    clingen_data = search_clingen_dosage_kb(gene_id)
    
    return ClinicalContext(
        mutalyzer=mutalyzer_result,
        clinvar=clinvar_result,
        clingen=clingen_data,
    )


class ProteinContext(TypedDict):
    hgvs: str
    gene_id: str
    exon_aa_range: list[int]
    protein_aa_length: int
    uniprot_id: str
    domains: list[InterProFeature]
    message: Optional[str]

def fetch_protein_context(hgvs: str) -> ProteinContext:
    """Fetch overlapping protein features from a given HGVS variant"""
    message = None

    # get exon coordinates from mutalyzer
    result = search_mutalyzer(hgvs, return_exons=True)
    if result.get("intronic"):
        message = f"Variant {hgvs} is intronic. Using the nearest exon instead."
    
    if not "cds" in result:
        # TODO: decide on a consistent error handling strategy, i.e. None vs. error msg
        print(f"No cds info found for {hgvs}. Please verify that the transcript is a valid coding region.")
        return None
    
    cds_start, cds_end = result["cds"]
    exon_index = result.get("nearest_exon")
    exon_positions = result.get("exon_positions")
    if exon_index is None or not exon_positions or exon_index < 1 or exon_index > len(exon_positions):
        print(f"No valid exon position found for {hgvs}.")
        return None
    exon_cdna_range = exon_positions[exon_index - 1]
    gene_id = result["gene_id"]
    hgvs = result["normalized"]

    # convert exon coordinates to protein coordinates
    apply_fn = lambda x: _coding_to_protein_coord(x, cds_start, cds_end)
    exon_aa_range = list(map(apply_fn, exon_cdna_range))

    # get UniProt ID from refseq ID
    uniprot_id = None
    try:
        refseq_id = hgvs.split(':')[0].split('.')[0]
        result = search_uniprot(refseq_id, gene_name=gene_id)
        if not result or not isinstance(result, list):
            print(f"Error fetching UniProt for {hgvs}: no results found")
            return None
        uniprot_id = result[0]["uniprot_id"]
    except Exception as e:
        print(f"Error fetching gene id for {hgvs}: {e}")
        return None

    # fetch InterPro features overlapping the protein entry
    uniprot_info = browse_uniprot(uniprot_id)
    if not uniprot_info or "protein_length" not in uniprot_info:
        print(f"Error fetching UniProt browse for {hgvs}: invalid response")
        return None
    protein_length = uniprot_info["protein_length"]
    interpro_hits = uniprot_info.get("results") or []

    # filter to entries overlapping the specific exon
    exon_aa_range[0] = exon_aa_range[0] or 1
    exon_aa_range[1] = exon_aa_range[1] or protein_length
    overlapping_domains = _filter_ipr_domains(interpro_hits, exon_aa_range, add_details=True)

    return ProteinContext(
        hgvs=hgvs,
        gene_id=gene_id,
        exon_aa_range=exon_aa_range,
        protein_aa_length=protein_length,
        uniprot_id=uniprot_id,
        domains=overlapping_domains,
        message=message,
    )


class ExonInfo(TypedDict):
    number: int  # 1-indexed
    length: int
    coordinates: tuple[int, int]  # (start, end) in genomic coordinates
    frame: int
    sequence: Optional[str]

class TranscriptContext(TypedDict):
    hgvs: str
    refseq_id: str
    chromosome: str
    transcript_exons: int
    exon_number: int
    flanking_exons: list[ExonInfo]
    message: Optional[str]

def fetch_transcript_context(hgvs: str) -> TranscriptContext:
    """Fetch exon context from a given HGVS variant"""
    # fetch genomic coordinates, chromosome, and strand from Ensembl
    ensembl_result = search_ensembl_vep(hgvs)
    if "error" in ensembl_result:
        print(f"Error fetching exon context for {hgvs}: {ensembl_result['error']}")
        return None
    
    chrom, start, end, strand, transcript_id = (
        ensembl_result["seq_region_name"],
        ensembl_result["start"],
        ensembl_result["end"],
        ensembl_result["strand"],
        ensembl_result["transcript_id"]
    )

    # fetch exon frames from UCSC Genome Browser (we only need the start position to fetch the overlapping transcript)
    ucsc_result = search_gencode(chrom, start, start+1, ensembl_id=transcript_id)  # UCSC excludes the end position
    if not ucsc_result or "error" in ucsc_result:
        print(f"Error fetching exon context for {hgvs}: {ucsc_result['error']}")
        return None
    
    # NOTE: UCSC returns coordinates on open-closed intervals, i.e. [start, end)
    exon_frames, exon_starts, exon_ends = ucsc_result["exon_frames"], ucsc_result["exon_starts"], ucsc_result["exon_ends"]
    exon_positions = list(zip(exon_frames, exon_starts, exon_ends))
    if strand < 0:
        exon_positions = reversed(exon_positions)
    
    exon_info = []
    curr_exon_idx = None
    for i, (f, s, e) in enumerate(exon_positions):
        if s <= start <= e or s <= end <= e:
            curr_exon_idx = i
        exon_info.append(ExonInfo(
            number=i + 1,
            length=e - s,
            coordinates=[s, e],  # open-closed intervals
            frame=f,
            sequence=None,
        ))
    exon_count = len(exon_info)

    # supplement exon sequences for neighboring exons (helpful for exon skipping evaluation)
    message = None
    if curr_exon_idx is None:
        print(f"Variant {hgvs} is intronic. Using the nearest exon instead.")
        mutalyzer_result = search_mutalyzer(hgvs, return_exons=True)
        nearest_exon = mutalyzer_result.get("nearest_exon")
        curr_exon_idx = (nearest_exon - 1) if nearest_exon is not None else 0
        message = f"Variant {hgvs} is intronic. Using the nearest exon instead."

    flanking_exons = []
    for i in range(curr_exon_idx - 1, curr_exon_idx + 2):
        if i < 0 or i >= len(exon_info):
            continue
        info = exon_info[i]
        s, e = info["coordinates"]
        # by taking the reverse complement for negative strands, we can use the same logic for exon joining
        info["sequence"] = fetch_dna_sequence(chrom, s, e, rev_comp=strand < 0)
        flanking_exons.append(info)

    return TranscriptContext(
        hgvs=hgvs,
        refseq_id=hgvs.split(":")[0],
        chromosome=chrom,
        transcript_exons=exon_count,
        exon_number=curr_exon_idx + 1,  # convert back to 1-indexed
        flanking_exons=flanking_exons,
        message=message,
    )


# TODO: deprecate. unnecessary since we can just use the pubmed tool directly
def fetch_pubmed_context(clinvar_result: ClinVarResult, abstracts_only: bool = False) -> List[PubMedPaper]:
    pubmed_ids = []
    supporting_submissions = clinvar_result.get("supporting_submissions") or {}
    comments = supporting_submissions.get("comments", []) if isinstance(supporting_submissions, dict) else []
    for entry in comments:
        classification = entry.get("classification") or {}
        for citation in (classification.get("citations") or []):
            if citation.get("source") and citation["source"] == "PubMed":
                pubmed_ids.append(citation["id"])
    if not pubmed_ids:
        return []
    if abstracts_only:
        papers = fetch_pubmed_abstracts(pubmed_ids)
    else:
        papers = fetch_pubmed(pubmed_ids)
    
    return papers



# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def _coding_to_protein_coord(cds_position: int, cds_start: int, cds_end: int) -> int:
    """Convert exon coordinates to protein coordinates"""
    if cds_position < cds_start or cds_position > cds_end:
        return None
    return cds_position // 3 + 1


def _filter_ipr_domains(
    interpro_results: list[any],
    aa_range: tuple[int, int],
    add_details: bool = False
) -> list[InterProFeature]:
    """Parse out InterPro domains overlapping the given protein coordinates
    Args:
        interpro_results: list of InterPro results
        aa_range: tuple of start and end protein coordinates
        full_details: whether to return full details of the InterPro domains (description, citations, etc.)
    Returns:
        dict containing the amino acid range and overlapping InterPro domains
    """
    domains = []
    for entry in interpro_results or []:
        if entry.get("type") != "domain":
            continue
        # get metadata
        name = entry.get("name")
        ipr_id = entry.get("interpro_id")
        go_terms = entry.get("go_terms") or []
        if isinstance(go_terms, list):
            go_terms = [gt["name"] for gt in go_terms if isinstance(gt, dict) and gt.get("name")]
        # find overlap
        for value in (entry.get("overlapping_regions") or []):
            if not isinstance(value, dict) or "start" not in value or "end" not in value:
                continue
            start, end = value["start"], value["end"]
            if (aa_range[0] <= start <= aa_range[1]) or (aa_range[0] <= end <= aa_range[1]):
                result = {"name": name, "interpro_id": ipr_id, "go_terms": go_terms}
                if add_details:
                    result = browse_interpro(ipr_id)
                result["overlap_region"] = [start, end]
                domains.append(result)
    
    return domains


if __name__ == "__main__":
    hgvs = "NM_000170.3:c.538C>T"
    print(f"Variant: {hgvs}")

    print(f"Fetching clinical context")
    var_context = fetch_clinical_context(hgvs)

    print(f"Scraping cited PubMed papers")
    pubmed_context = fetch_pubmed_context(var_context["clinvar"])

    print(f"Fetching molecular context for {hgvs}")
    molecular_context = search_ensembl_vep(hgvs)

    print(f"Fetching transcript details for {hgvs}")
    transcript_context = fetch_transcript_context(hgvs)

    print(f"Fetching protein features for {hgvs}")
    protein_context = fetch_protein_context(hgvs)

    print(f"Saving results to dumps/variant_overview.json")
    with open("dumps/variant_context.json", "w") as f:
        json.dump({
            "clinical_context": var_context,
            "molecular_context": molecular_context,
            "transcript_context": transcript_context,
            "protein_context": protein_context,
            "relevant_publications": pubmed_context,
        }, f, indent=2)