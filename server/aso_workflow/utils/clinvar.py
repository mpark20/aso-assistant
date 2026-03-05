"""
Module for parsing ClinVar RCV XML files.
"""

import xml.etree.ElementTree as ET
import json
from pathlib import Path
from typing import TypedDict, Optional, Dict, List
import requests


class ClinVarResult(TypedDict):
    uid: str
    accession: str
    title: str
    protein_change: str
    chromosome: str
    assembly: str
    start: int
    stop: int
    molecular_consequences: list[str]
    pathogenicity: str
    review_status: str
    last_evaluated: str
    associated_traits: list[str]
    supporting_submissions: list[dict[str, any]]
    url: str

# special formatting functions for ClinVar results
def clinvar_formatter(data: dict[str, any]) -> dict[str, any]:
    """
    Format a raw ClinVar eSummary result into a flattened, human-readable dictionary.

    Extracts and normalizes key fields from the nested ClinVar API response,
    preferring GRCh38 coordinates when multiple assemblies are present.

    Args:
        data: A raw ClinVar eSummary result dict as returned by the NCBI eSummary API.

    Returns:
        A flattened dict with keys: uid, accession, title, protein_change,
        chromosome, assembly, start, stop, molecular_consequences, pathogenicity,
        review_status, last_evaluated, associated_traits, supporting_submissions,
        and url (pointing to the ClinVar variation page).
    """
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

    # supporting submissions
    supporting_submissions = data.get("supporting_submissions", [])

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
        "supporting_submissions": supporting_submissions,
        "url": url,
    }
    return formatted_result



class ClinVarClassification(TypedDict, total=False):
    date_last_evaluated: Optional[str]
    review_status: Optional[str]
    germline_classification: Optional[str]
    comment: Optional[str]
    citations: Optional[List[str]]


class ClinVarAssertion(TypedDict, total=False):
    id: Optional[str]
    submission_name: Optional[str]
    accession: Optional[Dict[str, str]]
    record_status: Optional[str]
    classification: Optional[ClinVarClassification]
    assertion_type: Optional[str]
    attributes: List[dict[str, str]]


def fetch_clinvar_rcv(rcv: str, add_citations: bool = False) -> List[ClinVarAssertion]:
    """
    Parse full ClinVar XML file.
    """
    endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "clinvar",
        "id": rcv,
        "rettype": "clinvarset",
    }
    response = requests.get(endpoint, params=params)
    if not response.ok:
        return None
    xml_data = response.text
    root = ET.fromstring(xml_data)

    results = []
    for cvset in root.findall(".//ClinVarSet"):
        for assertion in cvset.findall("ClinVarAssertion"):
            parsed = parse_clinvar_assertion(assertion, add_citations=add_citations)
            if parsed["record_status"] != "current":
                continue
            results.append(parsed)

    return results


def get_text(elem, default=None):
    """Safely get stripped text from an element."""
    if elem is not None and elem.text:
        return elem.text.strip()
    return default


def parse_attributesets(parent):
    """
    Parse <AttributeSet> blocks into a list of dicts.
    """
    attrs = []

    for aset in parent.findall("AttributeSet"):
        for attr in aset.findall("Attribute"):
            data = {
                "type": attr.get("Type"),
                "value": get_text(attr),
            }

            # Add any XML attributes
            for k, v in attr.attrib.items():
                if k != "Type":
                    data[k] = v

            attrs.append(data)

    return attrs


def parse_classification(class_elem, add_citations: bool = False):
    """
    Parse <Classification> block.
    """
    if class_elem is None:
        return None

    classification = ClinVarClassification(
        date_last_evaluated=class_elem.get("DateLastEvaluated"),
        review_status=get_text(class_elem.find("ReviewStatus")),
        germline_classification=get_text(
            class_elem.find("GermlineClassification")
        ),
        comment=get_text(class_elem.find("Comment")),
    )

    if add_citations:
        classification["citations"] = []
        # Citations (PubMed etc.)
        for cit in class_elem.findall("Citation"):
            id_elem = cit.find("ID")
            url_elem = cit.find("URL")

            if id_elem is not None:
                classification["citations"].append({
                    "source": id_elem.get("Source"),
                    "id": get_text(id_elem),
                })

            elif url_elem is not None:
                classification["citations"].append({
                    "url": get_text(url_elem)
                })

    return classification


def parse_observed_in(parent):
    """
    Parse <ObservedIn> blocks.
    """
    observed_list = []

    for obs in parent.findall("ObservedIn"):

        sample = obs.find("Sample")

        sample_data = {}
        if sample is not None:
            sample_data = {
                "origin": get_text(sample.find("Origin")),
                "species": get_text(sample.find("Species")),
                "affected_status": get_text(sample.find("AffectedStatus")),
                "number_tested": get_text(sample.find("NumberTested")),
                "sex": get_text(sample.find("Sex")),
            }

        method = obs.find("Method")
        method_type = None
        if method is not None:
            method_type = get_text(method.find("MethodType"))

        observed_data = []

        for od in obs.findall("ObservedData"):
            for attr in od.findall("Attribute"):
                observed_data.append({
                    "type": attr.get("Type"),
                    "value": get_text(attr),
                    "integerValue": attr.get("integerValue"),
                })

        observed_list.append({
            "sample": sample_data,
            "method": method_type,
            "observed_data": observed_data,
        })

    return observed_list


def parse_measures(parent):
    """
    Parse <MeasureSet> blocks.
    """
    measures = []

    for mset in parent.findall("MeasureSet"):

        mset_data = {
            "type": mset.get("Type"),
            "measures": [],
            "attributes": parse_attributesets(mset),
        }

        for measure in mset.findall("Measure"):

            measure_data = {
                "type": measure.get("Type"),
                "attributes": parse_attributesets(measure),
                "sequence_locations": [],
                "genes": [],
            }

            # SequenceLocation
            for loc in measure.findall("SequenceLocation"):
                measure_data["sequence_locations"].append(loc.attrib)

            # Gene symbols
            for rel in measure.findall("MeasureRelationship"):
                symbol = rel.find(".//Symbol/ElementValue")
                if symbol is not None:
                    measure_data["genes"].append(get_text(symbol))

            mset_data["measures"].append(measure_data)

        measures.append(mset_data)

    return measures


def parse_traits(parent):
    """
    Parse <TraitSet> blocks.
    """
    traits = []

    for tset in parent.findall("TraitSet"):
        for trait in tset.findall("Trait"):

            trait_data = {
                "type": trait.get("Type"),
                "names": [],
                "xrefs": [],
            }

            # Names
            for name in trait.findall("Name"):
                ev = name.find("ElementValue")
                if ev is not None:
                    trait_data["names"].append({
                        "type": ev.get("Type"),
                        "value": get_text(ev)
                    })

            # XRefs
            for xr in trait.findall("XRef"):
                trait_data["xrefs"].append(xr.attrib)

            traits.append(trait_data)

    return traits


def parse_clinvar_assertion(assertion, add_citations: bool = False):
    """
    Parse one <ClinVarAssertion>.
    """
    record_status = get_text(assertion.find("RecordStatus"))

    classification = parse_classification(assertion.find("Classification"), add_citations=add_citations)

    asrt = assertion.find("Assertion")
    assertion_type = asrt.get("Type") if asrt is not None else None

    # # Submission info
    # sub = assertion.find("ClinVarSubmissionID")
    # if sub is not None:
    #     data["submission"] = sub.attrib
    # acc = assertion.find("ClinVarAccession")
    # accession = acc.attrib if acc is not None else None
    # attr_set = parse_attributesets(assertion)
    # obs_list = parse_observed_in(assertion)
    # measures = parse_measures(assertion)
    # traits = parse_traits(assertion)

    return ClinVarAssertion(
        id=assertion.get("ID"),
        submission_name=assertion.get("SubmissionName"),
        record_status=record_status,
        classification=classification,
        assertion_type=assertion_type,
    )


if __name__ == "__main__":
    xml_file = "dumps/clinvar.xml"   # path to your XML file
    output_file = "dumps/clinvar_assertions.json"

    data = fetch_clinvar_rcv("RCV001068714", add_citations=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Parsed {len(data)} ClinVarAssertion records")
    print(f"Saved to {output_file}")
