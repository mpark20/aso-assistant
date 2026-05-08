# ASO Assistant (Server Workflow Overview)

This repository contains an assessment engine that helps evaluate whether a genetic variant may be a good candidate for antisense oligonucleotide (ASO) therapeutic strategies.  
The core logic lives in `server/aso_workflow/pipeline.py`.

The workflow is designed to combine structured biomedical data sources with LLM-based reasoning so that each case ends with a structured report, including rationale and confidence signals for multiple ASO strategy types.

## Overview

This app allows users to input a genetic variant in HGVS format (e.g. `NM_000350.3(ABCA4):c.2626C>T`),
and outputs a report noting it's eligibility for splice correction, exon skipping, transcript knockdown, and wildtype upregulation ASO therapeutic strategies. The user is guided through the N1C VARIANT protocol step by step, and provided with links to resources used throughout the process.

## I. LLM Agent Setup

At each step, the LLM is given a step-specific system prompt (`server/aso_workflow/prompts.py`). Data from external resources specific to that step (See Section II) is fetched and appended to the prompt, along with the normalized variant name.

In the LLM's response, it has the option to either (1) answer immediately or (2) dive deeper into something mentioned in the initial prompt (e.g. a PubMed title, MIM number, Google search result preview). In the case of (2), the LLM can send the PMID/MIM/URL along with a set of questions of interest to a specialized literature review agent, which fetches the full paper/webpage text and attempts to answer the questions (if possible). The summarized results are returned to the main model, and the process can be repeated up to 6 times. Full implementation is in `llm.py`.

NOTE: This pipeline does NOT rely on inherent LLM knowledge about the variant of interest (we use genetic databases directly, as described in Section II and III). Rather, the LLM's main responsibility is to interpret papers, summarize information across sources, and apply logic from the guidelines.


## II. Protocol Pipeline

Full implementation is in `server/aso_workflow/pipeline.py`.
System prompts (slightly abbreviated) derived from the protocol are in `server/aso_workflow/prompts.py`. Prompts are supplemented with external data at each step in the pipeline.

### Background: Variant Check
Sources used: Mutalyzer
- Uses Mutalyzer to normalizes variant syntax and extract core metadata (gene, exon number, synonyms, etc.)
- Detects cases that are out-of-scope of the guidelines
- Detects incorrectly-formatted variant names

### Background: ASO Literature Check
Sources used: PubMed search, ClinVar
- Looks for existing ASO-related studies at variant, exon, and gene level using PubMed search and citations in ClinVar RCV comments. Synonyms of the variant listed in ClinVar/Mutalyzer are added to the PubMed query to expand coverage.
- Fetches full papers (or abstracts if unavailable) for titles that are deemed relevant for deeper analysis of findings
- Determines whether prior evidence may already support a specific strategy

### Step 1: Inheritance Pattern
Sources used: ClinVar, gnomAD, Google Search
- Assesses the inheritance pattern using public reports
- Uses population frequency data to supplement findings

### Step 2: Pathomechanism + Haploinsufficiency
Sources used: ClinVar, ClinGen Dosage KB, gnomAD, PubMed
- Classifies disease mechanism as loss-of-function, gain-of-function, dominant-negative, complex (i.e. mixed LoF/GoF), or unknown based on evidence retrieved from the sources above
- Assesses haploinsufficiency evidence, primarily from ClinGen, gnomAD, and other reports

### Step 3: Splicing Effects
Sources used: Ensembl VEP, ClinVar, PubMed, UCSC Genome Browser (wgEncodeGencodeBasicV48 and sequence tracks)
- Uses variant reports if there is evidence showing that the variant causes aberrant splicing
- Distinguishes stronger evidence (RNAseq, qPCR, or cDNA from patient-derived cells) from weaker evidence (in-silico, mini-gene, animal models)
- ClinVar and Ensembl results sometimes indicate whether a variant is previously associated with splicing effects. Ensembl VEP results will also return SpliceAI scores as supplemental information, if available.

### Step 4: Section Routing

Based on accumulated context, the pipeline decides which sections to evaluate:

- **Section A**: Canonical Exon Skipping
- **Section B**: Transcript Knockdown
- **Section C**: Wildtype Upregulation

Routing logic also handles special cases:

- CNV gain can route directly to knockdown-focused assessment.
- CNV loss can route directly to wildtype upregulation-focused assessment.
- Intronic cases can suppress exon-skipping evaluation.
- Existing successful ASO evidence can force a section to be included, even if unaligned with the protocol.

### Canonical Exon Skipping Assessment
Sources used: ClinVar, PubMed, UCSC Genome Browser, UniProt, InterPro
- Using UCSC Genome Browser, we fetch the sequence and frame of the exon the variant lies on, as well as the two exons adjacent to it. If the transcript is a negative strand, the reverse complement sequence is used. Metadata also includes the total number of exons in the transcript, providing info on whether the variant is in the first/last/only exon.
- During the assessment, the pipeline checks if a stop codon (TAA, TAG, TGA) is formed by joining the candidate exon's two neighboring exons (i.e if they are nonzero frame but in-phase). Since we take the reverse complement in the previous step, this should be strand-insensitive. The choice of nucleotides joined is based on the exon frames.
- To contextualize the importance of protein domains near the variant, we fetch the transcript product's UniProt ID and fetch overlapping domains from Interpro. Hits are filtered to those whose protein coordinates overlap with the variant (via conversion of coding coordinates to protein coordinates). A list of descriptions for overlapping domains as well as the total protein length is returned to the LLM as context to judge exon skipping potential.
- To supplement genomic data, PubMed results for relevant exon skipping therapies (if available) and ClinVar comments for the variant are also provided as context.

### Transcript Knockdown
Sources: ClinGen Dosage KB, gnomAD, prior steps

This step primarily relies on the variant's pathomechanism, haploinsufficiency, and prior ASO evidence. Since the external data needed for this is generated in prior steps, the assessment is made simply by providing such data along with the protocol instructions.


### Wildtype Upregulation
Sources: PubMed search, Ensembl VEP (with RiboseqORFs and UTRAnnotator plugins), Lim et al. (2020), Mittal et al. (2022), Felker et al. (2023)

At this stage, we check if alternative splicing events have previously been identified in the variant. Specifically: 
- Poison Exons (Lim et al., 2020; Felker et al., 2023; Mittal et al., 2022)
- Naturally-Occurring Antisense Transcripts (Mittal et al., 2022)
- Upstream Open Reading Frames (Mittal et al., 2022; Ribo-uORF) 

Unlike prior sections, a variant is only deemed eligible if an existing ASO already exists. If a variant has no existing therapy but has evidence of alternative splicing events (and meets the pathomechanism and inheritance criteria), it is simply flagged as "applicable". This is not the same as being marked "eligible". 

NOTE: In the future, we would like to integrate [VastDB](https://vastdb.crg.eu/wiki/Main_Page), a database of alternative splicing event. However, they currently don't have an API available so we recommend checking this resource manually (the LLM has been instructed to suggest this).

### Final Report Synthesis

The final report contains a summary of the entire workflow, including:
- Overview of the variant characteristics
- Summary of each step and associated external data used
- Per-strategy classifications
- Token-usage accounting (by model)


## III. External Resources Used

The workflow uses a mix of APIs, literature retrieval tools, and curated local reference files. Implementation is in `server/aso_workflow/utils`.

### Genetics Databases

- **Mutalyzer** (`https://mutalyzer.nl`)  
  Variant normalization, HGVS correction, exon numbers (cDNA), intron offsets.

- **Ensembl VEP REST API** (`https://rest.ensembl.org`)  
  Variant consequence and transcript-level annotations. We use the API endpoint with the following plugins: DosageSensitivity, mane, numbers, canonical, RiboseqORFs, SpliceAI, MaveDB, domains, LOEUF, UTRAnnotator

- **UCSC Genome Browser APIs** (`https://api.genome.ucsc.edu`)  
  Exon sequence, strands, and frames are taken from the `wgEncodeGencodeBasicV48` track endpoint, and sequences are from the `sequence` endpoint (the `revComp` flag is set for negative strands).

- **ClinVar** (`https://eutils.ncbi.nlm.nih.gov/entrez/eutils?db=clinvar`)  
  In addition to the variant summary, we also fetch each associated RCV to get submitter comments, dates, and relevant citations. For any citations associated, we fetch a preview from PubMed to get the title. 
  
  Implementation is in the `fetch_clinical_context` function in `server/aso_workflow/tasks.py`.

- **ClinGen Gene Dosage data** (`https://search.clinicalgenome.org/kb/gene-dosage/`)  
  Haploinsufficiency/triplosensitivity evidence is downloaded and cached locally, as there is no API currently available. 

- **OMIM API** (`https://api.omim.org`)  
  Fetches full OMIM entires. LLMs have the ability to call this on their own using MIM numbers, which usually appear in the variant's ClinVar entry.

- **gnomAD API** (`https://gnomad.broadinstitute.org/api`)  
  Population variant frequency.

- **Local curated datasets in `server/data`**  
Supplemental alternative splicing references for Section C of the N1C VARIANT protocol, including curated sources used for poison exon/uORF/NAT event checks.


### Literature and Web Search

- **PubMed/PMC**  
  Searches for relevant PMIDs given a search query. To zoom in on a specific paper, searching using a specific PMID will produce the full paper (if PMC open access) or abstract. PMC full texts are processed using the [BioC_json](https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json) endpoint.

- **Serper API** (`https://serper.dev`)  
  General and Google Scholar web search used for evidence discovery.

- **Crawl4AI** (`https://docs.crawl4ai.com/`)  
  Retrieves and cleans webpage content when structured endpoints (PubMed, OMIM) are not relevant. This is mainly for web search results, which may contain open access papers that are not in PubMed.

### Protein/Domain and Alternative Splicing Context

- **UniProt** (`https://rest.uniprot.org`)  
  Protein context used in exon skipping assessment. Specifically, we query by Refseq ID to identify the corresponding protein product.

- **InterPro** (`https://www.ebi.ac.uk/interpro/api`)
  Used for the identification of overlapping protein domain locations and their functions. We use InterPro because it conveniently links UniProt entries with a variety of other protein domain databases (e.g. PANTHER, Pfam).





