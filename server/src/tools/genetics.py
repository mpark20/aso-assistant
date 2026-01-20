from src.tools.general import BaseTool, Document
from src.tools._apis import browse_uniprot, search_ncbi, search_uniprot

class NCBISearchTool(BaseTool):
    def __init__(self, name: str = "ncbi_search"):
        super().__init__(name)
    
    def get_description(self) -> dict[str, any]:
        return {
            "name": "ncbi_search",
            "description": "Searches the NCBI database for a given query. Outputs snippets of the top NCBI search results.",
            "parameters": {
                "database": "(required) The NCBI database to search (e.g. clinvar, medgen, etc.)",
                "query": "(required) The query to search for (e.g. NM_000392.5 AND c.3399_3400delTT)",
            },
            "required_parameters": ["database", "query"],
        }
    
    async def _execute(self, database: str = None, query: str = None) -> dict[str, any]:
        return search_ncbi(database=database, search_term=query)
    
    def _parse_output(self, raw_output: dict[str, any]) -> list[Document]:
        results = raw_output["results"]
        documents = []
        for result in results:
            doc = Document(
                title=result.get("title"),
                url=result.get("url"),
                text=result
            )
            if doc["title"] or doc["text"] or doc["url"]:
                documents.append(doc)
        return documents


class UniProtSearchTool(BaseTool):
    def __init__(self, name: str = "uniprot_search"):
        super().__init__(name)
    
    def get_description(self) -> dict[str, any]:
        return {
            "name": "uniprot_search",
            "description": "Searches the UniProt database for a given query. Outputs snippets of the top UniProt search results.",
            "parameters": {
                "protein_name": "(required) The protein name to search for (e.g. BRCA1_HUMAN). Can be a protein or gene symbol.",
            },
            "required_parameters": ["protein_name"],
        }
    
    async def _execute(self, protein_name: str = None) -> dict[str, any]:
        return search_uniprot(protein_name=protein_name)
    
    def _parse_output(self, raw_output: list[any]) -> list[Document]:
        documents = []
        for result in raw_output:
            doc = Document(
                title=result.get("uniprot_name") + f" ({result.get('uniprot_id')})",
                url=result.get("url"),
                text=result
            )
            if doc["title"] or doc["text"] or doc["url"]:
                documents.append(doc)
        return documents

class BrowseUniProtTool(BaseTool):
    def __init__(self, name: str = "browse_uniprot"):
        super().__init__(name)
    
    def get_description(self) -> dict[str, any]:
        return {
            "name": "browse_uniprot",
            "description": "Browse a UniProt protein's overlapping InterPro features in detail.",
            "parameters": {
                "uniprot_id": "(required) The UniProt ID of the protein to browse (e.g. P01308)",
            },
            "required_parameters": ["uniprot_id"],
        }
    
    async def _execute(self, uniprot_id: str = None) -> dict[str, any]:
        return browse_uniprot(uniprot_id=uniprot_id)
    
    def _parse_output(self, raw_output: dict[str, any]) -> list[Document]:
        results = raw_output["results"]
        documents = []
        for result in results:
            doc = Document(
                title=result.get("title"),
                url=result.get("url"),
                text=result
            )
            if doc["title"] or doc["text"] or doc["url"]:
                documents.append(doc)
        return documents