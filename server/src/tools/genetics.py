from src.tools.general import BaseTool, Document
from src.tools._apis import search_ncbi

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

