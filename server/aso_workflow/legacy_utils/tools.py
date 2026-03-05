from abc import ABC, abstractmethod
import json
from typing import TypedDict, Optional
import uuid
from pydantic import BaseModel

from aso_workflow.utils.apis import (
    browse_webpage, search_serper, search_serper_scholar, search_ncbi, search_uniprot, browse_uniprot
)


class ToolOutput(TypedDict):
    call_id: str
    tool_name: str
    params: dict[str, any]
    result: str

# Some tools will return multiple results: in this case, each result
# will get a child document id under the parent tool call id.
class Document(TypedDict):
    title: Optional[str]
    url: Optional[str]
    text: Optional[str]
    metadata: Optional[dict[str, any]]

class BaseTool(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_description(self) -> str:
        pass

    @abstractmethod
    async def _execute(self, **kwargs) -> str:
        """Fetch the raw output."""
        pass

    @abstractmethod
    def _parse_output(self, raw_output: str) -> list[Document]:
        """Parse the raw output into a list of documents."""
        pass
    
    def get_call_id(self) -> str:
        return str(uuid.uuid4())[:8]

    async def __call__(self, **kwargs) -> ToolOutput:
        """Execute the tool and return the postprocessed output."""
        call_id = self.get_call_id()
        raw_output = await self._execute(**kwargs)
        output = self._parse_output(raw_output)
        result = ""
        for i, doc in enumerate(output):
            doc_str = "\n".join([f"{k}: {v}" for k, v in doc.items()])
            result += f"\n<snippet id={call_id}-{i}>\n{doc_str}\n</snippet>"

        return ToolOutput(
            call_id=call_id,
            tool_name=self.name,
            params=kwargs,
            result=result,
        )

class MockTool(BaseTool):
    def __init__(self, name: str = "mock_tool"):
        super().__init__(name)
    
    def get_description(self) -> dict[str, any]:
        return {
            "name": "mock_tool",
            "description": "A mock tool for testing.",
            "parameters": {
                "foo": "(required) the name of a fruit",
            },
            "required_parameters": ["foo"],
        }
    
    async def _execute(self, **kwargs) -> str:
        return "This is a mock tool output."
    
    def _parse_output(self, raw_output: str) -> list[Document]:
        return [Document(title="Mock tool", url="www.example.com", text=raw_output)]


class WebSearchTool(BaseTool):
    def __init__(self, name: str = "web_search"):
        super().__init__(name)
    
    def get_description(self) -> dict[str, any]:
        return {
            "name": "web_search",
            "description": "Searches the web for a given query. Outputs short previews of the first few web search results.",
            "parameters": {
                "query": "(required) The query to search for",
            },
            "required_parameters": ["query"],
        }
    
    async def _execute(self, query: str = None, gl: str = "us", hl: str = "en") -> dict[str, any]:
        return search_serper(query, gl, hl)

    def _parse_output(self, raw_output: dict[str, any]) -> list[Document]:
        """Extract documents from Serper response"""
        organic_results = raw_output.get("organic", [])
        documents = []

        for result in organic_results:
            if isinstance(result, dict):
                doc = Document(
                    title=result.get("title", "").strip(),
                    url=result.get("link", "").strip(),
                    text=result.get("snippet", "").strip(),
                )
                if doc["title"] or doc["text"] or doc["url"]:
                    documents.append(doc)
        return documents

class ScholarSearchTool(BaseTool):
    def __init__(self, name: str = "scholar_search"):
        super().__init__(name)
    
    def get_description(self) -> dict[str, any]:
        return {
            "name": "scholar_search",
            "description": "Searches the Google Scholar for academic papers using a given query. Outputs previews of search results.",
            "parameters": {
                "query": "(required) The query to search for",
            },
            "required_parameters": ["query"],
        }
    
    async def _execute(self, query: str = None) -> dict[str, any]:
        return search_serper_scholar(query)

    def _parse_output(self, raw_output: dict[str, any]) -> list[Document]:
        """Extract documents from Serper response"""
        organic_results = raw_output.get("organic", [])
        documents = []

        for result in organic_results:
            if isinstance(result, dict):
                metadata = {k:v for k,v in result.items() if k in ["publicationInfo", "year", "citedBy"]}
                doc = Document(
                    title=result.get("title", "").strip(),
                    url=result.get("link", "").strip(),
                    text=result.get("snippet", "").strip(),
                    metadata=metadata,
                )
                if doc["title"] or doc["text"] or doc["url"]:
                    documents.append(doc)
        return documents


class BrowseWebpageTool(BaseTool):
    def __init__(self, name: str = "browse_webpage"):
        super().__init__(name)
    
    def get_description(self) -> dict[str, any]:
        return {
            "name": "browse_webpage",
            "description": "Crawls a given webpage url and returns its contents (if available).",
            "parameters": {
                "url": "(required) The url of the webpage to crawl",
            },
            "required_parameters": ["url"],
        }
    
    async def _execute(self, url: str = None) -> dict[str, any]:
        return await browse_webpage(url=url)
    
    def _parse_output(self, raw_output: dict[str, any]) -> list[Document]:
        url = raw_output["url"]
        text = raw_output["text"]
        return [Document(title=f"Webpage content for {url}", url=url, text=text)]


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




async def main():
    tool = BrowseWebpageTool()
    out = await tool(url="https://www.ncbi.nlm.nih.gov/clinvar/RCV000355472/")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())