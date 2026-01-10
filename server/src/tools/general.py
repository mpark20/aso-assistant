from abc import ABC, abstractmethod
import json
from typing import TypedDict, Optional
import uuid
from pydantic import BaseModel
from src.tools._apis import (
    crawl_webpage, search_serper, search_serper_scholar, search_gpt,
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


class AdvancedWebSearchTool(BaseTool):
    def __init__(self, name: str = "advanced_web_search"):
        super().__init__(name)
    
    def get_description(self) -> dict[str, any]:
        return {
            "name": "advanced_web_search",
            "description": "Searches the web for a given query. Outputs snippets of web search results.",
            "parameters": {
                "query": "(required) The query to search for",
            },
            "required_parameters": ["query"],
        }
    
    async def _execute(self, query: str = None) -> dict[str, any]:
        return await search_gpt(query)

    def _parse_output(self, raw_output: dict[str, any]) -> list[Document]:
        text = raw_output["output"][-1]["content"][0]["text"]
        citations = []
        for item in raw_output["output"]:
            if item["type"] == "message":
                for content_part in item["content"]:
                    if "annotations" in content_part and content_part["annotations"]:
                        # each annotation has fields for start_index, end_index, title, url, and type
                        for ann in content_part["annotations"]:
                            if ann["type"] == "url_citation":
                                citations.append({
                                    "title": ann["title"],
                                    "url": ann["url"],
                                    "start_index": ann["start_index"],
                                    "end_index": ann["end_index"],
                                })
        sorted_supports = list(sorted(citations, key=lambda s: s["end_index"], reverse=True))
        for i in range(len(sorted_supports)):
            support = sorted_supports[i]
            start_index, end_index = support["start_index"], support["end_index"]
            j = i + 1
            while j < len(sorted_supports) and (start_index > sorted_supports[j]["end_index"]):
                j += 1
            citation_string = f"[{i+1}]"
            text = text[:end_index] + citation_string + text[end_index:]
        return text


class CrawlWebpageTool(BaseTool):
    def __init__(self, name: str = "crawl_webpage"):
        super().__init__(name)
    
    def get_description(self) -> dict[str, any]:
        return {
            "name": "crawl_webpage",
            "description": "Crawls a given webpage url and returns its contents (if available).",
            "parameters": {
                "url": "(required) The url of the webpage to crawl",
            },
            "required_parameters": ["url"],
        }
    
    async def _execute(self, url: str = None) -> dict[str, any]:
        return await crawl_webpage(url=url)
    
    def _parse_output(self, raw_output: dict[str, any]) -> list[Document]:
        url = raw_output["url"]
        text = raw_output["text"]
        return [Document(title=f"Webpage content for {url}", url=url, text=text)]



async def main():
    tool = CrawlWebpageTool()
    out = await tool(url="https://www.ncbi.nlm.nih.gov/clinvar/RCV000355472/")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())