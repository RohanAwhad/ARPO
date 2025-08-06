import sys
import os

sys.path.append(os.getcwd())
import time
import asyncio
import requests
import aiolimiter
from bs4 import BeautifulSoup
import re
from typing import Union, Dict, List
from concurrent.futures import ThreadPoolExecutor

from .base_tool import BaseTool
from .cache_manager import PreprocessCacheManager


class DuckDuckGoSearchTool(BaseTool):
    """DuckDuckGoSearchTool"""

    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    def __init__(
        self,
        max_results: int = 10,
        result_length: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        requests_per_second: float = 2.0,
        search_cache_file="",
    ):
        self._max_results = max_results
        self._result_length = result_length
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._limiter = aiolimiter.AsyncLimiter(
            max_rate=requests_per_second, time_period=1.0
        )
        self.search_cache_manager = PreprocessCacheManager(search_cache_file)

    @property
    def name(self) -> str:
        return "duckduckgo_search"

    @property
    def trigger_tag(self) -> str:
        return "search"

    def _call_request(self, query, timeout):
        base_url = "https://lite.duckduckgo.com/lite/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }

        error_cnt = 0
        while error_cnt < self._max_retries:
            try:
                params = {'q': query}
                response = requests.get(base_url, params=params, headers=headers, timeout=timeout)
                
                if response.status_code == 202 or 'anomaly.js' in response.text:
                    print(f"DuckDuckGo bot detection triggered for query: {query}")
                    return []
                    
                response.raise_for_status()
                return self._parse_response(response.text)
                
            except requests.exceptions.Timeout:
                error_cnt += 1
                print(f"error_cnt: {error_cnt}, DuckDuckGo search request timed out ({timeout} seconds) for query: {query}")
                time.sleep(self._retry_delay)
            except requests.exceptions.RequestException as e:
                error_cnt += 1
                print(f"error_cnt: {error_cnt}, Error occurred during DuckDuckGo search request: {e}")
                time.sleep(self._retry_delay)
        
        print(f"query: {query} has tried {error_cnt} times without success, just skip it.")
        return []

    def _parse_response(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        
        tables = soup.find_all('table')
        if not tables:
            return []
        
        results_table = tables[-1]
        rows = results_table.find_all('tr')
        
        i = 0
        while i < len(rows) and len(results) < self._max_results:
            current_row = rows[i]
            cells = current_row.find_all('td')
            
            if len(cells) >= 2:
                first_cell_text = cells[0].get_text(strip=True)
                
                if re.match(r'^\d+\.\s*$', first_cell_text):
                    second_cell = cells[1]
                    link = second_cell.find('a', href=True)
                    
                    if link:
                        title = link.get_text(strip=True)
                        url = link.get('href')
                        
                        description = ""
                        if i + 1 < len(rows):
                            desc_row = rows[i + 1]
                            desc_cells = desc_row.find_all('td')
                            if len(desc_cells) >= 2:
                                desc_text = desc_cells[1].get_text(strip=True)
                                if not re.match(r'^https?://', desc_text):
                                    description = desc_text
                        
                        description = re.sub(r'\s+', ' ', description).strip()
                        
                        if title and url:
                            results.append({
                                'title': title,
                                'url': url,
                                'description': description
                            })
                        
                        i += 3
                        continue
            i += 1
        
        return results

    def _make_request(self, query: str, timeout: int):
        """Send a request to DuckDuckGo Lite."""
        return self._call_request(query, timeout)

    async def postprocess_search_result(self, query, response, **kwargs) -> str:
        if not response:
            return f"DuckDuckGo search failed: {query}"
        
        result = self._format_results(response)
        return result if result else f"No results found for: {query}"

    async def execute(self, query: str, timeout: int = 60, **kwargs) -> str:
        """Execute a DuckDuckGo search query with support for cache."""
        hit_cache = self.search_cache_manager.hit_cache(query)
        if hit_cache:
            print("hit cache: ", query)
            response = hit_cache
        else:
            loop = asyncio.get_event_loop()

            async with self._limiter:
                response = await loop.run_in_executor(
                    self._executor, lambda: self._make_request(query, timeout)
                )
            
            if not response:
                return f"DuckDuckGo search failed: {query}"
            await self.search_cache_manager.add_to_cache(query, response)
        
        return await self.postprocess_search_result(query, response, **kwargs)

    def _format_results(self, results: List[Dict[str, str]]) -> Union[str, None]:
        """Format the search results."""
        if not results:
            return None

        formatted = []
        for idx, result in enumerate(results, 1):
            title = result.get('title', '')
            description = result.get('description', '')
            url = result.get('url', '')
            
            content = f"{title}. {description}".strip()
            content = content[:self._result_length]
            
            formatted.append(f"Page {idx}: {content}")
        
        return "\n".join(formatted)






import sys
import os

sys.path.append(os.getcwd())
import time
import langid
import asyncio
import requests
import aiolimiter
from typing import Union, Dict
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor

from .base_tool import BaseTool
from .cache_manager import PreprocessCacheManager


class BingSearchTool(BaseTool):
    """BingSearchTool"""

    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    def __init__(
        self,
        api_key: str,
        zone: str = "your_zone",
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "cn",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        requests_per_second: float = 2.0,
        search_cache_file="",
    ):
        self._api_key = api_key
        self._zone = zone
        self._max_results = max_results
        self._result_length = result_length
        self._location = location
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._limiter = aiolimiter.AsyncLimiter(
            max_rate=requests_per_second, time_period=1.0
        )
        self.search_cache_manager = PreprocessCacheManager(search_cache_file)

    @property
    def name(self) -> str:
        return "bing_search"

    @property
    def trigger_tag(self) -> str:
        return "search"

    def _call_request(self, query, headers, payload, timeout):
        error_cnt = 0
        while True:
            if error_cnt >= self._max_retries:
                print(
                    f"query: {query} has tried {error_cnt} times without success, just skip it."
                )
                break
            try:
                response = requests.post(
                    "https://api.brightdata.com/request",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                search_results = response.json()
                return search_results
            except requests.exceptions.Timeout:
                error_cnt += 1
                print(
                    f"error_cnt: {error_cnt}, Bing Web Search request timed out ({timeout} seconds) for query: {query}"
                )
                time.sleep(5)
            except requests.exceptions.RequestException as e:
                error_cnt += 1
                print(
                    f"error_cnt: {error_cnt}, Error occurred during Bing Web Search request: {e}, payload: {payload}"
                )
                time.sleep(5)
        return None

    def _pack_query(self, query):
        if langid.classify(query)[0] == "zh":
            mkt, setLang = "zh-CN", "zh"
        else:
            mkt, setLang = "en-US", "en"
        input_obj = {"q": query, "mkt": mkt, "setLang": setLang}
        encoded_query = urlencode(input_obj)
        return encoded_query

    def _make_request(self, query: str, timeout: int):
        """
        Send a request to the Brightdata API.

        Args:
            query: The search query.
            timeout: Request timeout in seconds.
            
        Returns:
            The response object.
        """
        encoded_query = self._pack_query(query)
        target_url = f"https://www.bing.com/search?{encoded_query}&brd_json=1&cc={self._location}"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"zone": self._zone, "url": target_url, "format": "raw"}
        result = self._call_request(query, headers, payload, timeout)
        return result

    async def postprocess_search_result(self, query, response, **kwargs) -> str:
        data = response
        if "organic" not in data:
            data["chunk_content"] = []
            result = self._format_results(data)
        else:
            chunk_content_list = []
            seen_snippets = set()
            for result_item in data["organic"]:
                snippet = result_item.get("description", "").strip()
                if snippet and snippet not in seen_snippets:
                    chunk_content_list.append(snippet)
                    seen_snippets.add(snippet)
            data["chunk_content"] = chunk_content_list
            result = self._format_results(data)
        return result

    async def execute(self, query: str, timeout: int = 60, **kwargs) -> str:
        """
        Execute a Bing search query with support for cache and semantic similarity cache hits.

        Args:
            query: The search query text.
            timeout: Request timeout in seconds.
            model: SBERT model used for semantic search.
            threshold: Minimum similarity threshold.
            top_k: Number of top most similar cached entries to consider.

        Returns:
            A string containing the search result.
        """
        hit_cache = self.search_cache_manager.hit_cache(query)
        if hit_cache:
            print("hit cache: ", query)
            response = hit_cache
        else:
            loop = asyncio.get_event_loop()

            async with self._limiter:
                response = await loop.run_in_executor(
                    self._executor, lambda: self._make_request(query, timeout)
                )
            if response is None:
                return f"Bing search failed: {query}"
            await self.search_cache_manager.add_to_cache(query, response)
        assert response is not None
        return await self.postprocess_search_result(query, response, **kwargs)

    def _format_results(self, results: Dict) -> Union[str, None]:
        """Format the search results."""
        if not results.get("chunk_content"):
            return None

        formatted = []
        for idx, snippet in enumerate(results["chunk_content"][: self._max_results], 1):
            snippet = snippet[: self._result_length]
            formatted.append(f"Page {idx}: {snippet}")
        return "\n".join(formatted)
