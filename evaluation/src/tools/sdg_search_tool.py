import os
import json
import time
import asyncio
import subprocess
import threading
import re
from subprocess import TimeoutExpired
from nbconvert import ScriptExporter
from typing import Optional, Dict, List, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .base_tool import BaseTool



class BashFindTool(BaseTool):
    """Bash find tool using standard find command for file and directory search."""

    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    def __init__(
        self,
        working_dir: Optional[str] = None,
        default_limit: int = 20,
        default_offset: int = 0,
        request_timeout: int = 30
    ):
        self._default_limit = default_limit
        self._default_offset = default_offset
        self._request_timeout = request_timeout

        # Set working directory
        if working_dir:
            self._cwd = Path(working_dir).resolve()
            if not self._cwd.exists() or not self._cwd.is_dir():
                raise ValueError(f"Invalid working directory: {working_dir}")
        else:
            self._cwd = Path.cwd().resolve()


    @property
    def name(self) -> str:
        return "bash_find"

    @property
    def trigger_tag(self) -> str:
        return "find"

    def _parse_xml_tags(self, query: str) -> Dict[str, str]:
        """Parse XML-like tags from query string."""
        tags = {}
        expected_tags = ["pattern", "search_path", "file_type"]

        for tag in expected_tags:
            pattern = f'<{tag}>(.*?)</{tag}>'
            match = re.search(pattern, query, re.DOTALL)
            if match:
                tags[tag] = match.group(1).strip()

        return tags

    def _paginate_results(self, results: List[str], limit: int, offset: int) -> Dict:
        """Apply pagination to results."""
        total = len(results)

        if limit is None or limit == 0:
            paginated = results[offset:]
            has_more = False
        else:
            end = offset + limit
            paginated = results[offset:end]
            has_more = end < total

        return {
            "results": paginated,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "returned": len(paginated)
        }

    def _make_request(self, query: str, timeout: int) -> Dict:
        """Execute find command."""
        try:
            tags = self._parse_xml_tags(query.strip())

            # Extract parameters with defaults
            pattern = tags.get("pattern", "*")
            search_path = tags.get("search_path", ".")
            file_type = tags.get("file_type", "")

            # Fallback: if no tags found, treat entire query as pattern
            if not tags and query.strip():
                pattern = query.strip()

        except Exception as e:
            return {"error": f"Invalid query format: {str(e)}"}

        try:
            # Build command using ripgrep for files (respects .gitignore)
            if file_type == "d":
                # ripgrep doesn't list directories, fall back to find for dirs
                cmd = ["find", search_path, "-type", "d"]
                if pattern and pattern != "*":
                    cmd.extend(["-name", pattern])
            else:
                # Use ripgrep for files (respects .gitignore)
                cmd = ["rg", "--files"]

                if pattern and pattern != "*":
                    cmd.extend(["--glob", pattern])

                if search_path != ".":
                    cmd.append(search_path)

            # Execute command
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self._request_timeout, cwd=self._cwd
            )

            if result.returncode != 0:
                return {"error": f"Command failed: {result.stderr}"}

            lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
            # Remove the search path prefix if it's just "."
            if search_path == ".":
                lines = [line[2:] if line.startswith("./") else line for line in lines]

            return self._paginate_results(lines, self._default_limit, self._default_offset)

        except TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as e:
            return {"error": f"Find operation failed: {str(e)}"}

    async def postprocess_search_result(self, query: str, response: Dict, **kwargs) -> str:
        """Format the find results."""
        if "error" in response:
            return f"Find failed: {response['error']}"

        if not response.get("results"):
            return f"No files found for: {query}"

        return json.dumps(response, ensure_ascii=False, indent=2)

    async def execute(self, query: str, timeout: int = 60, **kwargs) -> str:
        """Execute a bash find query."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self._executor, lambda: self._make_request(query, timeout)
        )
        return await self.postprocess_search_result(query, response, **kwargs)



class BashGrepTool(BaseTool):
    """Bash grep tool using standard grep command for text search."""

    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    def __init__(
        self,
        working_dir: Optional[str] = None,
        default_limit: int = 20,
        default_offset: int = 0,
        request_timeout: int = 30
    ):
        self._default_limit = default_limit
        self._default_offset = default_offset
        self._request_timeout = request_timeout

        # Set working directory
        if working_dir:
            self._cwd = Path(working_dir).resolve()
            if not self._cwd.exists() or not self._cwd.is_dir():
                raise ValueError(f"Invalid working directory: {working_dir}")
        else:
            self._cwd = Path.cwd().resolve()


    @property
    def name(self) -> str:
        return "bash_grep"

    @property
    def trigger_tag(self) -> str:
        return "grep"

    def _parse_xml_tags(self, query: str) -> Dict[str, str]:
        """Parse XML-like tags from query string."""
        tags = {}
        expected_tags = ["pattern", "search_path", "include_pattern"]

        for tag in expected_tags:
            pattern = f'<{tag}>(.*?)</{tag}>'
            match = re.search(pattern, query, re.DOTALL)
            if match:
                tags[tag] = match.group(1).strip()

        return tags

    def _paginate_results(self, results: List[str], limit: int, offset: int) -> Dict:
        """Paginate results."""
        total = len(results)

        if limit is None or limit == 0:
            paginated = results[offset:]
            has_more = False
        else:
            end = offset + limit
            paginated = results[offset:end]
            has_more = end < total

        return {
            "results": paginated,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "returned": len(paginated)
        }

    def _make_request(self, query: str, timeout: int) -> Dict:
        """Execute grep command."""
        try:
            tags = self._parse_xml_tags(query.strip())

            # Extract parameters with defaults
            pattern = tags.get("pattern", "")
            search_path = tags.get("search_path", ".")
            include_pattern = tags.get("include_pattern", "")

            # Fallback: if no tags found, treat entire query as pattern
            if not tags and query.strip():
                pattern = query.strip()

            if not pattern:
                return {"error": "No search pattern provided"}

        except Exception as e:
            return {"error": f"Invalid query format: {str(e)}"}

        try:
            # Use ripgrep (respects .gitignore)
            cmd = ["rg", pattern]

            if include_pattern:
                cmd.extend(["--glob", include_pattern])

            if search_path != ".":
                cmd.append(search_path)

            # Execute
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self._request_timeout, cwd=self._cwd
            )

            if result.returncode != 0:
                if result.returncode == 1:  # No matches found
                    return {"results": [], "total": 0, "offset": 0, "limit": self._default_limit, "has_more": False, "returned": 0}
                else:
                    return {"error": f"Grep command failed: {result.stderr}"}

            lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
            return self._paginate_results(lines, self._default_limit, self._default_offset)

        except TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as e:
            return {"error": f"Grep search failed: {str(e)}"}

    async def postprocess_search_result(self, query: str, response: Dict, **kwargs) -> str:
        """Format the grep results."""
        if "error" in response:
            return f"Grep failed: {response['error']}"

        if not response.get("results"):
            return f"No matches found for: {query}"

        return json.dumps(response, ensure_ascii=False, indent=2)

    async def execute(self, query: str, timeout: int = 60, **kwargs) -> str:
        """Execute a bash grep query."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self._executor, lambda: self._make_request(query, timeout)
        )
        return await self.postprocess_search_result(query, response, **kwargs)



class BashReadTool(BaseTool):
    """Bash read tool for file content reading with caching based on file modification time."""

    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    def __init__(
        self,
        working_dir: Optional[str] = None,
        default_limit: int = 0,
        default_offset: int = 0
    ):
        self._default_limit = default_limit
        self._default_offset = default_offset

        # Set working directory
        if working_dir:
            self._cwd = Path(working_dir).resolve()
            if not self._cwd.exists() or not self._cwd.is_dir():
                raise ValueError(f"Invalid working directory: {working_dir}")
        else:
            self._cwd = Path.cwd().resolve()


    @property
    def name(self) -> str:
        return "bash_read"

    @property
    def trigger_tag(self) -> str:
        return "read"

    def _parse_xml_tags(self, query: str) -> Dict[str, str]:
        """Parse XML-like tags from query string."""
        tags = {}
        expected_tags = ["filepath"]

        for tag in expected_tags:
            pattern = f'<{tag}>(.*?)</{tag}>'
            match = re.search(pattern, query, re.DOTALL)
            if match:
                tags[tag] = match.group(1).strip()

        return tags

    def _paginate_results(self, results: List[str], limit: int, offset: int) -> Dict:
        """Paginate results."""
        total = len(results)

        if limit is None or limit == 0:
            paginated = results[offset:]
            has_more = False
        else:
            end = offset + limit
            paginated = results[offset:end]
            has_more = end < total

        return {
            "results": "\n".join(paginated),
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "returned": len(paginated)
        }



    def _make_request(self, query: str, timeout: int) -> Dict:
        """Read file content."""
        try:
            tags = self._parse_xml_tags(query.strip())

            # Extract filepath
            filepath = tags.get("filepath", "")

            # Fallback: if no tags found, treat entire query as filepath
            if not tags and query.strip():
                filepath = query.strip()

            if not filepath:
                return {"error": "No filepath provided"}

        except Exception as e:
            return {"error": f"Invalid query format: {str(e)}"}

        try:
            # Validate filepath
            full_path = self._cwd / filepath

            # Read file
            if filepath.endswith('.ipynb'):
                exporter = ScriptExporter()
                script, _ = exporter.from_filename(str(full_path))
                lines = script.strip().split('\n')
            else:
                with open(full_path, "r", encoding="utf-8") as f:
                    lines = f.read().strip().split("\n")

            # Apply pagination
            response = self._paginate_results(lines, self._default_limit, self._default_offset)

            # Add modification time for cache validation
            response["mod_time"] = os.path.getmtime(str(full_path))
            return response

        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}

    async def postprocess_search_result(self, query: str, response: Dict, **kwargs) -> str:
        """Format the read results."""
        if "error" in response:
            return f"Read failed: {response['error']}"

        # Remove mod_time from response before returning
        result = {k: v for k, v in response.items() if k != "mod_time"}
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def execute(self, query: str, timeout: int = 60, **kwargs) -> str:
        """Execute a bash read query."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self._executor, lambda: self._make_request(query, timeout)
        )
        return await self.postprocess_search_result(query, response, **kwargs)

