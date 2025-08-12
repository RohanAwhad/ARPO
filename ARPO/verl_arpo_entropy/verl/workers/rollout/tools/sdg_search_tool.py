import os
import json
import fcntl
import pathlib
import re
import subprocess
from subprocess import TimeoutExpired
from nbconvert import ScriptExporter
from typing import Optional, Dict, List, Any
from pathlib import Path

from verl.workers.rollout.tools.base_tool import BaseTool



class BashFindTool(BaseTool):
  """
  Bash find tool using standard find command for file and directory search.
  """

  def __init__(
    self,
    working_dir: Optional[str] = None,
    default_limit: int = 20,
    default_offset: int = 0,
    request_timeout: int = 30
  ):
    """Initialize the bash find tool."""
    self._default_limit = default_limit
    self._default_offset = default_offset
    self._request_timeout = request_timeout
    
    # Set working directory
    if working_dir:
      self._cwd = Path(working_dir).resolve()
      if not self._cwd.exists():
        raise ValueError(f"Working directory does not exist: {working_dir}")
      if not self._cwd.is_dir():
        raise ValueError(f"Working directory is not a directory: {working_dir}")
    else:
      self._cwd = Path.cwd().resolve()



  def _parse_xml_tags(self, query: str) -> Dict[str, str]:
    """Parse XML-like tags from query string."""
    import re
    
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

  @property
  def name(self) -> str:
    """Tool name identifier."""
    return "bash_find"

  @property
  def trigger_tag(self) -> str:
    """Tag used to trigger this tool."""
    return "find"

  def execute(self, query: str, timeout: int = 60) -> str:
    """
    Execute bash find query using standard find command.
    
    Args:
      query: XML-like tags "<pattern>*.py</pattern><search_path>src</search_path><file_type>f</file_type>"
      timeout: Not used (kept for compatibility)
      
    Returns:
      Formatted find results as JSON string
    """
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
      return json.dumps({"error": f"Invalid query format: {str(e)}"})
    
    try:
      # Validate path
      validated_path = search_path
      
      # Build command using ripgrep for files (respects .gitignore)
      if file_type == "d":
        # ripgrep doesn't list directories, fall back to find for dirs
        cmd = ["find", validated_path, "-type", "d"]
        if pattern and pattern != "*":
          cmd.extend(["-name", pattern])
      else:
        # Use ripgrep for files (respects .gitignore)
        cmd = ["rg", "--files"]
        
        if pattern and pattern != "*":
          cmd.extend(["--glob", pattern])
        
        if validated_path != ".":
          cmd.append(validated_path)

      
      # Execute command
      result = subprocess.run(
        cmd, capture_output=True, text=True, 
        timeout=self._request_timeout, cwd=self._cwd
      )
      
      if result.returncode != 0:
        response = {"error": f"Command failed: {result.stderr}"}
      else:
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        # Remove the search path prefix if it's just "."
        if validated_path == ".":
          lines = [line[2:] if line.startswith("./") else line for line in lines]
        response = self._paginate_results(lines, self._default_limit, self._default_offset)
      
      return json.dumps(response, ensure_ascii=False, indent=2)
      
    except TimeoutExpired:
      return json.dumps({"error": "Command timed out"})
    except Exception as e:
      return json.dumps({"error": f"Find operation failed: {str(e)}"})




class BashGrepTool(BaseTool):
  """
  Bash grep tool using standard grep command for text search.
  """

  def __init__(
    self,
    working_dir: Optional[str] = None,
    default_limit: int = 20,
    default_offset: int = 0,
    request_timeout: int = 30
  ):
    """Initialize grep tool."""
    self._default_limit = default_limit
    self._default_offset = default_offset
    self._request_timeout = request_timeout
    
    # Set working directory
    if working_dir:
      self._cwd = Path(working_dir).resolve()
      if not self._cwd.exists():
        raise ValueError(f"Working directory does not exist: {working_dir}")
      if not self._cwd.is_dir():
        raise ValueError(f"Working directory is not a directory: {working_dir}")
    else:
      self._cwd = Path.cwd().resolve()



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

  @property
  def name(self) -> str:
    """Tool name."""
    return "bash_grep"

  @property
  def trigger_tag(self) -> str:
    """Trigger tag."""
    return "grep"

  def execute(self, query: str, timeout: int = 60) -> str:
    """
    Execute grep search using standard grep command.
    
    Args:
      query: XML-like tags "<pattern>TODO</pattern><search_path>src</search_path><include_pattern>*.py</include_pattern>"
      timeout: Not used
      
    Returns:
      JSON string with search results
    """
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
        return json.dumps({"error": "No search pattern provided"})
      
    except Exception as e:
      return json.dumps({"error": f"Invalid query format: {str(e)}"})
    
    try:
      # Validate path
      validated_path = search_path
      
      # Use ripgrep (respects .gitignore)
      cmd = ["rg", pattern]
      
      if include_pattern:
        cmd.extend(["--glob", include_pattern])
      
      if validated_path != ".":
        cmd.append(validated_path)

      
      # Execute
      result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=self._request_timeout, cwd=self._cwd
      )
      
      if result.returncode != 0:
        if result.returncode == 1:  # No matches found
          response = {"results": [], "total": 0, "offset": 0, "limit": self._default_limit, "has_more": False, "returned": 0}
        else:
          response = {"error": f"Grep command failed: {result.stderr}"}
      else:
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        response = self._paginate_results(lines, self._default_limit, self._default_offset)
      
      return json.dumps(response, ensure_ascii=False, indent=2)
      
    except TimeoutExpired:
      return json.dumps({"error": "Command timed out"})
    except Exception as e:
      return json.dumps({"error": f"Grep search failed: {str(e)}"})




class BashReadTool(BaseTool):
  """
  Bash read tool for file content reading.
  """

  def __init__(
    self,
    working_dir: Optional[str] = None,
    default_limit: int = 0,
    default_offset: int = 0
  ):
    """Initialize read tool."""
    self._default_limit = default_limit
    self._default_offset = default_offset
    
    # Set working directory
    if working_dir:
      self._cwd = Path(working_dir).resolve()
      if not self._cwd.exists():
        raise ValueError(f"Working directory does not exist: {working_dir}")
      if not self._cwd.is_dir():
        raise ValueError(f"Working directory is not a directory: {working_dir}")
    else:
      self._cwd = Path.cwd().resolve()



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

  @property
  def name(self) -> str:
    """Tool name."""
    return "bash_read"

  @property
  def trigger_tag(self) -> str:
    """Trigger tag."""
    return "read"

  def execute(self, query: str, timeout: int = 60) -> str:
    """
    Execute file read.
    
    Args:
      query: XML-like tags "<filepath>README.md</filepath>" or just filepath
      timeout: Not used
      
    Returns:
      JSON string with file content and pagination info
    """
    try:
      tags = self._parse_xml_tags(query.strip())
      
      # Extract filepath
      filepath = tags.get("filepath", "")
      
      # Fallback: if no tags found, treat entire query as filepath
      if not tags and query.strip():
        filepath = query.strip()
      
      if not filepath:
        return json.dumps({"error": "No filepath provided"})
      
    except Exception as e:
      return json.dumps({"error": f"Invalid query format: {str(e)}"})
    
    try:
      # Validate filepath
      validated_path = filepath
      full_path = self._cwd / validated_path
      
      # Read file
      if validated_path.endswith('.ipynb'):
        exporter = ScriptExporter()
        script, _ = exporter.from_filename(str(full_path))
        lines = script.strip().split('\n')
      else:
        with open(full_path, "r", encoding="utf-8") as f:
          lines = f.read().strip().split("\n")
      
      # Apply pagination
      response = self._paginate_results(lines, self._default_limit, self._default_offset)
      
      return json.dumps(response, ensure_ascii=False, indent=2)
      
    except Exception as e:
      return json.dumps({"error": f"Failed to read file: {str(e)}"})


