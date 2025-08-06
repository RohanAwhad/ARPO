import os
import json
import time
import queue
import atexit
import fcntl
import pathlib
import re
import subprocess
import threading
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
    cache_file: Optional[str] = None,
    async_cache_write: bool = True,
    cache_refresh_interval: float = 60.0,
    request_timeout: int = 30
  ):
    """Initialize the bash find tool."""
    self._default_limit = default_limit
    self._default_offset = default_offset
    self._request_timeout = request_timeout
    
    # Cache and synchronization
    self._cache = {}
    self._cache_timestamps = {}
    self._cache_lock = threading.Lock()
    self._async_cache_write = async_cache_write
    self._write_queue = queue.Queue() if async_cache_write else None
    self._cache_refresh_interval = cache_refresh_interval
    
    # Set working directory
    if working_dir:
      self._cwd = Path(working_dir).resolve()
      if not self._cwd.exists():
        raise ValueError(f"Working directory does not exist: {working_dir}")
      if not self._cwd.is_dir():
        raise ValueError(f"Working directory is not a directory: {working_dir}")
    else:
      self._cwd = Path.cwd().resolve()

    
    self._setup_cache_paths(cache_file)
    self._load_cache()
    
    if self._async_cache_write:
      self._init_async_writer()

  def _setup_cache_paths(self, cache_file: Optional[str]) -> None:
    """Setup cache file paths."""
    if cache_file is None:
      cache_dir = pathlib.Path.home() / ".verl_cache"
      cache_dir.mkdir(exist_ok=True)
      self._cache_file = cache_dir / "bash_find_cache.json"
    else:
      self._cache_file = pathlib.Path(cache_file)
      self._cache_file.parent.mkdir(parents=True, exist_ok=True)

  def _init_async_writer(self) -> None:
    """Initialize async cache writer."""
    self._stop_writer = threading.Event()
    self._writer_thread = threading.Thread(
      target=self._cache_writer_thread,
      daemon=True,
      name="BashFindCacheWriter"
    )
    self._writer_thread.start()
    atexit.register(self._cleanup)

  def _cleanup(self) -> None:
    """Cleanup on exit."""
    if self._async_cache_write and hasattr(self, '_stop_writer'):
      self._stop_writer.set()
      if self._writer_thread.is_alive():
        self._writer_thread.join(timeout=2.0)
      self._save_cache_sync()

  def _cache_writer_thread(self) -> None:
    """Background cache writer."""
    while not self._stop_writer.is_set():
      try:
        try:
          _ = self._write_queue.get(timeout=1.0)
          self._save_cache_sync()
          self._write_queue.task_done()
        except queue.Empty:
          continue
      except Exception as e:
        print(f"Cache writer error: {str(e)}")

  def _load_cache(self) -> None:
    """Load cache from disk."""
    if not self._cache_file.exists():
      return
    
    try:
      with open(self._cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        self._cache = data.get("cache", {})
        self._cache_timestamps = data.get("timestamps", {})
    except Exception as e:
      print(f"Failed to load find cache: {str(e)}")
      self._cache = {}
      self._cache_timestamps = {}

  def _save_cache_sync(self) -> None:
    """Save cache to disk."""
    try:
      with self._cache_lock:
        data = {
          "cache": dict(self._cache),
          "timestamps": dict(self._cache_timestamps)
        }
      
      temp_file = self._cache_file.with_suffix('.tmp')
      with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
      
      temp_file.replace(self._cache_file)
    except Exception as e:
      print(f"Failed to save find cache: {str(e)}")

  def _save_cache(self) -> None:
    """Save cache async or sync."""
    if self._async_cache_write:
      try:
        self._write_queue.put(True, block=False)
      except queue.Full:
        pass
    else:
      self._save_cache_sync()

  def _is_cache_valid(self, cache_key: str) -> bool:
    """Check if cache entry is still valid."""
    if cache_key not in self._cache_timestamps:
      return False
    
    timestamp = self._cache_timestamps[cache_key]
    return time.time() - timestamp < self._cache_refresh_interval

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
    
    # Create cache key
    cache_key = json.dumps({"pattern": pattern, "search_path": search_path, "file_type": file_type}, sort_keys=True)
    
    # Check cache
    with self._cache_lock:
      if cache_key in self._cache and self._is_cache_valid(cache_key):
        return self._cache[cache_key]
    
    try:
      # Validate path
      validated_path = search_path
      
      # Build find command
      cmd = ["find", validated_path]
      
      if file_type:
        cmd.extend(["-type", file_type])
      
      if pattern and pattern != "*":
        cmd.extend(["-name", pattern])
      
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
      
      # Cache result
      response_str = json.dumps(response, ensure_ascii=False, indent=2)
      with self._cache_lock:
        self._cache[cache_key] = response_str
        self._cache_timestamps[cache_key] = time.time()
      
      self._save_cache()
      return response_str
      
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
    cache_file: Optional[str] = None,
    async_cache_write: bool = True,
    cache_refresh_interval: float = 30.0,
    request_timeout: int = 30
  ):
    """Initialize grep tool."""
    self._default_limit = default_limit
    self._default_offset = default_offset
    self._request_timeout = request_timeout
    
    # Cache setup
    self._cache = {}
    self._cache_timestamps = {}
    self._cache_lock = threading.Lock()
    self._async_cache_write = async_cache_write
    self._write_queue = queue.Queue() if async_cache_write else None
    self._cache_refresh_interval = cache_refresh_interval
    
    # Set working directory
    if working_dir:
      self._cwd = Path(working_dir).resolve()
      if not self._cwd.exists():
        raise ValueError(f"Working directory does not exist: {working_dir}")
      if not self._cwd.is_dir():
        raise ValueError(f"Working directory is not a directory: {working_dir}")
    else:
      self._cwd = Path.cwd().resolve()

    
    self._setup_cache_paths(cache_file)
    self._load_cache()
    
    if self._async_cache_write:
      self._init_async_writer()

  def _setup_cache_paths(self, cache_file: Optional[str]) -> None:
    """Setup cache paths."""
    if cache_file is None:
      cache_dir = pathlib.Path.home() / ".verl_cache"
      cache_dir.mkdir(exist_ok=True)
      self._cache_file = cache_dir / "bash_grep_cache.json"
    else:
      self._cache_file = pathlib.Path(cache_file)
      self._cache_file.parent.mkdir(parents=True, exist_ok=True)

  def _init_async_writer(self) -> None:
    """Initialize async writer."""
    self._stop_writer = threading.Event()
    self._writer_thread = threading.Thread(
      target=self._cache_writer_thread,
      daemon=True,
      name="BashGrepCacheWriter"
    )
    self._writer_thread.start()
    atexit.register(self._cleanup)

  def _cleanup(self) -> None:
    """Cleanup."""
    if self._async_cache_write and hasattr(self, '_stop_writer'):
      self._stop_writer.set()
      if self._writer_thread.is_alive():
        self._writer_thread.join(timeout=2.0)
      self._save_cache_sync()

  def _cache_writer_thread(self) -> None:
    """Cache writer thread."""
    while not self._stop_writer.is_set():
      try:
        try:
          _ = self._write_queue.get(timeout=1.0)
          self._save_cache_sync()
          self._write_queue.task_done()
        except queue.Empty:
          continue
      except Exception as e:
        print(f"Grep cache writer error: {str(e)}")

  def _load_cache(self) -> None:
    """Load cache."""
    if not self._cache_file.exists():
      return
    
    try:
      with open(self._cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        self._cache = data.get("cache", {})
        self._cache_timestamps = data.get("timestamps", {})
    except Exception as e:
      print(f"Failed to load grep cache: {str(e)}")
      self._cache = {}
      self._cache_timestamps = {}

  def _save_cache_sync(self) -> None:
    """Save cache sync."""
    try:
      with self._cache_lock:
        data = {
          "cache": dict(self._cache),
          "timestamps": dict(self._cache_timestamps)
        }
      
      temp_file = self._cache_file.with_suffix('.tmp')
      with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
      
      temp_file.replace(self._cache_file)
    except Exception as e:
      print(f"Failed to save grep cache: {str(e)}")

  def _save_cache(self) -> None:
    """Save cache."""
    if self._async_cache_write:
      try:
        self._write_queue.put(True, block=False)
      except queue.Full:
        pass
    else:
      self._save_cache_sync()


  def _is_cache_valid(self, cache_key: str) -> bool:
    """Check cache validity."""
    if cache_key not in self._cache_timestamps:
      return False
    
    timestamp = self._cache_timestamps[cache_key]
    return time.time() - timestamp < self._cache_refresh_interval

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
    
    # Cache key
    cache_key = json.dumps({"pattern": pattern, "search_path": search_path, "include_pattern": include_pattern}, sort_keys=True)
    
    # Check cache
    with self._cache_lock:
      if cache_key in self._cache and self._is_cache_valid(cache_key):
        return self._cache[cache_key]
    
    try:
      # Validate path
      validated_path = search_path
      
      # Build grep command
      cmd = ["grep", "-n", "-r"]  # Always include line numbers and recursive
      
      if include_pattern:
        cmd.extend(["--include", include_pattern])
      
      cmd.append(pattern)
      
      if validated_path != ".":
        cmd.append(validated_path)
      else:
        cmd.append(".")
      
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
      
      # Cache result
      response_str = json.dumps(response, ensure_ascii=False, indent=2)
      with self._cache_lock:
        self._cache[cache_key] = response_str
        self._cache_timestamps[cache_key] = time.time()
      
      self._save_cache()
      return response_str
      
    except TimeoutExpired:
      return json.dumps({"error": "Command timed out"})
    except Exception as e:
      return json.dumps({"error": f"Grep search failed: {str(e)}"})



class BashReadTool(BaseTool):
  """
  Bash read tool for file content reading with caching based on file modification time.
  """

  def __init__(
    self,
    working_dir: Optional[str] = None,
    default_limit: int = 0,
    default_offset: int = 0,
    cache_file: Optional[str] = None,
    async_cache_write: bool = True
  ):
    """Initialize read tool."""
    self._default_limit = default_limit
    self._default_offset = default_offset
    
    # Cache with file modification tracking
    self._cache = {}
    self._cache_mod_times = {}
    self._cache_lock = threading.Lock()
    self._async_cache_write = async_cache_write
    self._write_queue = queue.Queue() if async_cache_write else None
    
    # Set working directory
    if working_dir:
      self._cwd = Path(working_dir).resolve()
      if not self._cwd.exists():
        raise ValueError(f"Working directory does not exist: {working_dir}")
      if not self._cwd.is_dir():
        raise ValueError(f"Working directory is not a directory: {working_dir}")
    else:
      self._cwd = Path.cwd().resolve()

    
    self._setup_cache_paths(cache_file)
    self._load_cache()
    
    if self._async_cache_write:
      self._init_async_writer()

  def _setup_cache_paths(self, cache_file: Optional[str]) -> None:
    """Setup cache paths."""
    if cache_file is None:
      cache_dir = pathlib.Path.home() / ".verl_cache"
      cache_dir.mkdir(exist_ok=True)
      self._cache_file = cache_dir / "bash_read_cache.json"
    else:
      self._cache_file = pathlib.Path(cache_file)
      self._cache_file.parent.mkdir(parents=True, exist_ok=True)

  def _init_async_writer(self) -> None:
    """Initialize async writer."""
    self._stop_writer = threading.Event()
    self._writer_thread = threading.Thread(
      target=self._cache_writer_thread,
      daemon=True,
      name="BashReadCacheWriter"
    )
    self._writer_thread.start()
    atexit.register(self._cleanup)

  def _cleanup(self) -> None:
    """Cleanup."""
    if self._async_cache_write and hasattr(self, '_stop_writer'):
      self._stop_writer.set()
      if self._writer_thread.is_alive():
        self._writer_thread.join(timeout=2.0)
      self._save_cache_sync()

  def _cache_writer_thread(self) -> None:
    """Cache writer thread."""
    while not self._stop_writer.is_set():
      try:
        try:
          _ = self._write_queue.get(timeout=1.0)
          self._save_cache_sync()
          self._write_queue.task_done()
        except queue.Empty:
          continue
      except Exception as e:
        print(f"Read cache writer error: {str(e)}")

  def _load_cache(self) -> None:
    """Load cache."""
    if not self._cache_file.exists():
      return
    
    try:
      with open(self._cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        self._cache = data.get("cache", {})
        self._cache_mod_times = data.get("mod_times", {})
    except Exception as e:
      print(f"Failed to load read cache: {str(e)}")
      self._cache = {}
      self._cache_mod_times = {}

  def _save_cache_sync(self) -> None:
    """Save cache sync."""
    try:
      with self._cache_lock:
        data = {
          "cache": dict(self._cache),
          "mod_times": dict(self._cache_mod_times)
        }
      
      temp_file = self._cache_file.with_suffix('.tmp')
      with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
      
      temp_file.replace(self._cache_file)
    except Exception as e:
      print(f"Failed to save read cache: {str(e)}")

  def _save_cache(self) -> None:
    """Save cache."""
    if self._async_cache_write:
      try:
        self._write_queue.put(True, block=False)
      except queue.Full:
        pass
    else:
      self._save_cache_sync()

  def _is_cache_valid(self, filepath: str, cache_key: str) -> bool:
    """Check if cached content is still valid based on file modification time."""
    if cache_key not in self._cache_mod_times:
      return False
    
    try:
      current_mod_time = os.path.getmtime(filepath)
      cached_mod_time = self._cache_mod_times[cache_key]
      return current_mod_time <= cached_mod_time
    except OSError:
      return False

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
      
      # Cache key
      cache_key = json.dumps({"filepath": filepath}, sort_keys=True)
      
      # Check cache validity
      with self._cache_lock:
        if cache_key in self._cache and self._is_cache_valid(str(full_path), cache_key):
          return self._cache[cache_key]
      
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
      
      # Cache result with current file modification time
      response_str = json.dumps(response, ensure_ascii=False, indent=2)
      current_mod_time = os.path.getmtime(str(full_path))
      
      with self._cache_lock:
        self._cache[cache_key] = response_str
        self._cache_mod_times[cache_key] = current_mod_time
      
      self._save_cache()
      return response_str
      
    except Exception as e:
      return json.dumps({"error": f"Failed to read file: {str(e)}"})

