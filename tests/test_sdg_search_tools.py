import sys
sys.path.append('ARPO/verl_arpo_entropy')

from verl.workers.rollout.tools.sdg_search_tool import BashFindTool, BashGrepTool, BashReadTool

def test_tools():
  working_dir = "/Users/rawhad/1_Projects/sdg_hub"
  
  print("=== Testing SDG Search Tools ===\n")
  
  # Initialize tools
  find_tool = BashFindTool(working_dir=working_dir)
  grep_tool = BashGrepTool(working_dir=working_dir)
  read_tool = BashReadTool(working_dir=working_dir)
  
  # Test 1: Find Python files
  print("1. Finding Python files:")
  find_query = "<pattern>*.py</pattern><search_path>.</search_path><file_type>f</file_type>"
  result = find_tool.execute(find_query)
  print(result)
  print("\n" + "="*50 + "\n")
  
  # Test 2: Grep for TODO
  print("2. Searching for TODO:")
  grep_query = "<pattern>TODO</pattern><search_path>.</search_path><include_pattern>*.py</include_pattern>"
  result = grep_tool.execute(grep_query)
  print(result)
  print("\n" + "="*50 + "\n")
  
  # Test 3: Read specific file
  print("3. Reading flow.yaml:")
  read_query = "<filepath>src/sdg_hub/flows/skills_tuning/annotation_classification/instructlab/flow.yaml</filepath>"
  result = read_tool.execute(read_query)
  print(result)

if __name__ == "__main__":
  test_tools()
