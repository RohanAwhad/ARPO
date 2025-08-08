import re
import sys
import string
import json
import requests
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import Counter
from transformers import AutoTokenizer


def validate_format_sdg(text: str) -> Tuple[bool, str]:
  """
  Validate if the text follows the required format for SDG agent with paired tags.

  Args:
      text: The text to validate

  Returns:
      tuple: (is_valid, reason)
  """
  if text.count('<think>') != text.count('</think>'):
    return False, "<think> </think> not paired"

  if text.count('<think>') == 0 or text.count('</think>') == 0:
    return False, "<think> or </think> not found"

  if text.count('<answer>') != 1 or text.count('</answer>') != 1:
    return False, "<answer> or </answer> not found"

  return True, "format is correct"


def extract_answer(text: str) -> Optional[str]:
  """
  Extract answer content from the text within <answer> tags.

  Args:
      text: The text to extract answer from

  Returns:
      Optional[str]: The extracted answer or None if no match
  """
  text = text.strip()

  pattern = r"<answer>(.*?)</answer>"
  match = re.search(pattern, text, re.DOTALL)
  if not match:
    return None

  return match.group(1).strip()


def count_tool_usage(text: str) -> Dict[str, int]:
    """
    Count the usage of different SDG tools in the response.

    Args:
        text: The response text to analyze

    Returns:
        Dict[str, int]: Dictionary with tool usage counts
    """
    tool_counts = {
        'find': text.count('</find>'),
        'grep': text.count('</grep>'),
        'read': text.count('</read>')
    }
    return tool_counts


def call_llm_judge(
    candidate_answer: str,
    ground_truth_answer: str,
    rubric: str,
    judge_endpoint: str = "http://localhost:8001/v1/chat/completions",
    judge_model: str = "Qwen2.5-72B-Instruct",
    judge_temperature: float = 0.1,
    judge_max_tokens: int = 2048,
    judge_timeout: int = 60
) -> Dict[str, Any]:
    """
    Call LLM judge to evaluate the candidate answer against ground truth using rubric.
    """

    system_prompt = """You are an expert evaluator tasked with judging the quality of answers about SDG (Sustainable Development Goals) related queries. You will be given:
1. A candidate answer to evaluate
2. A reference (ground truth) answer
3. An evaluation rubric

Your task is to evaluate the candidate answer against the reference answer using the provided rubric.

Please provide your evaluation in the following JSON format:
{
    "score": <float between 0.0 and 10.0>,
    "reasoning": "<detailed explanation of your scoring decision>",
    "strengths": "<what the candidate answer does well>",
    "weaknesses": "<what the candidate answer lacks or does poorly>"
}

Be objective and thorough in your evaluation. The score should reflect how well the candidate answer addresses the query according to the rubric."""

    user_prompt = f"""Please evaluate the following candidate answer against the reference answer using the provided rubric.

**Evaluation Rubric:**
{rubric}

**Reference Answer:**
{ground_truth_answer}

**Candidate Answer:**
{candidate_answer}

Please provide your evaluation in the specified JSON format."""

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": judge_temperature,
        "max_tokens": judge_max_tokens,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(judge_endpoint, json=payload, timeout=judge_timeout)
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse the JSON response
        evaluation = json.loads(content)

        # Ensure score is within valid range
        score = float(evaluation.get("score", 0.0)) / 10
        score = max(0.0, min(1.0, score))

        return {
            "score": score,
            "reasoning": evaluation.get("reasoning", ""),
            "strengths": evaluation.get("strengths", ""),
            "weaknesses": evaluation.get("weaknesses", ""),
            "raw_response": content
        }

    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM judge: {e}")
        return {"score": 0.0, "reasoning": f"LLM judge call failed: {e}", "strengths": "", "weaknesses": ""}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing LLM judge response: {e}")
        return {"score": 0.0, "reasoning": f"Failed to parse judge response: {e}", "strengths": "", "weaknesses": ""}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Dict[str, str],
    extra_info: Optional[Dict[str, Any]] = None,
    # Configuration parameters passed from Hydra config
    judge_endpoint: str = "http://localhost:8001/v1/chat/completions",
    judge_model: str = "Qwen2.5-72B-Instruct",
    judge_temperature: float = 0.1,
    judge_max_tokens: int = 2048,
    judge_timeout: int = 60,
    tool_bonus_multi: float = 0.1,
    tool_bonus_single: float = 0.05,
    min_tools_for_bonus: int = 2,
    **kwargs  # Capture any additional config parameters
) -> Dict[str, Any]:
    """
    Compute reward score for SDG agent solution using LLM as judge.
    All judge parameters are now configurable through Hydra config.
    """

    # Initialize return structure
    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "judge_score": 0.0,
        "judge_reasoning": "",
        "tool_usage": {},
        "tool_bonus": 0.0
    }

    response = solution_str

    # Validate format
    valid_template, reason = validate_format_sdg(response)

    if not valid_template:
        print(f"--------------------------------bad format: {reason}--------------------------------\nsolution_str: {solution_str}")
        result["score"] = -2
        result["reason"] = f"bad format: {reason}"
        return result

    # Remove EOS token if present
    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]

    # Extract answer
    answer_part = extract_answer(response)
    if answer_part is None:
        print(f"--------------------------------cannot extract answer--------------------------------\nsolution_str: {solution_str}")
        result["score"] = -1
        result["reason"] = "cannot extract answer"
        return result

    result["answer"] = answer_part

    ground_truth_answer = ground_truth['answer']
    rubric = ground_truth.get('rubric', 'No rubric found')

    # Count tool usage
    tool_usage = count_tool_usage(response)
    result["tool_usage"] = tool_usage

    # Calculate tool usage bonus using config parameters
    unique_tools_used = len([tool for tool, count in tool_usage.items() if count > 0])

    if unique_tools_used >= min_tools_for_bonus:
        result["tool_bonus"] = tool_bonus_multi
    elif unique_tools_used == 1:
        result["tool_bonus"] = tool_bonus_single

    # Call LLM judge with config parameters
    judge_result = call_llm_judge(
        candidate_answer=answer_part,
        ground_truth_answer=ground_truth_answer,
        rubric=rubric,
        judge_endpoint=judge_endpoint,
        judge_model=judge_model,
        judge_temperature=judge_temperature,
        judge_max_tokens=judge_max_tokens,
        judge_timeout=judge_timeout
    )

    judge_score = judge_result["score"]
    result["judge_score"] = judge_score
    result["judge_reasoning"] = judge_result["reasoning"]

    # Calculate final score
    base_score = judge_score
    final_score = base_score + result["tool_bonus"]

    # Cap the final score at 1.0
    final_score = min(1.0, final_score)

    result["score"] = final_score

    # Create detailed reason
    if judge_score > 0.7:
        quality_desc = "high quality"
    elif judge_score > 0.4:
        quality_desc = "moderate quality"
    else:
        quality_desc = "low quality"

    tool_desc = f"Used {unique_tools_used} different tools ({', '.join([tool for tool, count in tool_usage.items() if count > 0])})"

    result["reason"] = f"{quality_desc} answer (judge score: {judge_score:.3f}) + tool bonus: {result['tool_bonus']:.3f}. {tool_desc}. Final score: {final_score:.3f}"

    print(f"SDG Agent Evaluation:")
    print(f"  Judge Score: {judge_score:.3f}")
    print(f"  Tool Usage: {tool_usage}")
    print(f"  Tool Bonus: {result['tool_bonus']:.3f}")
    print(f"  Final Score: {final_score:.3f}")
    print(f"  Judge Config: {judge_model}@{judge_endpoint}, temp={judge_temperature}")

    return result


def compute_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[dict],
    extra_infos: list[dict] = None,
    judge_endpoint: str = "http://localhost:8001/v1/chat/completions",
    judge_model: str = "Qwen2.5-72B-Instruct",
    judge_temperature: float = 0.1,
    judge_max_tokens: int = 2048,
    judge_timeout: int = 60,
    tool_bonus_multi: float = 0.1,
    tool_bonus_single: float = 0.05,
    min_tools_for_bonus: int = 2,
    max_concurrency: int = 8,
    **unused_kwargs,
) -> list[dict]:
    """
    Vectorized batch judge: send N HTTP requests in parallel (bounded by a semaphore)
    and return list of score-dicts matching the input order.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    if extra_infos is None:
        extra_infos = [None] * len(solution_strs)

    semaphore = threading.Semaphore(max_concurrency)

    def process_single_item(args):
        data_source, solution_str, ground_truth, extra_info = args
        
        semaphore.acquire()
        try:
            # Initialize return structure
            result = {
                "score": 0,
                "reason": "",
                "answer": "",
                "judge_score": 0.0,
                "judge_reasoning": "",
                "tool_usage": {},
                "tool_bonus": 0.0
            }

            response = solution_str

            # Validate format
            valid_template, reason = validate_format_sdg(response)

            if not valid_template:
                result["score"] = -2
                result["reason"] = f"bad format: {reason}"
                return result

            # Remove EOS token if present
            if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
                response = response[:-len(extra_info["tokenizer"].eos_token)]

            # Extract answer
            answer_part = extract_answer(response)
            if answer_part is None:
                result["score"] = -1
                result["reason"] = "cannot extract answer"
                return result

            result["answer"] = answer_part

            ground_truth_answer = ground_truth['answer']
            rubric = ground_truth.get('rubric', 'No rubric found')

            # Count tool usage
            tool_usage = count_tool_usage(response)
            result["tool_usage"] = tool_usage

            # Calculate tool usage bonus
            unique_tools_used = len([tool for tool, count in tool_usage.items() if count > 0])

            if unique_tools_used >= min_tools_for_bonus:
                result["tool_bonus"] = tool_bonus_multi
            elif unique_tools_used == 1:
                result["tool_bonus"] = tool_bonus_single

            # Call LLM judge
            judge_result = call_llm_judge(
                candidate_answer=answer_part,
                ground_truth_answer=ground_truth_answer,
                rubric=rubric,
                judge_endpoint=judge_endpoint,
                judge_model=judge_model,
                judge_temperature=judge_temperature,
                judge_max_tokens=judge_max_tokens,
                judge_timeout=judge_timeout
            )

            judge_score = judge_result["score"]
            result["judge_score"] = judge_score
            result["judge_reasoning"] = judge_result["reasoning"]

            # Calculate final score
            base_score = judge_score
            final_score = base_score + result["tool_bonus"]

            # Cap the final score at 1.0
            final_score = min(1.0, final_score)

            result["score"] = final_score

            # Create detailed reason
            if judge_score > 0.7:
                quality_desc = "high quality"
            elif judge_score > 0.4:
                quality_desc = "moderate quality"
            else:
                quality_desc = "low quality"

            tool_desc = f"Used {unique_tools_used} different tools ({', '.join([tool for tool, count in tool_usage.items() if count > 0])})"

            result["reason"] = f"{quality_desc} answer (judge score: {judge_score:.3f}) + tool bonus: {result['tool_bonus']:.3f}. {tool_desc}. Final score: {final_score:.3f}"

            # Log individual sample completion
            print(f"Sample completed - Judge: {judge_score:.3f}, Tools: {unique_tools_used}, Bonus: {result['tool_bonus']:.3f}, Final: {final_score:.3f}")

            return result

            
        finally:
            semaphore.release()

    print(f"Starting batch evaluation with {len(solution_strs)} items, max_concurrency={max_concurrency}")
    
    # Spin up a small threadpool
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        # Preserve ordering
        batch_args = list(zip(data_sources, solution_strs, ground_truths, extra_infos))
        results = list(executor.map(process_single_item, batch_args))
    
    print(f"Batch evaluation completed. Sample scores: {[r['score'] for r in results[:3]]}")
    
    return results


if __name__ == "__main__":
    # Test the reward function
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    # Example test case
    response = """<think>
I need to find information about SDG goals in the codebase. Let me search for relevant files first.
</think>

<find>
<pattern>*.py</pattern>
<search_path>src</search_path>
<file_type>f</file_type>
</find>

<result>
Found files: sdg_goals.py, sustainability.py, environmental.py
</result>

<read>
<filepath>src/sdg_goals.py</filepath>
</read>

<result>
# SDG Goals Implementation
SDG_GOALS = {
    1: "No Poverty",
    2: "Zero Hunger",
    3: "Good Health and Well-being",
    ...
}
</result>

<think>
Now I have the information about SDG goals. The code defines the 17 sustainable development goals.
</think>

<answer>
The SDG repository contains a comprehensive implementation of the 17 Sustainable Development Goals as defined by the United Nations. The main goals include No Poverty, Zero Hunger, Good Health and Well-being, and 14 others, focusing on global sustainability and development challenges.
</answer>"""

    ground_truth = {
        "answer": "The repository implements the UN's 17 Sustainable Development Goals including poverty eradication, hunger elimination, health improvement, and environmental sustainability.",
        "rubric": "Evaluate based on: 1) Accuracy of SDG information (40%), 2) Completeness of goal coverage (30%), 3) Clarity of explanation (20%), 4) Use of repository evidence (10%)"
    }

    extra_info = {"tokenizer": tokenizer}

    result = compute_score("sdg_test", response, ground_truth, extra_info)
    print("Test Result:", result)
    
    # Test batch function
    print("\nTesting batch function:")
    batch_result = compute_score_batch(
        data_sources=["sdg_test"],
        solution_strs=[response], 
        ground_truths=[ground_truth],
        extra_infos=[extra_info]
    )
    print("Batch Test Result:", batch_result[0])

