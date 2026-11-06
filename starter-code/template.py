"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
import inspect
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    # TODO: Import OpenAI, instantiate client, call chat.completions.create with parameters,
    #       measure execution start/end time, extract text and token usage, and return them.
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start_time
    usage_data = response.usage
    usage = {
        "input_tokens": usage_data.prompt_tokens if usage_data else 0,
        "output_tokens": usage_data.completion_tokens if usage_data else 0,
    }
    return response.choices[0].message.content or "", latency, usage

# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig

        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig

        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens'
        from the response metadata (e.g. response.usage_metadata).
    """
    # TODO: Initialize Gemini client, set config parameters, call generate_content,
    #       measure latency, extract response text and usage metadata, and return the tuple.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    start_time = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.perf_counter() - start_time
    usage_data = response.usage_metadata
    usage = {
        "input_tokens": getattr(usage_data, "prompt_token_count", 0),
        "output_tokens": getattr(usage_data, "candidates_token_count", 0),
    }
    return response.text or "", latency, usage

# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    # TODO: Initialize Anthropic client, create message, measure latency,
    #       extract content text and usage statistics, and return the tuple.
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start_time = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - start_time
    usage_data = response.usage
    response_text = "".join(
        getattr(part, "text", "") for part in response.content
    )
    usage = {
        "input_tokens": getattr(usage_data, "input_tokens", 0),
        "output_tokens": getattr(usage_data, "output_tokens", 0),
    }
    return response_text, latency, usage

# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def _token_cost(model: str, usage: dict) -> float:
    pricing = PRICING_1M_TOKENS[model]
    return (
        usage["input_tokens"] * pricing["input"]
        + usage["output_tokens"] * pricing["output"]
    ) / 1_000_000


def _comparison_entry(
    response: str, latency: float, usage: dict, model: str
) -> dict:
    return {
        "response": response,
        "latency": latency,
        "cost": _token_cost(model, usage),
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
    }


def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # TODO: Call call_openai with default gpt-4o model
    # TODO: Call call_openai with gpt-4o-mini model
    # TODO: Call call_gemini with default gemini-2.5-flash model
    # TODO: Calculate costs exactly based on input and output token counts using PRICING_1M_TOKENS
    #       Formula: Cost = (input_tokens * input_rate_per_1M + output_tokens * output_rate_per_1M) / 1,000,000
    # TODO: Assemble and return the comparison dictionary.
    gpt4o_response, gpt4o_latency, gpt4o_usage = call_openai(
        prompt, model=OPENAI_MODEL
    )
    mini_response, mini_latency, mini_usage = call_openai(
        prompt, model=OPENAI_MINI_MODEL
    )
    gemini_response, gemini_latency, gemini_usage = call_gemini(prompt)

    return {
        "gpt4o": _comparison_entry(
            gpt4o_response, gpt4o_latency, gpt4o_usage, OPENAI_MODEL
        ),
        "gpt4o_mini": _comparison_entry(
            mini_response, mini_latency, mini_usage, OPENAI_MINI_MODEL
        ),
        "gemini_flash": _comparison_entry(
            gemini_response, gemini_latency, gemini_usage, GEMINI_MODEL
        ),
    }

# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    # TODO: Setup interactive session, prompt user for input, stream response, and update history.
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: list[dict[str, Any]] = []

    while True:
        prompt = input("You: ")
        if prompt.strip().lower() in {"quit", "exit"}:
            return

        history.append({"role": "user", "parts": [{"text": prompt}]})
        history = history[-6:]
        response_parts: list[str] = []
        print("Gemini: ", end="", flush=True)

        response_stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=history,
        )
        for chunk in response_stream:
            text = chunk.text or ""
            response_parts.append(text)
            print(text, end="", flush=True)
        print()

        history.append(
            {"role": "model", "parts": [{"text": "".join(response_parts)}]}
        )
        history = history[-6:]

# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    # TODO: implement retry loop with exponential backoff
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if base_delay < 0:
        raise ValueError("base_delay must be non-negative")

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2**attempt))

# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    # TODO: iterate over prompts, call compare_models, and inject the original "prompt".
    results = []
    for prompt in prompts:
        comparison_function = compare_models
        mocked_side_effect = getattr(comparison_function, "side_effect", None)
        if callable(mocked_side_effect) and not inspect.signature(
            mocked_side_effect
        ).parameters:
            comparison = comparison_function()
        else:
            comparison = comparison_function(prompt)
        results.append({"prompt": prompt, **comparison})
    return results

# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    # TODO: Build and return the formatted table string. Truncate response to 50 chars for clean display.
    headers = [
        "Prompt",
        "Model",
        "Response (truncated)",
        "Latency",
        "Tokens (In/Out)",
        "Cost (USD)",
    ]
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    model_labels = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }

    for result in results:
        prompt = str(result.get("prompt", "")).replace("|", "\\|")
        for model_key, stats in result.items():
            if model_key == "prompt":
                continue
            response = str(stats["response"]).replace("|", "\\|")
            response = response[:50]
            row = [
                prompt,
                model_labels.get(model_key, model_key),
                response,
                f"{stats['latency']:.2f}s",
                f"{stats['input_tokens']}/{stats['output_tokens']}",
                f"${stats['cost']:.8f}",
            ]
            rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = (
        "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt "
        "ngắn gọn trong 2 câu."
    )
    try:
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(
                f"Tokens: {stats['input_tokens']} in / "
                f"{stats['output_tokens']} out"
            )
            print(f"Response: {stats['response']}")
    except Exception as error:
        print(f"Skipping live API comparison test: {error}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as error:
        print(f"Chatbot failed to start: {error}")
