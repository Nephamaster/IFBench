#!/usr/bin/env python3
"""Generate IFBench responses through a vLLM OpenAI-compatible Chat API."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

from config import get_settings


def load_prompts(input_file: str) -> list[dict[str, Any]]:
    """Load prompt keys and prompt texts from the IFBench test file."""
    prompts: list[dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {input_file}:{line_number}: {exc}"
                ) from exc

            if "key" not in example or "prompt" not in example:
                raise ValueError(
                    f"Missing key/prompt at {input_file}:{line_number}"
                )
            prompts.append({"key": example["key"], "prompt": example["prompt"]})
    return prompts


def parse_stop_token_ids(value: str) -> list[int]:
    """Parse comma-separated vLLM stop token IDs."""
    try:
        return [
            int(token_id.strip())
            for token_id in value.split(",")
            if token_id.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "stop token IDs must be comma-separated integers"
        ) from exc


def parse_json_object(value: str) -> dict[str, Any]:
    """Parse a JSON object used as chat_template_kwargs."""
    if not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"chat template kwargs must be valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            "chat template kwargs must decode to a JSON object"
        )
    return parsed


def generate_response(
    client: httpx.Client,
    api_base: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    repetition_penalty: float,
    api_key: str | None,
    seed: int | None,
    stop_token_ids: list[int],
    chat_template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Generate one response and retain stopping/usage diagnostics."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # IFBench reports use greedy decoding (temperature=0).
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
    }
    if seed is not None:
        payload["seed"] = seed
    if stop_token_ids:
        # vLLM extension: stop when the model emits a chat-turn terminator.
        payload["stop_token_ids"] = stop_token_ids
    if chat_template_kwargs:
        # Used by Qwen3 to make non-thinking prompt construction explicit.
        payload["chat_template_kwargs"] = chat_template_kwargs

    response = client.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    response.raise_for_status()

    body = response.json()
    choice = body["choices"][0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}

    return {
        "response": message.get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        # vLLM may expose the concrete stop token/string in this field.
        "stop_reason": choice.get("stop_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def write_results(output_file: str, results: list[dict[str, Any]]) -> None:
    """Write generation results atomically enough for resumable evaluation."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Generate responses from a vLLM OpenAI-compatible API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-base", default=settings.api_base)
    parser.add_argument("--model", default=settings.model)
    parser.add_argument("--input-file", default=settings.input_file)
    parser.add_argument(
        "--output-file",
        help="Output file (default: data/{model}-responses.jsonl)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=settings.temperature,
        help="IFBench paper setting is 0.0",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=settings.max_tokens,
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=settings.repetition_penalty,
        help="Keep 1.0 for comparable IFBench evaluation",
    )
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--api-key", default=settings.api_key)
    parser.add_argument(
        "--stop-token-ids",
        default=settings.stop_token_ids,
        help=(
            "Comma-separated vLLM stop IDs. "
            "Qwen3: 151645 (<|im_end|>); Llama-3.1: 128009 (<|eot_id|>)"
        ),
    )
    parser.add_argument(
        "--chat-template-kwargs",
        default=settings.chat_template_kwargs,
        help='JSON object, e.g. {"enable_thinking": false}',
    )
    parser.add_argument("--workers", type=int, default=settings.workers)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by IFBench key from an existing output file",
    )

    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required (or set MODEL in .env)")
    if not args.api_base:
        parser.error("--api-base is required (or set API_BASE in .env)")
    if args.temperature < 0:
        parser.error("--temperature must be >= 0")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0")
    if args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be > 0")
    if args.workers <= 0:
        parser.error("--workers must be > 0")

    try:
        stop_token_ids = parse_stop_token_ids(args.stop_token_ids)
        chat_template_kwargs = parse_json_object(args.chat_template_kwargs)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    prompts = load_prompts(args.input_file)
    print(f"Loaded {len(prompts)} prompts from {args.input_file}")

    if not args.output_file:
        safe_model_name = args.model.replace("/", "-")
        args.output_file = f"data/{safe_model_name}-responses.jsonl"

    print(f"Model                : {args.model}")
    print(f"API                  : {args.api_base}")
    print(f"Temperature          : {args.temperature}")
    print(f"Max tokens           : {args.max_tokens}")
    print(f"Repetition penalty   : {args.repetition_penalty}")
    print(f"Stop token IDs       : {stop_token_ids or 'none'}")
    print(f"Chat template kwargs : {chat_template_kwargs or 'none'}")

    existing_keys: set[Any] = set()
    existing_prompts: set[str] = set()
    existing_responses: list[dict[str, Any]] = []

    if args.resume and Path(args.output_file).exists():
        with open(args.output_file, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                result = json.loads(line)
                if "key" in result:
                    existing_keys.add(result["key"])
                elif "prompt" in result:
                    # Compatibility with output produced by the old script.
                    existing_prompts.add(result["prompt"])
                existing_responses.append(result)
        print(f"Resuming: {len(existing_responses)} responses already completed")

    remaining = [
        prompt
        for prompt in prompts
        if prompt["key"] not in existing_keys
        and prompt["prompt"] not in existing_prompts
    ]
    print(f"Generating responses for {len(remaining)} prompts...")

    results = list(existing_responses)
    errors: list[dict[str, Any]] = []
    newly_completed = 0

    limits = httpx.Limits(
        max_connections=max(args.workers * 2, 20),
        max_keepalive_connections=max(args.workers, 10),
    )
    with httpx.Client(limits=limits) as client:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_prompt = {
                executor.submit(
                    generate_response,
                    client,
                    args.api_base,
                    args.model,
                    prompt["prompt"],
                    args.temperature,
                    args.max_tokens,
                    args.repetition_penalty,
                    args.api_key,
                    args.seed,
                    stop_token_ids,
                    chat_template_kwargs,
                ): prompt
                for prompt in remaining
            }

            with tqdm(total=len(remaining), desc="Generating") as progress:
                for future in as_completed(future_to_prompt):
                    prompt_data = future_to_prompt[future]
                    try:
                        generation = future.result()
                        results.append(
                            {
                                "key": prompt_data["key"],
                                "prompt": prompt_data["prompt"],
                                **generation,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        errors.append(
                            {
                                "key": prompt_data["key"],
                                "error": str(exc),
                            }
                        )
                        results.append(
                            {
                                "key": prompt_data["key"],
                                "prompt": prompt_data["prompt"],
                                "response": "",
                                "finish_reason": "error",
                                "stop_reason": None,
                                "prompt_tokens": None,
                                "completion_tokens": None,
                                "total_tokens": None,
                                "error": str(exc),
                            }
                        )

                    newly_completed += 1
                    progress.update(1)
                    if newly_completed % 10 == 0:
                        write_results(args.output_file, results)

    write_results(args.output_file, results)

    finish_reasons = Counter(
        result.get("finish_reason") for result in results
    )
    length_limited = finish_reasons.get("length", 0)

    print(f"\nSaved {len(results)} responses to {args.output_file}")
    print(f"Finish reasons: {dict(finish_reasons)}")
    if length_limited:
        print(
            f"[WARN] {length_limited} responses ended because max_tokens was reached. "
            "Inspect these cases for repetition or missing stop-token handling."
        )

    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors[:5]:
            print(f"  - Key {error['key']}: {error['error']}")
        raise SystemExit(1)

    print("\nRun evaluation with:")
    print(
        "  uv run python3 -m run_eval "
        f"--input_data={args.input_file} "
        f"--input_response_data={args.output_file} "
        "--output_dir=eval"
    )


if __name__ == "__main__":
    main()
