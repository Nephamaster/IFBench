"""Configuration for IFBench generation using pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchmarkSettings(BaseSettings):
    """Settings for running IFBench benchmarks."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API configuration
    api_base: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for the OpenAI-compatible API",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authentication",
    )
    model: str = Field(
        default="",
        description="Served model name",
    )

    # Generation parameters. IFBench reports use greedy decoding.
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature; use 0.0 for IFBench",
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum completion tokens",
    )
    repetition_penalty: float = Field(
        default=1.0,
        description="Keep 1.0 for comparable IFBench evaluation",
    )
    seed: int | None = Field(
        default=42,
        description="Random seed; mostly irrelevant for temperature=0",
    )
    stop_token_ids: str = Field(
        default="",
        description="Comma-separated vLLM stop token IDs",
    )
    chat_template_kwargs: str = Field(
        default="",
        description="JSON object forwarded as chat_template_kwargs",
    )

    # Benchmark parameters
    input_file: str = Field(
        default="data/IFBench_test.jsonl",
        description="Path to IFBench test file",
    )
    output_file: str = Field(
        default="data/responses.jsonl",
        description="Output file for responses",
    )
    workers: int = Field(
        default=8,
        description="Number of parallel requests",
    )


def get_settings() -> BenchmarkSettings:
    """Load settings from environment variables and .env."""
    return BenchmarkSettings()
