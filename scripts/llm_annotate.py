"""Annotate S2 papers using an LLM (OpenAI) for pipeline stage, topics, and relevance."""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from scripts.prompts import SYSTEM_PROMPT, USER_PROMPT, ResearchPaperAnnotation

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

# Suppress httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser("Annotate papers using a language model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_csv", type=str, default="data/s2_papers/papers.csv", help="Path to the input CSV file.")
    parser.add_argument("-o", "--output_dir", type=str, default="data/llm_annotate", help="Directory to save annotated CSV.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18", help="Language model to use for annotation.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of papers to annotate in each batch.")
    parser.add_argument("--limit", type=int, default=-1, help="Number of papers to annotate. Set to -1 for no limit.")
    parser.add_argument("--id_column", type=str, default="s2_id", help="Column name for the unique paper identifier.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        logging.error("OPENAI_API_KEY is not set!")
        sys.exit(1)

    df = pd.read_csv(args.input_csv)
    logging.info(f"Loaded {len(df)} papers from {args.input_csv}")

    if args.limit > 0:
        df = df.head(args.limit)

    requests_list = []
    for _, row in df.iterrows():
        title = row.get("title", "")
        abstract = row.get("abstract", "")
        user_prompt = USER_PROMPT.format(title=title, abstract=abstract)
        request = (
            row.get(args.id_column),
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        requests_list.append(request)

    client = AsyncOpenAI(api_key=api_key)
    results = asyncio.run(
        submit_async_requests(
            requests_list,
            model=args.model,
            client=client,
            batch_size=args.batch_size,
            response_model=ResearchPaperAnnotation,
            id_column=args.id_column,
        )
    )

    results_df = pd.DataFrame(results)
    merged_df = df.merge(results_df, on=args.id_column, how="left")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{timestamp}_llm_annotations.csv"
    merged_df.to_csv(output_path, index=False)
    logging.info(f"Saved {len(merged_df)} annotated papers to {output_path}")


async def submit_async_requests(
    messages: list[tuple[str, list[dict]]],
    *,
    model: str,
    client: AsyncOpenAI,
    batch_size: int,
    response_model: type[BaseModel],
    delay_between_batches: float = 1.0,
    id_column: str = "s2_id",
) -> list[BaseModel]:
    """Submit async requests to OpenAI API in batches."""
    all_results = []
    total_batches = (len(messages) + batch_size - 1) // batch_size

    batches = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]  # fmt: skip

    for batch_num, batch in enumerate(tqdm_asyncio(batches, desc="Processing batches", unit="batch"), start=1):  # fmt: skip
        results = await _process_batch(
            batch,
            client=client,
            model=model,
            response_model=response_model,
            id_column=id_column,
        )
        all_results.extend(results)

        if batch_num < total_batches:
            await asyncio.sleep(delay_between_batches)

    return all_results


async def _process_batch(
    requests: list[tuple[str, list[dict]]],
    *,
    client: AsyncOpenAI,
    model: str,
    response_model: type[BaseModel],
    id_column: str = "s2_id",
) -> list[BaseModel]:
    """Process a batch of requests concurrently."""
    tasks = [
        _process_single_request(
            client,
            request,
            model=model,
            response_model=response_model,
            id_column=id_column,
        )
        for request in requests
    ]
    return await asyncio.gather(*tasks)


async def _process_single_request(
    client: AsyncOpenAI,
    inputs: tuple[str, list[dict]],
    *,
    model: str,
    response_model: type[BaseModel],
    id_column: str = "s2_id",
) -> dict[str, Any]:
    """Process a single async OpenAI request with structured output."""
    s2_id, messages = inputs
    response = await client.responses.parse(
        input=messages,
        model=model,
        text_format=response_model,
    )
    parsed_output = response.output_parsed.model_dump()
    return {id_column: s2_id, **parsed_output}


if __name__ == "__main__":
    main()
