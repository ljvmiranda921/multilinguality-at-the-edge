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
from openai import AsyncAzureOpenAI, AsyncOpenAI
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
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Language model to use for annotation.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of papers to annotate in each batch.")
    parser.add_argument("--limit", type=int, default=-1, help="Number of papers to annotate. Set to -1 for no limit.")
    parser.add_argument("--id_column", type=str, default="s2_id", help="Column name for the unique paper identifier.")
    parser.add_argument("--use_azure", action="store_true", help="Use Azure OpenAI instead of OpenAI. Requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env.")
    parser.add_argument("--azure_api_version", type=str, default="2024-12-01-preview", help="Azure OpenAI API version.")
    parser.add_argument("--resume", type=str, default=None, help="Path to an existing output CSV to resume from (skips already-annotated papers).")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    if args.use_azure:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_endpoint or not azure_key:
            logging.error("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env")
            sys.exit(1)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            logging.error("OPENAI_API_KEY is not set!")
            sys.exit(1)

    df = pd.read_csv(args.input_csv)
    logging.info(f"Loaded {len(df)} papers from {args.input_csv}")

    if args.limit > 0:
        df = df.head(args.limit)

    # Set up output path (use --resume to append to an existing file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        output_path = Path(args.resume)
        if not output_path.exists():
            logging.error(f"Resume file not found: {output_path}")
            sys.exit(1)
        existing_df = pd.read_csv(output_path)
        done_ids = set(existing_df[args.id_column].dropna().astype(str))
        logging.info(f"Resuming from {output_path} ({len(done_ids)} papers already annotated)")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{timestamp}_llm_annotations.csv"
        done_ids = set()

    requests_list = []
    for _, row in df.iterrows():
        paper_id = str(row.get(args.id_column))
        if paper_id in done_ids:
            continue
        title = row.get("title", "")
        abstract = row.get("abstract", "")
        user_prompt = USER_PROMPT.format(title=title, abstract=abstract)
        request = (
            paper_id,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        requests_list.append(request)

    logging.info(f"{len(requests_list)} papers to annotate ({len(done_ids)} already done)")

    if not requests_list:
        logging.info("Nothing to do!")
        return

    if args.use_azure:
        client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=args.azure_api_version,
        )
    else:
        client = AsyncOpenAI(api_key=api_key)

    asyncio.run(
        submit_async_requests(
            requests_list,
            model=args.model,
            client=client,
            batch_size=args.batch_size,
            response_model=ResearchPaperAnnotation,
            id_column=args.id_column,
            output_path=output_path,
            input_df=df,
        )
    )

    logging.info(f"Done! Results saved to {output_path}")


async def submit_async_requests(
    messages: list[tuple[str, list[dict]]],
    *,
    model: str,
    client: AsyncOpenAI,
    batch_size: int,
    response_model: type[BaseModel],
    delay_between_batches: float = 1.0,
    id_column: str = "s2_id",
    output_path: Path,
    input_df: pd.DataFrame,
) -> None:
    """Submit async requests to OpenAI API in batches, appending results to CSV after each batch."""
    total_batches = (len(messages) + batch_size - 1) // batch_size
    write_header = not output_path.exists()

    batches = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]  # fmt: skip

    for batch_num, batch in enumerate(tqdm_asyncio(batches, desc="Processing batches", unit="batch"), start=1):  # fmt: skip
        results = await _process_batch(
            batch,
            client=client,
            model=model,
            response_model=response_model,
            id_column=id_column,
        )

        # Merge with input data and append to CSV
        results_df = pd.DataFrame(results)
        batch_ids = results_df[id_column].tolist()
        input_subset = input_df[input_df[id_column].astype(str).isin(batch_ids)]
        merged = input_subset.merge(results_df, on=id_column, how="left")
        merged.to_csv(output_path, mode="a", header=write_header, index=False)
        write_header = False
        logging.info(f"Batch {batch_num}/{total_batches}: appended {len(merged)} rows to {output_path}")

        if batch_num < total_batches:
            await asyncio.sleep(delay_between_batches)


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
    response = await client.beta.chat.completions.parse(
        messages=messages,
        model=model,
        response_format=response_model,
    )
    parsed_output = response.choices[0].message.parsed.model_dump()
    return {id_column: s2_id, **parsed_output}


if __name__ == "__main__":
    main()
