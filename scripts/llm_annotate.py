import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset
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
    parser = argparse.ArgumentParser("Annotate a paper using a language model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dataset", type=str, default="ljvmiranda921/edgeml-ltl-survey", help="Path to the input dataset.")
    parser.add_argument("--output_dataset", type=str, default="ljvmiranda921/edgeml-ltl-survey-annotations", help="Path to save the annotated dataset.")
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
        logging.warning("OPENAI_API_KEY is not set!")

    df = load_dataset(args.input_dataset, split="train").to_pandas()
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
        )
    )

    results_df = pd.DataFrame(results)
    merged_df = df.merge(results_df, on=args.id_column, how="left")

    ds = Dataset.from_pandas(merged_df)
    ds.push_to_hub(args.output_dataset, private=True)
    logging.info(f"Saved annotated dataset to HuggingFace Hub at {args.output_dataset}")
    output_dir = Path(f"data/{args.output_dataset.replace('/', '___')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(output_dir)
    merged_df.to_csv(output_dir / "papers_with_annotations.csv", index=False)
    logging.info(f"Also saved dataset locally at {output_dir}")


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

    for batch_num, batch in enumerate(tqdm_asyncio(batches, desc="Processing batches", unit="batch"),start=1):  # fmt: skip
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
