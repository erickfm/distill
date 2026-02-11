"""
Data preparation for Knowledge Acquisition task.

The paper constructs a corpus of Wikipedia articles describing natural
disasters that occurred in 2025 (after the model's knowledge cutoff).

Articles:
  - 2025 Myanmar earthquake
  - 2025 Kamchatka earthquake
  - 2025 Uttarakhand flash flood
  - Typhoon Kalmaegi
  - Tropical Storm Wipha
  - Cyclonic Ditwah
  - Hurricane Melissa
  - Kentwood Carson Tornado
  - July 2025 Central Texas floods

QA pairs are generated from these articles following Mecklenburg et al. (2024).
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

from datasets import load_dataset


DATA_DIR = Path(__file__).parent


# Wikipedia article titles for the knowledge acquisition task
WIKI_ARTICLES = [
    "2025 Myanmar earthquake",
    "2025 Kamchatka earthquake",
    "2025 Uttarakhand flash flood",
    "Typhoon Kalmaegi (2025)",
    "Tropical Storm Wipha (2025)",
    "Cyclone Ditwah",
    "Hurricane Melissa (2025)",
    "Kentwood–Carson tornado",
    "July 2025 Central Texas floods",
]


QA_GENERATION_PROMPT = """You are a helpful assistant that helps me write questions for an exam. You \
will be given a wiki article and you will need to write 100 questions on the content of the wiki article. \
The question should require recalling multiple pieces of information from the wiki article. \
Do not repeat the same question.

The questions should be in the following format:
Question: <question>
Answer: <answer>

Article:
{article}
"""


def prep_knowledge(
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Prepare Knowledge Acquisition dataset.

    This downloads the Wikipedia articles and stores them. The QA pairs
    need to be generated separately using an LLM (GPT-5 in the paper).

    For reproduction, we provide:
    1. The raw articles (for teacher context / CPT baselines)
    2. A placeholder for QA pairs that should be generated
    """
    output_dir = output_dir or DATA_DIR / "knowledge"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing Knowledge Acquisition dataset...")

    # Save article list and prompts for QA generation
    articles_dir = output_dir / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)

    # Try to fetch articles from Wikipedia
    try:
        import wikipedia

        for title in WIKI_ARTICLES:
            print(f"  Fetching: {title}")
            try:
                page = wikipedia.page(title, auto_suggest=False)
                article = {
                    "title": page.title,
                    "content": page.content,
                    "url": page.url,
                    "summary": page.summary,
                }
                safe_name = title.replace(" ", "_").replace("/", "_")
                with open(articles_dir / f"{safe_name}.json", "w") as f:
                    json.dump(article, f, indent=2)
            except Exception as e:
                print(f"    WARNING: Could not fetch '{title}': {e}")

    except ImportError:
        print("  NOTE: 'wikipedia' package not installed. Skipping article fetching.")
        print("  Install with: pip install wikipedia")

    # Save the QA generation prompt template
    with open(output_dir / "qa_generation_prompt.txt", "w") as f:
        f.write(QA_GENERATION_PROMPT)

    # Save article list
    with open(output_dir / "article_list.json", "w") as f:
        json.dump(WIKI_ARTICLES, f, indent=2)

    print(f"\nArticles saved to {articles_dir}")
    print(f"QA generation prompt saved to {output_dir / 'qa_generation_prompt.txt'}")
    print("\nTo generate QA pairs, run the QA generation prompt against each article")
    print("using an LLM (the paper uses GPT-5), then save as train.jsonl / test.jsonl")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Knowledge Acquisition dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prep_knowledge(seed=args.seed)
    print("\nDone!")

