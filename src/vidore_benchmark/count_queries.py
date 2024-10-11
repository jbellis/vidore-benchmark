from typing import cast

import huggingface_hub
from datasets import Dataset, load_dataset


def main():
    collection = huggingface_hub.get_collection("vidore/vidore-benchmark-667173f98e70a1c0fa4db00d")
    datasets = collection.items

    for dataset_item in datasets:
        print(f"\nCounting queries for {dataset_item.item_id}")
        dataset = cast(Dataset, load_dataset(dataset_item.item_id, split="test"))
        num_queries = len(set(dataset["query"]))
        print(f"Number of unique queries: {num_queries}")

if __name__ == "__main__":
    main()
