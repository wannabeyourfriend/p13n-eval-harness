"""LaMP task label sets. Copied verbatim from upstream data/datasets.py — no
torch/datasets imports so we can use it without the heavy ML stack."""


def get_all_labels(task: str) -> list[str]:
    if task == "LaMP-1":
        return ["[1]", "[2]"]
    if task == "LaMP-2":
        return [
            "sci-fi", "based on a book", "comedy", "action", "twist ending",
            "dystopia", "dark comedy", "classic", "psychology", "fantasy",
            "romance", "thought-provoking", "social commentary", "violence",
            "true story",
        ]
    if task == "LaMP-3":
        return ["1", "2", "3", "4", "5"]
    if task in {"LaMP-4", "LaMP-5", "LaMP-6", "LaMP-7"}:
        return []
    raise ValueError(f"Unknown LaMP task: {task}")
