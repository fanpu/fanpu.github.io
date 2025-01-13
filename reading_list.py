import arxiv
import os
import argparse

# Path to the reading list markdown file
READING_LIST_PATH = "_pages/reading_list.md"


def fetch_arxiv_title(arxiv_id):
    """Fetch the title of an arXiv paper given its ID."""
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(search.results())
        return result.title.strip()
    except StopIteration:
        raise ValueError(f"No paper found with arXiv ID {arxiv_id}.")


def append_to_reading_list(arxiv_id, title):
    """Append the paper title and link to the reading list file."""
    link = f"https://arxiv.org/abs/{arxiv_id}"
    entry = f"- [{title}]({link})\n"

    if not os.path.exists(READING_LIST_PATH):
        raise FileNotFoundError(f"The file {READING_LIST_PATH} does not exist.")

    with open(READING_LIST_PATH, "a") as file:
        file.write(entry)

    print(f"Appended to {READING_LIST_PATH}: {entry.strip()}")


def main():
    parser = argparse.ArgumentParser(
        description="Append an arXiv paper to the reading list."
    )
    parser.add_argument("arxiv_id", type=str, help="The arXiv ID of the paper.")
    args = parser.parse_args()

    arxiv_id = args.arxiv_id
    try:
        title = fetch_arxiv_title(arxiv_id)
        append_to_reading_list(arxiv_id, title)
    except ValueError as e:
        print(e)
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()
