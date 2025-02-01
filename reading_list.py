#!python
import requests
import yaml
import random
import sys
import os
import xml.etree.ElementTree as ET


def fetch_arxiv_metadata(arxiv_id):
    """Fetch metadata from arXiv."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch metadata for {arxiv_id}.")

    # Parse XML response
    root = ET.fromstring(response.content)
    entry = root.find("{http://www.w3.org/2005/Atom}entry")
    if entry is None:
        raise Exception(f"No entry found for arXiv ID {arxiv_id}.")

    # Extract title
    title = entry.find("{http://www.w3.org/2005/Atom}title")
    if title is None or not title.text:
        raise Exception(f"Failed to parse title for arXiv ID {arxiv_id}.")

    # Extract authors
    authors = []
    for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
        name = author.find("{http://www.w3.org/2005/Atom}name")
        if name is not None and name.text:
            authors.append(name.text.strip())

    # Extract link to the paper
    link = entry.find("{http://www.w3.org/2005/Atom}id")
    if link is None or not link.text:
        raise Exception(f"Failed to find URL for arXiv ID {arxiv_id}.")

    # Extract published date (only date part)
    published = entry.find("{http://www.w3.org/2005/Atom}published")
    if published is None or not published.text:
        raise Exception(f"Failed to find published date for arXiv ID {arxiv_id}.")
    published_date = published.text.strip().split("T")[0]  # Keep only the date part

    return {
        "arxiv_id": arxiv_id,
        "title": title.text.strip(),
        "authors": ", ".join(authors),
        "url": link.text.strip(),
        "published": published_date,
    }


def smart_title_case(text):
    words = text.split()  # Split the string into words
    capitalized_words = [
        word if word[:1].isupper() else word.capitalize() for word in words
    ]
    return " ".join(capitalized_words)  # Join the processed words back


def normalize_tags(tags):
    """Normalize tags by capitalizing the first letter of each word."""
    return [smart_title_case(tag) for tag in tags]


def update_papers_file(arxiv_id, notes, tags):
    """Update the _papers.yml file."""
    papers_file = "_data/papers.yml"

    # Load existing papers
    if os.path.exists(papers_file):
        with open(papers_file, "r") as file:
            papers = yaml.safe_load(file) or []
    else:
        papers = []

    # Check if arXiv ID already exists
    for paper in papers:
        if paper["arxiv_id"] == arxiv_id:
            print(f"Paper with ID {arxiv_id} already exists:")
            print(f"  Title: {paper['title']}")
            print(f"  Authors: {paper['authors']}")
            print(f"  Published: {paper['published']}")
            print(f"  URL: {paper['url']}")
            print(f"  Notes: {paper.get('notes', '')}")
            print(f"  Tags: {', '.join(paper.get('tags', []))}")

            override = input("Would you like to override it? (y/n): ").strip().lower()
            if override != "y":
                print("Operation aborted. No changes were made.")
                sys.exit(0)
            else:
                papers.remove(paper)
                break

    # Fetch metadata
    metadata = fetch_arxiv_metadata(arxiv_id)
    metadata["notes"] = notes
    metadata["tags"] = normalize_tags(tags)

    # Append to papers list
    papers.append(metadata)

    # Save updated file
    with open(papers_file, "w") as file:
        yaml.dump(papers, file, sort_keys=False)

    return metadata


def update_tag_colors(tags):
    """Update the tag_colors.yml file."""
    colors_file = "_data/tag_colors.yml"

    # Load existing tag colors
    if os.path.exists(colors_file):
        with open(colors_file, "r") as file:
            tag_colors = yaml.safe_load(file) or {}
    else:
        tag_colors = {}

    # Assign random colors to new tags
    for tag in tags:
        if tag not in tag_colors:
            tag_colors[tag] = f"#{random.randint(0, 0xFFFFFF):06x}"

    # Save updated file
    with open(colors_file, "w") as file:
        yaml.dump(tag_colors, file)

    return tag_colors


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <arxiv_id> [notes] [tags...]")
        sys.exit(1)

    arxiv_id = sys.argv[1]
    notes = sys.argv[2] if len(sys.argv) > 2 else ""
    tags = sys.argv[3:] if len(sys.argv) > 3 else []

    # Normalize tags
    tags = normalize_tags(tags)

    # Update papers and tag colors
    metadata = update_papers_file(arxiv_id, notes, tags)
    update_tag_colors(tags)

    # Success message
    print(
        f"Successfully added/updated paper:\n  Title: {metadata['title']}\n  Authors: {metadata['authors']}\n  Published: {metadata['published']}\n  URL: {metadata['url']}\n  Notes: {notes}\n  Tags: {', '.join(tags)}"
    )


if __name__ == "__main__":
    main()
