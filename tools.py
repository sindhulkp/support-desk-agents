"""
tools.py — small helpers for loading data files.

"Tools" in agent language are the functions an agent can call to interact
with the world. Here the tools are simple file loaders: read the sample
tickets, and list the knowledge-base articles. The retriever builds its
search index from these files, so adding a new help article is as easy as
dropping a new .md file into the knowledge/ folder — no code changes.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_tickets(path=None):
    """Load the sample tickets from data/tickets.json as a list of dicts."""
    path = path or os.path.join(BASE_DIR, "data", "tickets.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_ticket(tickets, ticket_id):
    """Find one ticket by id (e.g. 'T-1001'). Returns None if not found."""
    for ticket in tickets:
        if ticket["id"] == ticket_id:
            return ticket
    return None


def knowledge_dir():
    """Path to the folder of markdown help articles."""
    return os.path.join(BASE_DIR, "knowledge")


def load_env_file(path=None):
    """Load KEY=VALUE lines from a .env file into os.environ.

    A tiny replacement for the `python-dotenv` package (which we don't
    install, to keep dependencies to one). Lines starting with # are
    comments; existing real environment variables are never overwritten.
    """
    path = path or os.path.join(BASE_DIR, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
