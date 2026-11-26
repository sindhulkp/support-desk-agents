"""
agents/responder.py — Agent 3: draft the reply to the customer.

Job: write a helpful reply to the ticket, using ONLY the help articles the
retriever found. Two hard rules, written into the prompt:
  1. Grounding — every fact in the reply must come from the articles.
     If the articles don't cover the question, say "I don't know" and
     hand the ticket to a human instead of inventing a policy.
  2. Citations — the reply names the articles it used as [1], [2], [3],
     so a human can check the answer in seconds.

This is the second "THINK" step of the agent loop: the agent reasons
over the retrieved facts and produces the final output. Grounded
generation is the whole point of RAG — it is what stops the LLM from
hallucinating refund policies or delivery times that don't exist.
"""

SYSTEM_PROMPT = """You are a friendly customer support agent. ROLE: responder

Write a short, polite reply to the customer. Follow these rules strictly:
1. Use ONLY the information in the help articles provided below. Do not
   add facts, numbers, or policies from your own knowledge.
2. Cite every article you use like this: [1], [2], [3].
3. If the articles do not contain the answer, say clearly that you do not
   have that information and that a human agent will follow up. Never guess.
4. Keep the reply under 150 words. Be warm and direct.
5. If the customer has written before (see CUSTOMER HISTORY), acknowledge
   it briefly in one sentence at the start of the reply.
"""


def draft_reply(llm, ticket, retrieved_docs, prior_tickets=0,
                prior_categories=None):
    """Draft a reply. retrieved_docs is a list of (doc, score) from BM25.

    prior_tickets: how many times this customer has written before.
    prior_categories: the categories of those earlier tickets.
    Both come from the orchestrator's memory — the reply is written
    knowing the customer's history, not as if every ticket is the first.
    Returns the reply text.
    """
    if retrieved_docs:
        articles_text = "\n\n".join(
            f"ARTICLE {i + 1}: {doc['title']}\n{doc['text']}"
            for i, (doc, _score) in enumerate(retrieved_docs)
        )
    else:
        articles_text = "NO RELEVANT ARTICLES FOUND"

    history_text = ""
    if prior_tickets > 0:
        cats = ", ".join(prior_categories or [])
        history_text = (
            f"\n\nCUSTOMER HISTORY: This customer has written "
            f"{prior_tickets} time(s) before"
            + (f" about: {cats}." if cats else ".")
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"CUSTOMER TICKET:\n{ticket['subject']}\n{ticket['message']}\n\n"
                f"RETRIEVED ARTICLES:\n{articles_text}"
                f"{history_text}"
            ),
        },
    ]

    return llm.chat(messages, temperature=0.0)
