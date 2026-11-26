"""
llm.py — The only place in the project that talks to a language model.

Two classes live here:
  * LLMClient — a thin wrapper around the `openai` package. It sends a list
    of chat messages to any OpenAI-compatible API and returns the reply text.
  * MockLLM — a rule-based fake LLM used in demo mode. It reads the same
    prompts the real LLM would get and answers with simple rules, so the
    whole pipeline runs with NO api key and NO internet.

Why a thin wrapper? Every agent in this project needs the same two things:
send a prompt, get text back. If the API details ever change (a new model,
a different provider), only this file changes — the agents stay the same.

Configuration comes ONLY from environment variables (never hardcoded keys),
read through config.py:
  LLM_API_KEY   — your API key (e.g. an OpenAI key, or a free Google Gemini key)
  LLM_BASE_URL  — which API server to talk to (default: OpenAI)
  LLM_MODEL     — which model to use (default: gpt-4o-mini)

To use Google Gemini for free, set:
  LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
  LLM_MODEL=gemini-2.0-flash
Gemini offers an OpenAI-compatible endpoint, so the same code works.
"""

import json
import re

from config import settings


# ---------------------------------------------------------------------------
# Real LLM client (uses the `openai` package)
# ---------------------------------------------------------------------------
class LLMClient:
    """Sends chat-completion requests to an OpenAI-compatible API."""

    def __init__(self, api_key=None, base_url=None, model=None):
        # Defaults come from config.py (which reads the environment).
        # Explicit arguments win over the environment — useful in tests.
        self.api_key = api_key or settings.llm_api_key
        self.base_url = base_url or settings.llm_base_url
        self.model = model or settings.llm_model

        if not self.api_key:
            raise ValueError(
                "No API key found. Set the LLM_API_KEY environment variable, "
                "or run with --mock / --demo to use the built-in fake LLM."
            )

        # Imported here (not at the top of the file) on purpose: demo mode
        # must work even if the `openai` package is not installed yet.
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages, temperature=0.0, json_mode=False):
        """Send messages, return the model's reply as a plain string.

        messages: a list like [{"role": "system", "content": "..."},
                               {"role": "user", "content": "..."}]
        temperature: 0.0 = deterministic (best for support answers).
        json_mode: when True, ask the model to reply with valid JSON only.
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            # response_format tells the API: "reply with a JSON object".
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Mock LLM (rule-based fake, for demo mode — no key, no internet)
# ---------------------------------------------------------------------------
class MockLLM:
    """A fake LLM that answers with simple keyword rules.

    It looks at the system prompt to figure out WHICH agent is calling it
    (classifier or responder) and then applies plain rules to the ticket.
    This is deliberately simple: its job is to let you run and demo the
    whole pipeline without an API key.
    """

    def chat(self, messages, temperature=0.0, json_mode=False):
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""

        if "ROLE: classifier" in system:
            return self._classify(user)
        if "ROLE: responder" in system:
            return self._respond(user)
        return "I am the mock LLM. I only know how to play classifier and responder."

    # -- classifier rules -------------------------------------------------
    def _classify(self, user_prompt):
        # The ticket text sits after the "TICKET:" marker in the prompt.
        ticket = user_prompt.split("TICKET:", 1)[-1].lower()

        # Category: check the most specific signals first. A damaged item
        # that mentions "refund" is still a returns problem, not billing.
        if any(w in ticket for w in ["damaged", "broken", "cracked", "defective"]):
            category = "returns"
        elif any(w in ticket for w in ["refund", "charged twice", "billing", "payment", "invoice", "subscription"]):
            category = "billing"
        elif any(w in ticket for w in ["login", "password", "crash", "app", "error", "reset"]):
            category = "technical"
        elif any(w in ticket for w in ["delivery", "tracking", "shipped", "package", "address", "arrive"]):
            category = "shipping"
        elif any(w in ticket for w in ["return", "exchange", "replacement"]):
            category = "returns"
        else:
            category = "general"

        # Priority: urgent words mean high priority.
        if any(w in ticket for w in ["urgent", "asap", "immediately", "angry", "furious", "twice", "cannot", "can't"]):
            priority = "high"
        elif any(w in ticket for w in ["please", "wondering", "question", "when"]):
            priority = "normal"
        else:
            priority = "low"

        # Sentiment: angry / disappointed words mean negative. "Please" is
        # just politeness, not a positive feeling, so it is not counted.
        if any(w in ticket for w in ["angry", "furious", "terrible", "worst", "frustrated", "unacceptable", "disappointed", "disappointing"]):
            sentiment = "negative"
        elif any(w in ticket for w in ["thanks", "thank you", "great", "love", "happy", "wonderful"]):
            sentiment = "positive"
        else:
            sentiment = "neutral"

        # Return exactly the JSON shape the real LLM is asked for.
        return json.dumps(
            {"category": category, "priority": priority, "sentiment": sentiment}
        )

    # -- responder rules --------------------------------------------------
    def _respond(self, user_prompt):
        # If the retriever found nothing, the honest answer is "I don't know".
        if "NO RELEVANT ARTICLES FOUND" in user_prompt:
            return (
                "Hi, thanks for reaching out. I checked our help articles and "
                "I don't have information about this in our knowledge base, so "
                "I don't want to guess. I've passed your ticket to a human "
                "support agent who will follow up with you shortly."
            )

        # Otherwise, pull each article (title + a short snippet of its text)
        # out of the prompt and cite it. This is the mock version of
        # "grounded generation": the reply quotes the retrieved facts.
        blocks = re.findall(
            r"ARTICLE \d+: (.+?)\n(.*?)(?=ARTICLE \d+:|$)",
            user_prompt,
            flags=re.DOTALL,
        )

        lines = ["Hi, thanks for getting in touch — here's what I found in our help articles:\n"]
        for i, (title, body) in enumerate(blocks, start=1):
            snippet = self._first_meaningful_lines(body)
            lines.append(f"[{i}] {title.strip()}: {snippet}")
        lines.append(
            "\nIf those steps don't resolve it, please reply and let us know — "
            "we're happy to help further."
        )
        return "\n".join(lines)

    @staticmethod
    def _first_meaningful_lines(body, max_chars=220):
        """Take the first real content lines of an article (skip headings)."""
        text = []
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Drop markdown list markers and bold markers for readability.
            line = re.sub(r"^[-*]\s+", "", line).replace("**", "")
            text.append(line)
            if sum(len(t) for t in text) >= max_chars:
                break
        joined = " ".join(text)
        return joined[:max_chars] + ("..." if len(joined) > max_chars else "")


def get_llm(mock=False):
    """Factory: returns a MockLLM in demo mode, otherwise a real LLMClient."""
    if mock:
        return MockLLM()
    return LLMClient()
