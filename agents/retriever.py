"""
agents/retriever.py — Agent 2: find the right help articles.

Job: given the ticket text, search the local knowledge base and return the
3 most relevant help articles. This is RAG (Retrieval-Augmented
Generation): instead of letting the LLM answer from memory, we first
RETRIEVE the facts, then let the LLM answer using only those facts.

How it works: BM25, a classic keyword search algorithm, written in pure
Python (no extra libraries). For each word in the ticket, BM25 scores
every article by:
  * how often the word appears in the article (term frequency),
  * how rare the word is across all articles (IDF — rare words count more),
  * and it corrects for article length, so long articles don't win
    just for being long.
Common stopwords ("the", "I", "please") are removed from the query
first, so the ranking reflects the actual topic words.

The math (one line each, nothing scary):
  IDF(word)      = ln(1 + (N - docs_with_word + 0.5) / (docs_with_word + 0.5))
  score(article) = sum over query words of IDF * (tf * (k1+1)) / (tf + k1*(1 - b + b*len/doc_avg_len))
  with k1 = 1.5 and b = 0.75 (the standard BM25 settings).

This is the "ACT" step of the agent loop: the agent uses a TOOL
(the search index) to gather information before answering.
"""

import math
import os
import re


def tokenize(text):
    """Split text into lowercase words. Simple and good enough for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def stem_word(word):
    """Light stemming: strip a common suffix so word forms match.

    "ordered" and "order", "arrived" and "arrival", "charges" and
    "charge" should count as the same word — without this, a customer
    who writes "my package never arrived" scores far below one who
    writes "my package never arrives", even though they mean the same
    thing. Applied to both the articles and the query, so both sides
    always agree. Deliberately crude (no dictionary, no exceptions):
    this is a demo search, not a linguistics project.
    """
    if len(word) > 4 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 3 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


# Relevance threshold: a top score below this means "no article really
# answers this ticket", even if a word or two happens to match.
# Measured on 2026-11-26 (with stemming on): all 8 real sample
# tickets score 15.49–36.45, while off-topic questions score at most
# 9.18 (the Bitcoin ticket's "pay"/"order" words weakly match Payment
# Failed) and 2.40 for the short probes. 10.0 sits cleanly between
# the two, so the desk says "I don't know" instead of answering from
# a barely-related article. Re-measure if the knowledge base changes.
# Honest limit: a short, heavily paraphrased question can score below
# 10.0 even when it is answerable ("my package never arrived" scores
# 9.51) — then the desk says "I don't know" and escalates, which is
# the safe failure, never a wrong answer. Fixing that properly needs
# semantic (embedding) search, not a better threshold: no single
# number separates short paraphrases from off-topic questions.
MIN_SCORE = 10.0


# Common English words carry almost no meaning for search ("the", "I",
# "please", ...). Removing them from the QUERY stops them adding noise
# to the scores, so the ranking reflects the actual topic words.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "so", "is", "are", "was",
    "were", "be", "been", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "it", "its", "they", "them", "their", "this", "that",
    "these", "those", "am", "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "should", "please", "hi", "hello",
    "thanks", "thank", "of", "to", "in", "on", "at", "for", "with",
    "from", "by", "as", "not", "no", "just", "very", "really", "get",
    "got", "all", "any", "when", "where", "how", "what", "there", "here",
}


class BM25Retriever:
    """A small BM25 search index over a folder of markdown articles."""

    def __init__(self, knowledge_dir):
        self.docs = []  # each doc: {"title": ..., "path": ..., "text": ...}
        for filename in sorted(os.listdir(knowledge_dir)):
            if not filename.endswith(".md"):
                continue
            path = os.path.join(knowledge_dir, filename)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            # The article title is the first "# ..." heading.
            title = next(
                (line[2:].strip() for line in text.splitlines() if line.startswith("# ")),
                filename,
            )
            self.docs.append({"title": title, "path": path, "text": text})

        # Pre-compute term frequencies and document lengths once.
        # Words are stemmed here (and in search) so "ordered" matches
        # "order" — see stem_word. Stopwords are NOT stemmed: they are
        # filtered from the query on their raw form, before stemming.
        self.doc_tokens = [
            [stem_word(w) for w in tokenize(doc["text"])] for doc in self.docs
        ]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

        # Document frequency: in how many docs does each word appear?
        self.doc_freq = {}
        for tokens in self.doc_tokens:
            for word in set(tokens):
                self.doc_freq[word] = self.doc_freq.get(word, 0) + 1

        self.num_docs = len(self.docs)

    def _idf(self, word):
        """Inverse document frequency — rare words are more informative."""
        docs_with_word = self.doc_freq.get(word, 0)
        return math.log(1 + (self.num_docs - docs_with_word + 0.5) / (docs_with_word + 0.5))

    def _score(self, query_words, doc_index):
        """BM25 score of one document for the query."""
        k1, b = 1.5, 0.75
        tokens = self.doc_tokens[doc_index]
        length = self.doc_lengths[doc_index]
        score = 0.0
        for word in query_words:
            tf = tokens.count(word)  # term frequency in this doc
            if tf == 0:
                continue
            idf = self._idf(word)
            # Length normalisation: penalise long docs, reward short ones.
            norm = tf + k1 * (1 - b + b * length / self.avg_length)
            score += idf * (tf * (k1 + 1)) / norm
        return score

    def search(self, query, top_k=3):
        """Return the top_k most relevant docs as (doc, score) pairs.

        Returns an empty list when nothing matches well enough: either
        no word overlaps at all, or the best score is below MIN_SCORE
        (a weak, coincidental match). The responder then knows to say
        "I don't know" instead of guessing from the wrong article.
        """
        # Drop stopwords so the ranking reflects topic words, not "the"/"I",
        # then stem so word forms match ("ordered" -> "order").
        query_words = [stem_word(w) for w in tokenize(query)
                       if w not in STOPWORDS]
        scored = [
            (doc, self._score(query_words, i)) for i, doc in enumerate(self.docs)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            (doc, score)
            for doc, score in scored[:top_k]
            if score >= MIN_SCORE
        ]
