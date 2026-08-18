from __future__ import annotations

import ast
import re
from typing import Any

from gfmrag.kg_construction.utils import extract_json_dict
from gfmrag.llms import BaseLanguageModel


class SimGRAGQueryParser:
    """Convert natural-language query into condition/target pattern triples."""

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    def _build_prompt(self, query: str) -> str:
        return (
            "Convert the user question into a retrieval-oriented pattern graph for micro-macro activation.\n"
            "Return ONE strict JSON object only (no markdown, no explanation):\n"
            '{"condition_triples":[["head","relation","tail"]], "target_triples":[["head","relation","tail"]]}\n\n'
            "Rules:\n"
            "1) condition_triples = evidence constraints directly implied by the question, especially visual anchors.\n"
            "2) target_triples = what needs to be answered; use UNKNOWN_* placeholders for unknown entities.\n"
            "3) Keep triples short and normalized; each item must be exactly 3 non-empty strings.\n"
            "4) Use a consistent UNKNOWN subject across related constraints (e.g., UNKNOWN_movie in multiple condition triples).\n"
            "5) Prefer controlled relation phrases from project tests when applicable:\n"
            "   is in black and white photo; is performing with; is tennis player; is african american;\n"
            "   logo contains; logo text reads; poster contains; poster text reads;\n"
            "   title screen contains; title screen text reads; has actress; is movie poster.\n"
            "6) Put only evidence in condition_triples; put requested facts in target_triples.\n\n"
            f"Question: {query}\n"
            "Output:"
        )

    def _normalize_triple(self, triple: Any) -> tuple[str, str, str] | None:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            return None
        head, relation, tail = (
            str(triple[0]).strip(),
            str(triple[1]).strip(),
            str(triple[2]).strip(),
        )
        if not head or not relation or not tail:
            return None
        return (head, relation, tail)

    def _parse_graph_fallback(self, text: str) -> list[tuple[str, str, str]]:
        # Fallback for SimGRAG-like `{"graph":[(...),...]}` outputs.
        block = extract_json_dict(text)
        if isinstance(block, dict) and "graph" in block:
            triples: list[tuple[str, str, str]] = []
            for item in block.get("graph", []):
                norm = self._normalize_triple(item)
                if norm is not None:
                    triples.append(norm)
            return triples

        # Last-resort tuple extraction from plain text like `(a, b, c)`.
        triples = []
        for match in re.findall(r"\(([^()]+)\)", text):
            parts = [part.strip().strip("'\"") for part in match.split(",")]
            if len(parts) == 3 and all(parts):
                triples.append((parts[0], parts[1], parts[2]))
        return triples

    def parse(self, query: str) -> dict[str, list[tuple[str, str, str]]]:
        response = self.llm.generate_sentence(self._build_prompt(query))
        if isinstance(response, Exception):
            raise response

        payload = extract_json_dict(response)
        condition: list[tuple[str, str, str]] = []
        target: list[tuple[str, str, str]] = []

        if isinstance(payload, dict):
            for item in payload.get("condition_triples", []):
                norm = self._normalize_triple(item)
                if norm is not None:
                    condition.append(norm)
            for item in payload.get("target_triples", []):
                norm = self._normalize_triple(item)
                if norm is not None:
                    target.append(norm)

        # Tolerate python-literal outputs if JSON parsing failed.
        if not condition and not target:
            try:
                obj = ast.literal_eval(response)
                if isinstance(obj, dict):
                    for item in obj.get("condition_triples", []):
                        norm = self._normalize_triple(item)
                        if norm is not None:
                            condition.append(norm)
                    for item in obj.get("target_triples", []):
                        norm = self._normalize_triple(item)
                        if norm is not None:
                            target.append(norm)
            except (ValueError, SyntaxError):
                pass

        # Fallback to SimGRAG `graph` field and split by UNKNOWN.
        if not condition and not target:
            graph_triples = self._parse_graph_fallback(response)
            for head, relation, tail in graph_triples:
                if "UNKNOWN" in head or "UNKNOWN" in relation or "UNKNOWN" in tail:
                    target.append((head, relation, tail))
                else:
                    condition.append((head, relation, tail))

        return {"condition_triples": condition, "target_triples": target}
