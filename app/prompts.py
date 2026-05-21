from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are an NLP extraction engine. Return compact strict JSON only. Do not include Markdown."""


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def prompt_step1(dialogue: str, doc_id: str) -> str:
    return f"""Task: identify all target-aspect pairs in the dialogue.

Definitions:
- target: entity or object being discussed.
- aspect: attribute, component, or concern of the target.

Return JSON with this schema:
{{
  "doc_id": "{doc_id}",
  "target_aspects": [
    {{"target": "...", "aspect": "..."}}
  ]
}}

Dialogue:
{dialogue}
"""


def prompt_step2(dialogue: str, doc_id: str, target_aspects: list[dict[str, str]]) -> str:
    return f"""Task: detect the speaker and opinion expression for each target-aspect pair.

Return JSON with this schema:
{{
  "doc_id": "{doc_id}",
  "quadruples": [
    {{"speaker": "...", "target": "...", "aspect": "...", "opinion": "..."}}
  ]
}}

Target-aspect pairs:
{_json(target_aspects)}

Dialogue:
{dialogue}
"""


def prompt_step3(dialogue: str, doc_id: str, quadruples: list[dict[str, str]]) -> str:
    return f"""Task: infer sentiment polarity and text rationale for each quadruple.

Allowed sentiment labels: positive, negative, neutral, mixed.
Keep rationale short: no more than 12 English words or one short Chinese phrase.
Return JSON with this schema:
{{
  "doc_id": "{doc_id}",
  "sextuples": [
    {{
      "doc_id": "{doc_id}",
      "speaker": "...",
      "target": "...",
      "aspect": "...",
      "opinion": "...",
      "sentiment": "positive|negative|neutral|mixed",
      "rationale": "...",
      "confidence": 0.0,
      "sentiment_flip": "None"
    }}
  ]
}}

Quadruples:
{_json(quadruples)}

Dialogue:
{dialogue}
"""


def prompt_step4(dialogue: str, doc_id: str, sextuples: list[dict[str, Any]]) -> str:
    return f"""Task: detect sentiment flips in a multi-turn dialogue.

A sentiment flip means the same speaker changes sentiment toward the same target/aspect after new information.
For a single-turn review, a one-speaker product comment, or text without an explicit later attitude change, return an empty list.
Return JSON with this schema:
{{
  "doc_id": "{doc_id}",
  "sentiment_flips": [
    {{
      "speaker": "...",
      "target": "...",
      "aspect": "...",
      "from_sentiment": "positive|negative|neutral|mixed",
      "to_sentiment": "positive|negative|neutral|mixed",
      "trigger": "..."
    }}
  ]
}}

Return an empty list when no flip is present.

Sextuples:
{_json(sextuples)}

Dialogue:
{dialogue}
"""


def prompt_one_shot(dialogue: str, doc_id: str) -> str:
    return f"""Task: extract sentiment sextuples and sentiment flips from the dialogue in one pass.

Extract only explicitly expressed opinions. Keep each rationale within 12 English words or one short Chinese phrase.
For a single-turn review, a one-speaker product comment, or text without an explicit later attitude change, set "sentiment_flips" to [].
Return JSON with this schema:
{{
  "doc_id": "{doc_id}",
  "sextuples": [
    {{
      "doc_id": "{doc_id}",
      "speaker": "...",
      "target": "...",
      "aspect": "...",
      "opinion": "...",
      "sentiment": "positive|negative|neutral|mixed",
      "rationale": "...",
      "confidence": 0.0,
      "sentiment_flip": "None"
    }}
  ],
  "sentiment_flips": [
    {{
      "speaker": "...",
      "target": "...",
      "aspect": "...",
      "from_sentiment": "...",
      "to_sentiment": "...",
      "trigger": "..."
    }}
  ]
}}

Dialogue:
{dialogue}
"""
