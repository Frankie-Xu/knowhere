"""Shared hit rendering for every ``corpus.*`` tool.

``build_snippet`` windows body text around the first match: head + first-match
window + tail, ``...``-joined, overlap-merged. Only the first match is
windowed.

``format_row`` renders the one model-visible row shape every tool shares —
outline/node_filter (map rows: indented, ``summary``, ``chunk_count``),
grep/recall (hit rows: flat, ``snippet``, ``score``), and assets (asset rows:
``chunk_id`` plus the hosting ``section_path``) all call this instead of
building their own line format, so a model reads one row shape regardless of
which tool produced it and copies the same fields (``document_id`` +
``section_path``/``chunk_id``) into ``corpus.read``.
"""

from __future__ import annotations

HIT_CONTEXT_CHARS = 80
HEAD_TAIL_CHARS = 50


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(s for s in spans if s[1] > s[0])
    merged: list[list[int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def build_snippet(
    text: str,
    hit: tuple[int, int] | None = None,
    *,
    hit_context: int = HIT_CONTEXT_CHARS,
    head_tail: int = HEAD_TAIL_CHARS,
) -> str:
    """Head/tail anchor + first-match window, joined by ``...`` where spans don't touch.

    ``hit`` is the ``(start, end)`` char offset of the located match in
    ``text``, or ``None`` when no specific position is known (falls back to
    head/tail anchors only). Short text (<= ``head_tail * 2`` chars) is
    returned unchanged.
    """
    if not text:
        return ""
    if len(text) <= head_tail * 2:
        return text

    spans: list[tuple[int, int]] = [
        (0, head_tail),
        (max(len(text) - head_tail, 0), len(text)),
    ]
    if hit is not None:
        hit_start, hit_end = hit
        spans.append(
            (max(hit_start - hit_context, 0), min(hit_end + hit_context, len(text)))
        )

    merged = _merge_spans(spans)
    parts: list[str] = []
    prev_end = 0
    for start, end in merged:
        if start > prev_end:
            parts.append("...")
        parts.append(text[start:end])
        prev_end = end
    if prev_end < len(text):
        parts.append("...")
    return "".join(parts)


def format_row(
    *,
    kind: str,
    document_id: object,
    section_path: object,
    title: object = "",
    chunk_id: object | None = None,
    chunk_count: int | None = None,
    summary: str = "",
    snippet: str = "",
    score: object | None = None,
    depth: int = 0,
    is_hit: bool = False,
) -> str:
    """Render one row. Every ``corpus.*`` search/map tool shares this shape.

    ``kind`` is ``section`` (outline/node_filter node), ``text``/``page``
    (body hit), or ``image``/``table`` (asset — its own body chunk or a hit
    on one). ``chunk_id`` is included for ``image``/``table`` rows (their
    stored ``section_path`` is the document Root — this field then names the
    *hosting* section instead, or stays ``Root`` when no host was found).
    ``chunk_count``/``summary`` are map-row fields; ``snippet``/``score`` are
    hit-row fields; ``depth`` indents a map row under its parent; ``is_hit``
    marks a row lit up by node_filter/lighting scoring.
    """
    indent = "  " * max(depth, 0)
    label = str(title or "").strip()
    header = f"{indent}- [{kind}]"
    if label:
        header += f" {label}"
    header += f" | document_id={document_id} section_path={section_path}"
    if chunk_id:
        header += f" chunk_id={chunk_id}"
    if chunk_count is not None:
        header += f" (chunks={chunk_count})"
    if score is not None:
        header += f" score={score}"
    if is_hit:
        header += " [Hit]"
    lines = [header]
    if summary:
        lines.append(f"{indent}  summary: {summary}")
    if snippet:
        lines.append(f"{indent}  snippet: {snippet!r}")
    return "\n".join(lines)
