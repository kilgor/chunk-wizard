"""
chunk-wizard.py - Dynamic Token-Aware Text Chunking

VERSION: v0.2.0
LAST_UPDATED: 2026-06-12

Author: Ali Cem Topcu
GitHub: @kilgor
LinkedIn: https://www.linkedin.com/in/ali-cem-topcu-b223a5a4/

MIT License

Copyright (c) 2025 Ali Cem Topcu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

================================================================================

MODULE PURPOSE:
Split text into token-budget-aware chunks and reassemble them.

Two modes:
  - File-based : split_file_into_chunks / assemble_chunks
    Reads a text file, writes numbered chunk files, reassembles them back.
    Chunk boundaries calculated dynamically using character counting (~4 chars = 1 token).

  - In-memory  : select_chunking_strategy
    Takes a text string, auto-detects structure (headings / paragraphs / lines / sentences),
    returns a list of chunk strings sized to fit an LLM context window.
    Falls back to OLLAMA_NUM_CTX and CHUNK_BUFFER_RATIO env vars if params not provided.

DOMAIN: Text Chunking
Security: Path validation, encoding safety
"""

import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _success(operation: str, data: Any) -> Dict[str, Any]:
    return {
        "success": True,
        "data": data,
        "error": None,
        "metadata": {"operation": operation, "timestamp": datetime.now().isoformat()},
    }


def _error(operation: str, message: str) -> Dict[str, Any]:
    return {
        "success": False,
        "data": None,
        "error": message,
        "metadata": {"operation": operation, "timestamp": datetime.now().isoformat()},
    }


def _validate_path(path: str) -> Optional[str]:
    if not path or not path.strip():
        return "Path cannot be empty"
    return None


_CHUNK_PATTERN = re.compile(r"^(.+)-(\d+)(\.[^.]+)$")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ChunkConfig:
    """Configuration for chunk operations."""
    buffer_ratio: float = 0.4
    chars_per_token: int = 4
    default_encoding: str = "utf-8"


# =============================================================================
# MODULE METADATA
# =============================================================================

MODULE_METADATA = {
    "name": "chunk-wizard",
    "version": "v0.2.0",
    "description": "Dynamic token-aware text chunking and reassembly.",
    "operations": ["calculate_chunks", "split_file_into_chunks", "assemble_chunks", "select_chunking_strategy"],
}


# =============================================================================
# CHUNKING STRATEGIES (private helpers)
# =============================================================================

def _split_by_heading(text: str) -> List[str]:
    # splits before each heading line; \n consumed so heading has no leading newline
    return re.split(r"\n(?=#+\s)", text)


def _split_by_paragraph(text: str) -> List[str]:
    return re.split(r"\n{2,}", text)


def _split_by_line(text: str) -> List[str]:
    return text.splitlines()


def _split_by_sentence(text: str) -> List[str]:
    # lookbehind keeps punctuation with the sentence
    return re.split(r"(?<=[.!?])\s+", text)


def _group_units(units: List[str], char_budget: int, separator: str) -> List[str]:
    chunks: List[str] = []
    batch: List[str] = []
    current_len = 0
    sep_len = len(separator)

    for unit in units:
        unit_len = len(unit)
        if not batch:
            batch.append(unit)
            current_len = unit_len
        else:
            addition = sep_len + unit_len
            if current_len + addition > char_budget:
                raw = separator.join(batch)
                chunks.append(re.sub(f"({re.escape(separator)})+", separator, raw))
                batch = [unit]
                current_len = unit_len
            else:
                batch.append(unit)
                current_len += addition

    if batch:
        raw = separator.join(batch)
        chunks.append(re.sub(f"({re.escape(separator)})+", separator, raw))

    return chunks


# =============================================================================
# PUBLIC API
# =============================================================================

def calculate_chunks(
    lines: List[str],
    num_ctx: int,
    buffer_ratio: float = 0.4,
    chars_per_token: int = 4,
) -> List[Tuple[int, int]]:
    """
    Calculate chunk boundaries for a list of lines using token estimation.

    Formula:
        token_budget = int(num_ctx * buffer_ratio)
        line_tokens  = len(line) / chars_per_token
        Accumulate lines until next line would exceed token_budget.
        Boundary = last line that fits (inclusive). Overflowing line starts next chunk.

    Args:
        lines:           List of text lines (e.g. from file.readlines())
        num_ctx:         Model context window size in tokens
        buffer_ratio:    Fraction of context reserved for reading (default 0.4)
        chars_per_token: Estimated characters per token (default 4)

    Returns:
        List of (start, end) tuples -- 0-indexed, inclusive. No I/O.
    """
    token_budget = int(num_ctx * buffer_ratio)
    chunks: List[Tuple[int, int]] = []
    start = 0
    running_tokens = 0.0

    for i, line in enumerate(lines):
        line_tokens = len(line) / chars_per_token
        if running_tokens + line_tokens > token_budget:
            chunks.append((start, i - 1))
            start = i
            running_tokens = line_tokens
        else:
            running_tokens += line_tokens

    if start < len(lines):
        chunks.append((start, len(lines) - 1))

    return chunks


def split_file_into_chunks(
    file_path: str,
    output_dir: str,
    num_ctx: int,
    buffer_ratio: float = 0.4,
    chars_per_token: int = 4,
    encoding: str = "utf-8",
    config: Optional[ChunkConfig] = None,
) -> Dict[str, Any]:
    """
    Split a text file into token-budget-aware chunk files.

    Output filenames: {stem}-{n}{suffix}  e.g. document-1.txt, document-2.txt
    If the file fits in one chunk, it is copied as-is (no suffix added).

    Args:
        file_path:       Path to the source text file
        output_dir:      Directory where chunk files will be written
        num_ctx:         Model context window size in tokens
        buffer_ratio:    Fraction of context reserved for reading (default 0.4)
        chars_per_token: Estimated characters per token (default 4)
        encoding:        File encoding (default utf-8)
        config:          Optional ChunkConfig override

    Returns:
        dict with success, data (chunk metadata), error
    """
    err = _validate_path(file_path) or _validate_path(output_dir)
    if err:
        return _error("split_file_into_chunks", err)

    try:
        path = Path(file_path)
        if not path.exists():
            return _error("split_file_into_chunks", f"File not found: {file_path}")

        os.makedirs(output_dir, exist_ok=True)

        with open(file_path, encoding=encoding) as f:
            lines = f.readlines()

        chunks = calculate_chunks(lines, num_ctx, buffer_ratio, chars_per_token)
        chunk_meta = []

        if len(chunks) == 1:
            dest = Path(output_dir) / path.name
            shutil.copy(file_path, dest)
            chunk_meta.append({"chunk_id": 1, "start_line": 0, "end_line": len(lines) - 1,
                                "line_count": len(lines), "token_estimate": int(sum(len(l) / chars_per_token for l in lines)),
                                "output_file": path.name, "split": False})
        else:
            for n, (start, end) in enumerate(chunks, 1):
                chunk_lines = lines[start: end + 1]
                token_estimate = int(sum(len(l) / chars_per_token for l in chunk_lines))
                out_name = f"{path.stem}-{n}{path.suffix}"
                out_path = Path(output_dir) / out_name
                with open(out_path, "w", encoding=encoding) as f:
                    f.writelines(chunk_lines)
                chunk_meta.append({"chunk_id": n, "start_line": start, "end_line": end,
                                   "line_count": end - start + 1, "token_estimate": token_estimate,
                                   "output_file": out_name, "split": True})

        return _success("split_file_into_chunks", {
            "source_file": path.name,
            "total_lines": len(lines),
            "total_chunks": len(chunks),
            "token_budget": int(num_ctx * buffer_ratio),
            "chunks": chunk_meta,
        })

    except Exception as e:
        return _error("split_file_into_chunks", str(e))


def assemble_chunks(
    input_dir: str,
    output_dir: str,
    encoding: str = "utf-8",
    config: Optional[ChunkConfig] = None,
) -> Dict[str, Any]:
    """
    Reassemble chunk files produced by split_file_into_chunks back into originals.

    Detects chunks by pattern {stem}-{n}{suffix} where n is an integer.
    Chunks are sorted by n and concatenated in order.
    Files with no chunk number are copied as-is.

    Args:
        input_dir:  Directory containing chunk files
        output_dir: Directory where reassembled files will be written
        encoding:   File encoding (default utf-8)
        config:     Optional ChunkConfig override

    Returns:
        dict with success, data (assembly summary), error
    """
    err = _validate_path(input_dir) or _validate_path(output_dir)
    if err:
        return _error("assemble_chunks", err)

    try:
        if not Path(input_dir).exists():
            return _error("assemble_chunks", f"Input directory not found: {input_dir}")

        os.makedirs(output_dir, exist_ok=True)

        chunked: Dict[Tuple[str, str], List[Tuple[int, Path]]] = defaultdict(list)
        plain: List[Path] = []

        for path in sorted(Path(input_dir).glob("*")):
            if not path.is_file():
                continue
            m = _CHUNK_PATTERN.match(path.name)
            if m:
                stem, n, suffix = m.group(1), int(m.group(2)), m.group(3)
                chunked[(stem, suffix)].append((n, path))
            else:
                plain.append(path)

        assembled = []

        for path in plain:
            dest = Path(output_dir) / path.name
            shutil.copy(path, dest)
            assembled.append({"file": path.name, "chunks_merged": 0, "total_lines": None, "note": "copied as-is"})

        for (stem, suffix), parts in sorted(chunked.items()):
            parts.sort(key=lambda x: x[0])
            original_name = f"{stem}{suffix}"
            out_path = Path(output_dir) / original_name
            total_lines = 0
            with open(out_path, "w", encoding=encoding) as out:
                for _, chunk_path in parts:
                    with open(chunk_path, encoding=encoding) as f:
                        lines = f.readlines()
                    out.writelines(lines)
                    total_lines += len(lines)
            assembled.append({"file": original_name, "chunks_merged": len(parts), "total_lines": total_lines})

        return _success("assemble_chunks", {
            "files_assembled": len(assembled),
            "output_dir": output_dir,
            "results": assembled,
        })

    except Exception as e:
        return _error("assemble_chunks", str(e))


def select_chunking_strategy(
    text: str,
    num_ctx: Optional[int] = None,
    buffer_ratio: Optional[float] = None,
    chars_per_token: int = 4,
) -> List[str]:
    """
    Split text into token-budget-sized chunks using the best semantic strategy.

    Tries 4 strategies (heading, paragraph, line, sentence), scores each by
    utilization (avg fitting chunk size / char_budget), returns chunks from the
    highest-scoring strategy. If text already fits in one chunk, returns [text].

    num_ctx and buffer_ratio fall back to env vars if not provided:
        OLLAMA_NUM_CTX     (default 8192)
        CHUNK_BUFFER_RATIO (default 0.4)

    Strategy selection:
        heading   -- splits at markdown # headings; heading line attached to content
        paragraph -- splits at blank lines
        line      -- splits at each newline
        sentence  -- splits at .!? boundaries (punctuation stays with sentence)

    Scoring: utilization = sum(fitting_chunk_sizes) / (total_chunks * char_budget)
    Oversized chunks (single unit > char_budget) count as 0 in the numerator,
    penalizing strategies that cannot split the text.

    Known limitation: if a single unit is larger than char_budget it becomes its
    own oversized chunk passed through as-is (greedy fallback planned for v0.3.0).

    Args:
        text:            Full text to split
        num_ctx:         Model context window size in tokens (falls back to OLLAMA_NUM_CTX)
        buffer_ratio:    Fraction of context for input (falls back to CHUNK_BUFFER_RATIO)
        chars_per_token: Estimated characters per token (default 4)

    Returns:
        List[str] -- chunk strings sized to fit within char_budget
    """
    num_ctx = num_ctx or int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    buffer_ratio = buffer_ratio or float(os.getenv("CHUNK_BUFFER_RATIO", "0.4"))
    char_budget = int(num_ctx * buffer_ratio) * chars_per_token

    if char_budget <= 0 or len(text) <= char_budget:
        return [text]

    STRATEGIES = [
        ("heading",   _split_by_heading,   "\n"),
        ("paragraph", _split_by_paragraph, "\n\n"),
        ("line",      _split_by_line,      "\n"),
        ("sentence",  _split_by_sentence,  " "),
    ]

    best_chunks: Optional[List[str]] = None
    best_utilization = -1.0

    for _name, split_fn, separator in STRATEGIES:
        units = split_fn(text)
        chunks = _group_units(units, char_budget, separator)
        if not chunks:
            continue
        fitting = [len(c) for c in chunks if len(c) <= char_budget]
        utilization = sum(fitting) / (len(chunks) * char_budget)
        if utilization > best_utilization:
            best_utilization = utilization
            best_chunks = chunks

    return best_chunks if best_chunks is not None else [text]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ChunkConfig",
    "MODULE_METADATA",
    "calculate_chunks",
    "split_file_into_chunks",
    "assemble_chunks",
    "select_chunking_strategy",
]


# =============================================================================
# QUICK DEMO
# =============================================================================

if __name__ == "__main__":
    print("=== chunk-wizard v0.2.0 demo ===\n")

    # calculate_chunks demo
    lines = [f"Line {i}: {'x' * 80}\n" for i in range(200)]
    chunks = calculate_chunks(lines, num_ctx=4096, buffer_ratio=0.4)
    print(f"calculate_chunks: 200 lines @ 4096 ctx -> {len(chunks)} chunks")
    for i, (s, e) in enumerate(chunks, 1):
        tokens = int(sum(len(lines[j]) / 4 for j in range(s, e + 1)))
        print(f"  chunk {i}: lines {s}-{e} ({e - s + 1} lines, ~{tokens} tokens)")

    # select_chunking_strategy demo
    print("\nselect_chunking_strategy demo:")
    sample = "# Introduction\nThis is the intro.\n\n# Section 1\nContent for section one.\n\n# Section 2\nContent for section two."
    result = select_chunking_strategy(sample, num_ctx=256)
    print(f"  input: {len(sample)} chars -> {len(result)} chunk(s)")
    for i, c in enumerate(result, 1):
        print(f"  chunk {i} ({len(c)} chars): {c[:60]!r}...")

    print("\nDemo complete.")
