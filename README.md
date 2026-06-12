# chunk-wizard

A single-file Python utility for splitting text into token-budget-aware chunks before feeding it to a large language model.

No dependencies. No setup. Drop `chunk-wizard.py` into your project and import it.

---

## Why

LLMs have a fixed context window. If you feed them a document that is too large, the model either truncates it silently or errors out. The naive fix — split every N characters — cuts sentences in half and destroys structure.

`chunk-wizard` measures token budget, detects the natural structure of your text (headings, paragraphs, lines, sentences), and picks the strategy that keeps chunks as full as possible without ever exceeding the budget.

---

## How it works

### Token budget

```
char_budget = int(num_ctx * buffer_ratio) * chars_per_token
```

- `num_ctx` — your model's context window in tokens (e.g. 8192 for Llama 3)
- `buffer_ratio` — fraction reserved for reading input; the rest is for the model's reply (default 0.4)
- `chars_per_token` — estimated characters per token (default 4, works well for English prose and code)

### Strategy selection

Four splitting strategies are tried in order:

| Strategy  | Splits at                        | Best for               |
|-----------|----------------------------------|------------------------|
| heading   | Markdown `# headings`            | Markdown documentation |
| paragraph | Blank lines (`\n\n`)             | Articles, reports      |
| line      | Every newline                    | Logs, code, data files |
| sentence  | `.` `!` `?` boundaries          | Unstructured prose     |

Each strategy produces a set of chunks. The one with the highest **utilization score** wins:

```
utilization = sum(sizes of fitting chunks) / (total_chunks × char_budget)
```

A strategy that produces oversized chunks scores lower, so the algorithm naturally prefers strategies that actually fit the text within budget.

---

## Usage

### In-memory (recommended for LLM pipelines)

```python
from chunk_wizard import select_chunking_strategy

chunks = select_chunking_strategy(text, num_ctx=8192)

for chunk in chunks:
    response = your_llm_call(chunk)
```

Pass explicit parameters or let the function read from environment variables:

```bash
export OLLAMA_NUM_CTX=8192
export CHUNK_BUFFER_RATIO=0.4
```

```python
# reads env vars automatically when params are omitted
chunks = select_chunking_strategy(text)
```

Full signature:

```python
def select_chunking_strategy(
    text: str,
    num_ctx: Optional[int] = None,       # falls back to OLLAMA_NUM_CTX (default 8192)
    buffer_ratio: Optional[float] = None, # falls back to CHUNK_BUFFER_RATIO (default 0.4)
    chars_per_token: int = 4,
) -> List[str]:
```

Returns a `List[str]`. If `text` already fits within the budget, returns `[text]` unchanged.

---

### File-based (split a file into chunk files, reassemble later)

```python
from chunk_wizard import split_file_into_chunks, assemble_chunks

# Split
result = split_file_into_chunks(
    file_path="document.txt",
    output_dir="chunks/",
    num_ctx=4096,
)
# produces: chunks/document-1.txt, chunks/document-2.txt, ...

# Reassemble
result = assemble_chunks(
    input_dir="chunks/",
    output_dir="output/",
)
# produces: output/document.txt
```

Both functions return a dict:

```python
{
    "success": True,
    "data": { ... },   # chunk metadata or assembly summary
    "error": None,
    "metadata": { "operation": "...", "timestamp": "..." }
}
```

---

### Calculate chunk boundaries only (no I/O)

```python
from chunk_wizard import calculate_chunks

lines = open("document.txt").readlines()
boundaries = calculate_chunks(lines, num_ctx=4096)
# [(0, 142), (143, 287), ...]  — (start_line, end_line) inclusive
```

---

## Quick demo

```bash
python chunk-wizard.py
```

---

## Known limitations

- **Single unit larger than budget**: if one paragraph or heading block is larger than `char_budget`, it is passed through as an oversized chunk. The model sees it in full; no further splitting is attempted. A greedy word-level fallback is planned for a future version.
- **Token estimate is approximate**: `chars_per_token=4` works well for English. Code, JSON, and non-Latin scripts may need a lower value (e.g. 3).
- **Binary files not supported**: `split_file_into_chunks` reads text files only. PDFs and Word documents need to be converted to plain text first.

---

## Requirements

Python 3.8+. Standard library only (`os`, `re`, `shutil`, `pathlib`, `dataclasses`, `collections`, `typing`, `datetime`).

---

## License

MIT — see the license header in `chunk-wizard.py`.

---

## Author

Ali Cem Topcu  
GitHub: [@kilgor](https://github.com/kilgor)  
LinkedIn: [ali-cem-topcu](https://www.linkedin.com/in/ali-cem-topcu-b223a5a4/)
