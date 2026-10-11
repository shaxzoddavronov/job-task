# job-task

Two independent take-home tasks on Uzbek-language documents. Each folder is self-contained, with its own README, setup, tests and Docker configuration. Start with the README inside the folder.

| Task | Folder | In short |
|---|---|---|
| **1. Uzbek legal QA bot** (semantic search) | [`semantic-search/`](semantic-search/) | Answers questions about government services and law from a fixed bank of 505 answers, **without LLMs or transformers at inference** |
| **2. Uzbek document analysis** (entity extraction) | [`entity-extraction/`](entity-extraction/) | Reads an official PDF and extracts its details plus every task it assigns, with deadline and responsible party, using a local LLM |

## 1. Uzbek legal QA bot — [`semantic-search/`](semantic-search/)

A user asks a question in Uzbek (Latin or Cyrillic, with typos, colloquial or Russian words, abbreviations). The system returns the best matching answer, offers "Did you mean…?" suggestions when unsure, and says "I don't know" for off-topic questions or for laws that are not in the data.

- **Method:** script and apostrophe normalization, rule-based entity tagging (law numbers, articles, organizations), character n-gram TF-IDF plus Model2Vec static embeddings, rank fusion and a small linear re-ranker. Thresholds are calibrated for about 95% precision when answering.
- **Delivered as:** a FastAPI service with Swagger, a Telegram bot (aiogram 3) and PostgreSQL for data, feedback and query logs. One `docker compose up` starts all three.
- **Measured:** 67% Top-1 / 82% Top-3 on hand-written realistic questions, 99.6% on held-out paraphrases, 91% of off-topic questions rejected, 100% of invented law numbers rejected, 5–7 ms per query on CPU. It is compared with BGE-M3 as a benchmark-only reference.
- **Run:** `cd semantic-search && make up`. It needs the private data files and a bot token; see its README.

## 2. Uzbek document analysis — [`entity-extraction/`](entity-extraction/)

Upload a PDF and press **Analyze** to get the document's details (number, date, organization, signatory) and every task it assigns, each with its deadline and who is responsible.

- **Method:** text extraction with Docling and Tesseract OCR for scanned pages, Cyrillic → Latin transliteration with an offset map back to the original text, regex for numbers and dates, and a local LLM (served by Ollama) for organization, signatory and per-clause tasks. Every extracted value is grounded, i.e. checked against the source text, and deadlines are parsed into dates.
- **Works with** Uzbek documents in Latin or Cyrillic, typed or scanned. Everything runs locally, and no document leaves the machine. The model is chosen automatically from the available GPU, or runs on CPU.
- **Run:** `cd entity-extraction && ./start.sh` (Windows: `.\start.bat`), then open the address it prints.
