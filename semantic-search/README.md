# Uzbek Legal QA Bot

Ask a question in Uzbek (Latin or Cyrillic) and get the best matching answer from a bank of 505 answers about government services and law. When the system is unsure it suggests similar questions; when the question is off-topic it says "I don't know".

**No LLMs and no transformer models are used to answer questions.** The search uses classic text matching, static word vectors and a few simple rules, so it runs on a normal CPU in a few milliseconds.

It comes with:
- a REST API (FastAPI) with Swagger docs;
- a Telegram bot: [@justiceQA_bot](https://t.me/justiceQA_bot);
- a PostgreSQL database for the answers, user feedback and query logs.

## Quick start

You need Docker, the data files `answers.csv` and `questions.csv` (not included in the repo), and a Telegram bot token.

```bash
cp answers.csv questions.csv data/
cp .env.example .env        # then put BOT_TOKEN=... in .env
make up                     # starts the database, the API and the bot
```

The first start takes about 2 minutes: it builds the image, downloads the embedding model and imports the data. When it is ready, open <http://localhost:8000/docs>.

- If port 8000 is busy, add `API_PORT=8010` to `.env`.
- To stop: `make down`. To also delete the database: `docker compose down -v`.

**Without Docker** (Python 3.11):

```bash
make install    # create .venv and install dependencies
make models     # download the embedding model (~530 MB)
make api        # API on http://localhost:8000/docs
make bot        # Telegram bot (in a second terminal)
```

## How to test

**Swagger (easiest):** open <http://localhost:8000/docs>, choose `POST /ask` → **Try it out**, pick one of the Uzbek examples and press **Execute**.

**curl:**

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "ID-karta qancha muddatga beriladi?"}'
```

The response has a `status`:
- `answer`: the system is confident, and the answer text is included;
- `suggest`: not sure, so the top 3 candidate questions are returned;
- `not_found`: off-topic, or the question cites a law that is not in the data.

**Telegram:** open [@justiceQA_bot](https://t.me/justiceQA_bot), press **Start** and ask a question. Answers have 👍 / 👎 buttons, and uncertain questions get "Did you mean…?" buttons.

**Automated tests:** `make test` runs 193 tests (no database needed). `make eval` recomputes all the accuracy numbers below.

Other endpoints: `GET /answers/{id}`, `GET /health`, `GET /stats`, `POST /feedback`, `POST /reload`.

## How it works

Every question goes through these steps:

1. **Clean the text.** Cyrillic is converted to Latin, all apostrophe variants (`‘ ’ ʻ`) become `'`, and the text is lowercased. So `Паспорт`, `pasport` and `PASPORT` are treated the same.
2. **Recognize key terms.** Law numbers (`PQ-4567`), article numbers (`15-modda`) and organization names (`DXM` = `Davlat xizmatlari markazi`) are turned into exact tokens, so similar numbers like `PQ-4567` and `PQ-4576` are never confused.
3. **Add synonyms.** Colloquial and Russian words get the official wording added, for example `prava` → `haydovchilik guvohnomasi` and `spravka` → `ma'lumotnoma`.
4. **Search.** Two methods look for the most similar stored questions:
   - character-level text matching, which tolerates typos and Uzbek word endings;
   - static word vectors (Model2Vec), which capture some meaning.
5. **Rank.** The results are combined and the top 20 are re-scored using text similarity, matching key terms and category.
6. **Decide.**
   - If the question mentions a law that is not in the data → **"I don't know"**. A wrong law is worse than no answer.
   - If the best score is high and clearly ahead of the second → **answer**.
   - If the score is medium → **suggest** the top 3.
   - Otherwise → **"I don't know"**.

The search index is kept in memory, so answering takes about 5–7 ms. PostgreSQL only stores the data, feedback and logs; the index is rebuilt from it at startup or with `POST /reload`.

## Results

The test questions were written by hand to look like real user questions (typos, Cyrillic, colloquial words, abbreviations), and they were never used for tuning.

| | Result |
|---|---|
| Correct answer ranked first | **67%** |
| Correct answer in the top 3 | **82%** |
| Direct answers that are correct | 86% |
| Off-topic questions → "I don't know" | 91% |
| Unknown law number → "I don't know" | 100% |
| Speed | 5–7 ms per question on CPU |

What helped most: character-level matching (better than word-based BM25 for Uzbek), synonyms for colloquial words, and the Model2Vec word vectors.

For comparison, the large transformer model **BGE-M3** reached 80% on the same questions. It understands rephrased questions better, but it is 35× slower on CPU, needs PyTorch and a 2.3 GB model, and transformers are not allowed by the task rules. It was used only as a benchmark.

## Limitations

- Questions that are worded very differently from the stored ones are the main weakness. They usually get suggestions rather than a direct answer.
- About 1 in 7 direct answers on realistic questions is wrong. Raising `ANSWER_THRESHOLD` in `.env` makes the bot more careful.
- Some answers in the bank are nearly identical (for example two "certificate duplicate" answers), which can cause mix-ups.
- The best way to improve: add real user questions from `query_logs` and 👎 feedback to the bank as extra wordings of existing answers.

## Project structure

```text
app/core/      search engine: text cleaning, term recognition, search, ranking
app/api/       FastAPI service
app/bot/       Telegram bot (talks to the API over HTTP)
app/db/        database models and storage for feedback and logs
app/eval/      evaluation code
scripts/       evaluation, data import, model download, benchmarks
data/          term lists, synonyms and test questions (the private data files go here too)
tests/         automated tests
```

All settings (thresholds, weights, paths) are in `app/core/config.py` and can be changed in `.env`.
