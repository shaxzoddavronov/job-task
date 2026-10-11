# Uzbek Legal QA Bot

Ask a question in Uzbek (Latin or Cyrillic) and get the best matching answer from a bank of 505 answers about government services and law. When the system is unsure it suggests similar questions; when the question is off-topic it says "I don't know".

The bot **does not write answers**: it picks the best one from the fixed answer bank. To understand the question it uses the transformer model **BGE-M3** and, on a GPU, a second model that re-checks the best candidates (**bge-reranker-v2-m3**). No LLM / chat model is used.

It comes with:
- a REST API (FastAPI) with Swagger docs;
- a Telegram bot: [@justiceQA_bot](https://t.me/justiceQA_bot);
- a PostgreSQL database for the answers, user feedback and query logs.

## Quick start

You need Docker, the data files `answers.csv` and `questions.csv` (not included in the repo), and a Telegram bot token.

```bash
cp answers.csv questions.csv data/
cp .env.example .env        # then put BOT_TOKEN=... in .env
./start.sh                  # Linux / macOS   (or: make up)
.\start.bat                 # Windows
```

The launcher checks whether Docker can use an NVIDIA GPU and starts in **GPU mode** if it can, otherwise in **CPU mode**. It prints which one it chose.

The first start takes a while: it builds the image (~7.5 GB, includes PyTorch with CUDA) and downloads the two models (~4.5 GB) into a Docker volume. Later starts reuse both and are ready in under a minute. When it is ready, open <http://localhost:8000/docs>.

- If port 8000 is busy, add `API_PORT=8010` to `.env`.
- To stop: `make down` (or `docker compose down`). To also delete the database: `docker compose down -v`.
- GPU on Windows needs Docker Desktop with the WSL2 backend and a current NVIDIA driver.

**Without Docker** (Python 3.11):

```bash
make install    # create .venv and install dependencies
make models     # optional: download the models now (~4.5 GB)
make api        # API on http://localhost:8000/docs
make bot        # Telegram bot (in a second terminal)
```

On Windows with an NVIDIA GPU, install the CUDA build of PyTorch first: `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126` (the default Windows build is CPU-only).

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

**Automated tests:** `make test` runs the unit tests (no GPU, no model download). `make eval` recomputes the accuracy numbers below, and `make latency` measures the speed on your machine.

Other endpoints: `GET /answers/{id}`, `GET /health` (shows `device` and `mode`), `GET /stats`, `POST /feedback`, `POST /reload`.

## How it works

At startup the API loads the models and turns every stored question and answer into a vector (a list of numbers that captures its meaning). The vectors are cached on disk, so later starts are fast.

Every question then goes through these steps:

1. **Clean the text.** Cyrillic is converted to Latin and all apostrophe variants (`‘ ’ ʻ`) become `'`.
2. **Recognize key terms.** Law numbers (`PQ-4567`), article numbers (`15-modda`) and organization names (`DXM` = `Davlat xizmatlari markazi`) are detected.
3. **Safety rule.** If the question mentions a law that is not in the data, the answer is **"I don't know"**. A wrong law is worse than no answer.
4. **Add synonyms.** Colloquial and Russian words get the official wording added, for example `prava` → `haydovchilik guvohnomasi`.
5. **Find candidates.** BGE-M3 turns the question into a vector and finds the stored questions and answers with the closest meaning.
6. **Re-check (GPU mode).** The reranker reads the question together with each of the 10 best candidates and scores how well they fit. Its score is combined with the BGE-M3 score.
7. **Decide.**
   - Score high and clearly ahead of the second → **answer**.
   - Score medium → **suggest** the top 3.
   - Otherwise → **"I don't know"**.

**GPU mode vs CPU mode.** With a GPU, both models run (~50 ms per question). Without a GPU, only BGE-M3 runs (~0.2 s per question), because the reranker would take seconds on a CPU. Each mode has its own calibrated thresholds.

PostgreSQL only stores the data, feedback and logs; the search index lives in memory and is rebuilt from the database at startup or with `POST /reload`.

## Results

The test questions were written by hand to look like real user questions (typos, Cyrillic, colloquial words, abbreviations, rephrasings), and they were never used for tuning.

| | GPU mode | CPU mode | Previous engine (no transformers) |
|---|---|---|---|
| Correct answer ranked first | **92.5%** | 85.0% | 67.3% |
| Correct answer in the top 3 | **96.3%** | 93.5% | 82.2% |
| Direct answers that are correct | **96.5%** | 95.2% | 86.4% |
| Off-topic questions → "I don't know" | 100% | 100% | 91.3% |
| Unknown law number → "I don't know" | 100% | 100% | 100% |
| Speed per question | ~50 ms (GPU) | ~0.2 s (CPU) | ~7 ms (CPU) |

On questions held out from your own data (which are close paraphrases of the stored ones), all three score about 99%.

The biggest gain is on rephrased questions: 88% correct first with the transformer engine, against 24% with the previous character-matching engine.

## Limitations

- About 1 in 30 direct answers on realistic questions is wrong. Raising `ANSWER_THRESHOLD_RERANK` (GPU) or `ANSWER_THRESHOLD_DENSE` (CPU) in `.env` makes the bot more careful.
- CPU mode is slower (~0.2 s per question) and a little less accurate, because the reranker only runs on a GPU.
- The first start needs to download ~4.5 GB of models.
- Some answers in the bank are nearly identical (for example two "certificate duplicate" answers), which can cause mix-ups.
- The best way to improve further: add real user questions from `query_logs` and 👎 feedback to the bank as extra wordings of existing answers.

## Project structure

```text
app/core/      search engine: text cleaning, term recognition, models, retrieval, ranking
app/api/       FastAPI service
app/bot/       Telegram bot (talks to the API over HTTP)
app/db/        database models and storage for feedback and logs
app/eval/      evaluation and tuning code
scripts/       evaluation, data import, model download, latency check
data/          term lists, synonyms and test questions (the private data files go here too)
tests/         automated tests
```

All settings (device, models, thresholds, weights, paths) are in `app/core/config.py` and can be changed in `.env`.
