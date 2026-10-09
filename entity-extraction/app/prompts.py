"""System prompts, few-shot examples and JSON schemas for the LLM calls.

Plan section 7.1.

Every constant here is module level and byte-stable, so the prompt prefix is
identical on every request and Ollama can reuse its cached KV for it. The
variable part — the chunk — is always the last message.

Instructions are in English; the few-shot examples are in Uzbek (Latin). The
examples are invented in the style of lex.uz acts: nothing here is copied from
a document that the evaluation scores, which would inflate the numbers.
"""

# --- Call B: one clause or table row ---------------------------------------

CLAUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_task": {"type": "boolean"},
        "deadline": {"type": ["string", "null"]},
        "executors": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["is_task", "deadline", "executors"],
}

CLAUSE_SYSTEM = """You read one clause from an official Uzbek document and report whether it assigns a task.

Return JSON with exactly these fields:
- is_task: true if the clause assigns work to someone; false otherwise.
- deadline: the time limit, copied exactly as written, or null if none is given.
- executors: the organisations or officials made responsible, copied exactly as written. Empty list if none is named.

Rules:
- Copy values exactly as they appear in the text. Never translate, reword, normalise or invent.
- A clause assigns work when it tells someone to do something — usually a verb ending in -sin/-sun (taʼminlasin, kiritsin, koʻrsin, oʻtkazilsin, tashkil etilsin).
- A clause that only states a fact, defines a status or says what something "hisoblanadi" is NOT a task.
- A clause saying a document comes into force is NOT a task.
- A clause that only introduces a list of sub-items, ending with a colon, is NOT itself a task — its sub-items are the tasks.
- Assigning oversight or control of execution to someone ("ijrosini nazorat qilish ... zimmasiga yuklatilsin") IS a task, even though it carries no deadline.
- Name as executors only those made responsible for doing the work, never the body the work is submitted to, and never an organisation that is merely the object being transferred or renamed.
- Deadlines may be absolute (1-noyabrgacha, 15.12.2026 gacha), relative (ikki oy muddatda, bir hafta muddatda), recurring (har oyda, har chorakda) or open-ended (doimiy).
- If the Context line names the responsible body and the clause itself does not, use the one from Context.
- If the Context shows the clause is one item in a list of definitions, areas of responsibility or powers — the parent ends in "hisoblanadi", "Belgilab qoʻyilsinki" or similar — then the item is NOT a task, however it is worded.
- If a value is not present, return null or an empty list. Do not guess."""

#: ``(user, assistant)`` pairs. Invented examples in lex.uz style.
CLAUSE_FEWSHOT = [
    (
        "Context:\nClause: 3. Moliya vazirligi 1-dekabrgacha byudjet mablagʻlari sarflanishi "
        "boʻyicha hisobotni Vazirlar Mahkamasiga taqdim etsin.",
        '{"is_task": true, "deadline": "1-dekabrgacha", "executors": ["Moliya vazirligi"]}',
    ),
    (
        "Context:\nClause: 4. Adliya vazirligi manfaatdor idoralar bilan birgalikda uch oy "
        "muddatda qonunchilik hujjatlariga oʻzgartirishlar toʻgʻrisida takliflar kiritsin.",
        '{"is_task": true, "deadline": "uch oy muddatda", "executors": '
        '["Adliya vazirligi", "manfaatdor idoralar"]}',
    ),
    (
        "Context: 2. Raqamli texnologiyalar vazirligi quyidagilarni taʼminlasin:\n"
        "Clause: b) elektron platformaga integratsiya har chorakda oʻtkazilsin;",
        '{"is_task": true, "deadline": "har chorakda", "executors": '
        '["Raqamli texnologiyalar vazirligi"]}',
    ),
    (
        "Context:\nClause: 2. Belgilab qoʻyilsinki, Ekologiya vazirligi atrof-muhitni muhofaza "
        "qilish sohasida respublika ijro etuvchi hokimiyat organi hisoblanadi.",
        '{"is_task": false, "deadline": null, "executors": []}',
    ),
    (
        "Context:\nClause: 1. Moliya vazirligi quyidagi vazifalarni bajarsin:",
        '{"is_task": false, "deadline": null, "executors": []}',
    ),
    (
        "Context: (a) quyidagi sohalarda yagona davlat siyosatini amalga oshirish boʻyicha "
        "respublika ijro etuvchi hokimiyat organi hisoblanadi:\n"
        "Clause: (iii) kasbiy taʼlim;",
        '{"is_task": false, "deadline": null, "executors": []}',
    ),
]


def clause_messages(text: str, context: str = "") -> list[dict]:
    """System + few-shot + the chunk. The chunk is always last."""
    messages = [{"role": "system", "content": CLAUSE_SYSTEM}]
    for user, assistant in CLAUSE_FEWSHOT:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": f"Context: {context}\nClause: {text}"})
    return messages


# --- Call B, batched: several clauses in one call --------------------------
#
# One call per clause re-sends the whole prompt prefix every time: 47 000 prompt
# tokens for a 5-page act. Packing several clauses into one call cuts that to
# about 11 000 and the LLM stage to roughly half. The risk is alignment — the
# model must return exactly one result per clause, in order — so the caller
# checks the length and falls back to individual calls if it does not match.

BATCH_SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": CLAUSE_SCHEMA}},
    "required": ["results"],
}

BATCH_SYSTEM = CLAUSE_SYSTEM.replace(
    "You read one clause from an official Uzbek document and report whether it assigns a task.",
    "You read several numbered clauses from an official Uzbek document and report, for each one,"
    " whether it assigns a task.",
).replace(
    """Return JSON with exactly these fields:
- is_task: true if the clause assigns work to someone; false otherwise.""",
    """Return JSON shaped {"results": [ ... ]} with exactly one entry per clause, in the same
order as the clauses were given. If there are five clauses there must be five entries.
Each entry has these fields:
- is_task: true if that clause assigns work to someone; false otherwise.""",
)

BATCH_FEWSHOT = [
    (
        "[1] Context:\n"
        "    Clause: 1. Moliya vazirligi quyidagi vazifalarni bajarsin:\n"
        "[2] Context: 1. Moliya vazirligi quyidagi vazifalarni bajarsin:\n"
        "    Clause: (a) ichki audit oʻtkazilsin, har oyda;\n"
        "[3] Context:\n"
        "    Clause: 2. Belgilab qoʻyilsinki, vazirlik ijro etuvchi hokimiyat organi hisoblanadi.",
        '{"results": ['
        '{"is_task": false, "deadline": null, "executors": []}, '
        '{"is_task": true, "deadline": "har oyda", "executors": ["Moliya vazirligi"]}, '
        '{"is_task": false, "deadline": null, "executors": []}'
        ']}',
    ),
]


def batch_messages(items: list[tuple[str, str]]) -> list[dict]:
    """System + few-shot + several clauses, numbered so results can be aligned."""
    messages = [{"role": "system", "content": BATCH_SYSTEM}]
    for user, assistant in BATCH_FEWSHOT:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    body = "\n".join(
        f"[{i + 1}] Context: {context}\n    Clause: {text}"
        for i, (text, context) in enumerate(items)
    )
    messages.append({"role": "user", "content": body})
    return messages


# --- Call B: a free paragraph, which may hold several tasks ----------------

PARAGRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "deadline": {"type": ["string", "null"]},
                    "executors": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["task", "deadline", "executors"],
            },
        }
    },
    "required": ["tasks"],
}

PARAGRAPH_SYSTEM = CLAUSE_SYSTEM.replace(
    "You read one clause from an official Uzbek document and report whether it assigns a task.",
    "You read one paragraph from an official Uzbek document and list every task it assigns.",
).replace(
    """Return JSON with exactly these fields:
- is_task: true if the clause assigns work to someone; false otherwise.
- deadline: the time limit, copied exactly as written, or null if none is given.
- executors: the organisations or officials made responsible, copied exactly as written. Empty list if none is named.""",
    """Return JSON with a "tasks" array. One entry per task assigned, each with:
- task: the assigned work, copied from the paragraph.
- deadline: the time limit, copied exactly as written, or null if none is given.
- executors: the organisations or officials made responsible, copied exactly as written.

Return an empty array if the paragraph assigns no work.""",
)

PARAGRAPH_FEWSHOT = [
    (
        "Paragraph: Moliya vazirligi zimmasiga ichki audit oʻtkazish vazifasi yuklatiladi. "
        "Ushbu vazifa 1-noyabrgacha bajarilishi lozim.",
        '{"tasks": [{"task": "ichki audit oʻtkazish", "deadline": "1-noyabrgacha", '
        '"executors": ["Moliya vazirligi"]}]}',
    ),
    (
        "Paragraph: Oʻzbekiston Respublikasi Qurilish vazirligi quyidagilarni maʼlum qiladi.",
        '{"tasks": []}',
    ),
]


def paragraph_messages(text: str) -> list[dict]:
    messages = [{"role": "system", "content": PARAGRAPH_SYSTEM}]
    for user, assistant in PARAGRAPH_FEWSHOT:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": f"Paragraph: {text}"})
    return messages


# --- Call A: document details ----------------------------------------------

DETAILS_FIELDS = {
    "organization": "the organisation that issued the document",
    "signatory_name": "the name of the person who signed it",
    "signatory_position": "the position of the person who signed it",
    "document_number": "the registration number of the document",
    "document_date": "the date of the document, exactly as written",
}

DETAILS_SYSTEM = """You read the header and footer of an official Uzbek document and extract its details.

Copy every value exactly as written in the text. Never translate, reword or invent.
If a value is not present, return null. Do not guess.
The organisation is the body that issued the document, usually in the letterhead at the top.
The signatory is the person who signed it, usually at the end."""

DETAILS_FEWSHOT = [
    (
        "Text: OʻZBEKISTON RESPUBLIKASI MOLIYA VAZIRLIGI\n2026-yil 5-mart № 15/2-784\n"
        "...\nVazir oʻrinbosari   B. Toshmatov",
        '{"organization": "OʻZBEKISTON RESPUBLIKASI MOLIYA VAZIRLIGI", '
        '"signatory_name": "B. Toshmatov", "signatory_position": "Vazir oʻrinbosari"}',
    ),
]


def details_schema(fields: list[str]) -> dict:
    """Schema for exactly the fields regex did not already find (section 7.2)."""
    return {
        "type": "object",
        "properties": {f: {"type": ["string", "null"]} for f in fields},
        "required": list(fields),
    }


def details_messages(header: str, footer: str) -> list[dict]:
    messages = [{"role": "system", "content": DETAILS_SYSTEM}]
    for user, assistant in DETAILS_FEWSHOT:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": f"Text: {header}\n...\n{footer}"})
    return messages
