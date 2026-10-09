# Regex accuracy (plan section 6)

Documents: 52

| Slice | N | Number | Date |
|---|---:|---:|---:|
| **all** | 52 | 87% | 94% |
| lexuz | 12 | 92% | 100% |
| synthetic | 40 | 85% | 92% |
| latin | 26 | 100% | 96% |
| cyrillic | 26 | 73% | 92% |
| text layer | 26 | 100% | 100% |
| scanned | 26 | 73% | 88% |
| latin / text | 13 | 100% | 100% |
| latin / scan | 13 | 100% | 92% |
| cyrillic / text | 13 | 100% | 100% |
| cyrillic / scan | 13 | 46% | 85% |

## Known limitations

Every remaining failure is OCR character corruption on a *scanned*
page, not a missing pattern. Both fields are 100% on text-layer
documents in both scripts.

* `/` inside a number is read as `1`: `17/1-674` -> `1711-674`. The
  number is found, but two characters are wrong. The LLM fallback
  reads the same corrupted text, so it cannot recover the original
  either.
* A day digit is read as a letter: `«6»` -> `кб`, `«5»` -> `S5`. The
  date then has no digit to match and is lost.
* Guillemets around the day are read as letters (`«29»` -> `к29э`),
  which breaks the day/month join in `DATE_DMY`.
* `№` printed *after* the number is not matched.
* Where a scan loses the document's own number, the leftmost match can
  fall on a *referenced* act's number instead (seen once:
  `lexuz_8517274_cyr_scan` returning `PQ-265` from the preamble).


## Failures (10)

| Document | Field | Expected | Found |
|---|---|---|---|
| doc_01_cyr_scan | number | `17/1-674` | `1711-674` |
| doc_03_cyr_scan | date | `2026-03-29` | `None` |
| doc_04_cyr_scan | number | `18/9-100` | `1819-100` |
| doc_05_cyr_scan | number | `1/2-824` | `None` |
| doc_05_lat_scan | date | `2026-02-05` | `2026-09-29` |
| doc_06_cyr_scan | number | `7/9-559` | `719-559` |
| doc_07_cyr_scan | date | `2026-09-06` | `None` |
| doc_08_cyr_scan | number | `15/9-820` | `1519-820` |
| doc_09_cyr_scan | number | `14/3-142` | `1413-142` |
| lexuz_8517274_cyr_scan | number | `3948` | `PQ-265` |
