import pytest

from app.core.entities import EntityExtractor, Gazetteer

GAZETTEER_YAML = """
- id: dxm
  type: ORG
  aliases: [Davlat xizmatlari markazi, DXM]
- id: soliq_qomitasi
  type: ORG
  aliases: [Davlat soliq qo'mitasi]
- id: soliq
  type: ORG
  aliases: [Soliq qo'mitasi]
- id: eri
  type: TERM
  aliases: [Elektron raqamli imzo]
  case_sensitive: [ERI]
- id: id_karta
  type: TERM
  aliases: [ID-karta]
"""


@pytest.fixture
def extractor(tmp_path):
    path = tmp_path / "gazetteer.yaml"
    path.write_text(GAZETTEER_YAML, encoding="utf-8")
    return EntityExtractor(Gazetteer.from_yaml(path))


def tokens(extractor, text):
    return extractor.process(text).tokens


@pytest.mark.parametrize(
    "text", ["PQ-4567", "pq 4567", "pq4567", "ПҚ-4567", "PQ–4567", "PQ-4567-son"]
)
def test_law_variants_give_same_token(extractor, text):
    assert tokens(extractor, text) == {"LAW_pq_4567"}


def test_close_law_numbers_stay_distinct(extractor):
    assert tokens(extractor, "PQ-4567") != tokens(extractor, "PQ-4576")


@pytest.mark.parametrize("text", ["O'RQ-1074-son qonun", "O‘RQ 1074", "ORQ-1074", "ЎРҚ-1074"])
def test_orq_law_is_one_entity(extractor, text):
    result = extractor.process(text)
    assert [e.token for e in result.entities] == ["LAW_orq_1074"]


@pytest.mark.parametrize("text", ["PF-60", "VMQ-45"])
def test_other_law_types(extractor, text):
    (entity,) = extractor.process(text).entities
    assert entity.type == "LAW"


@pytest.mark.parametrize("text", ["15-modda", "15-moddasi", "15 moddasiga", "15–moddada"])
def test_articles(extractor, text):
    assert tokens(extractor, text) == {"ART_15"}


@pytest.mark.parametrize("text", ["4567-son", "4567-sonli qaror"])
def test_numbers(extractor, text):
    assert tokens(extractor, text) == {"NUM_4567"}


@pytest.mark.parametrize(
    "text",
    [
        "DXM",
        "Davlat xizmatlari markazi",
        "davlat xizmatlari markaziga boring",
        "DXMga",
        "DXM-ga",
        "DXM ga",
        "ДХМ га",
    ],
)
def test_org_aliases_and_suffixes(extractor, text):
    assert tokens(extractor, text) == {"ORG_dxm"}


def test_longest_alias_wins(extractor):
    assert tokens(extractor, "Davlat soliq qo'mitasiga ariza") == {"ORG_soliq_qomitasi"}
    assert tokens(extractor, "Soliq qo'mitasi") == {"ORG_soliq"}


def test_case_sensitive_alias(extractor):
    assert tokens(extractor, "ERI orqali imzolang") == {"TERM_eri"}
    assert tokens(extractor, "Elektron raqamli imzo") == {"TERM_eri"}
    assert tokens(extractor, "eri ishlaydi") == set()  # "eri" = "her husband"


def test_hyphenated_term_with_suffix(extractor):
    assert tokens(extractor, "ID-kartamni yo'qotdim") == {"TERM_id_karta"}
    assert tokens(extractor, "id karta") == {"TERM_id_karta"}


def test_no_partial_word_match(extractor):
    assert tokens(extractor, "DXMlarning gazetasi") == {"ORG_dxm"}
    assert tokens(extractor, "ADXM") == set()


def test_tagged_text_is_clean(extractor):
    result = extractor.process("PQ-4567 qaroriga ko'ra DXMga murojaat qiling!")
    assert result.text == "LAW_pq_4567 qaroriga ko'ra ORG_dxm murojaat qiling"
    assert [e.type for e in result.entities] == ["LAW", "ORG"]


def test_missing_gazetteer_file_still_tags_laws(tmp_path):
    extractor = EntityExtractor(Gazetteer.from_yaml(tmp_path / "missing.yaml"))
    assert extractor.process("PQ-1 DXM").tokens == {"LAW_pq_1"}


def test_duplicate_alias_rejected(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text("- {id: a, aliases: [DXM]}\n- {id: b, aliases: [dxm]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="used by both"):
        Gazetteer.from_yaml(path)


def test_bad_entry_rejected(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text("- {id: Bad-Id, aliases: [X]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Bad gazetteer entry"):
        Gazetteer.from_yaml(path)
