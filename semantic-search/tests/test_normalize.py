import pytest

from app.core.normalize import (
    APOSTROPHE_VARIANTS,
    clean_text,
    normalize,
    normalize_script,
    transliterate,
)


def test_all_apostrophe_variants_normalize_identically():
    forms = {normalize(f"o{ch}zgartirish") for ch in "'" + APOSTROPHE_VARIANTS}
    assert forms == {"o'zgartirish"}


def test_tutuq_belgisi_kept_inside_word():
    assert normalize("Ma’lumot bering!") == "ma'lumot bering"


@pytest.mark.parametrize(
    ("cyrillic", "latin"),
    [
        ("Паспортни қандай ўзгартириш мумкин?", "Pasportni qanday o'zgartirish mumkin?"),
        ("Шахсий маълумотлар ғалаба ҳужжат", "Shaxsiy ma'lumotlar g'alaba hujjat"),
        ("Чет эллик фуқаро ёки юридик шахс", "Chet ellik fuqaro yoki yuridik shaxs"),
        ("Мен ерга бордим, Ер юзи", "Men yerga bordim, Yer yuzi"),
    ],
)
def test_cyrillic_and_latin_normalize_identically(cyrillic, latin):
    assert normalize(cyrillic) == normalize(latin)


def test_ye_rule():
    assert transliterate("ер") == "yer"
    assert transliterate("келди") == "keldi"
    assert transliterate("оила ерга") == "oila yerga"


def test_case_is_kept_before_cleaning():
    assert normalize_script("ДХМ") == "DXM"
    assert normalize_script("ЎРҚ-1074") == "O'RQ-1074"
    assert normalize_script("Шахс") == "Shaxs"
    assert normalize_script("ЕРИ") == "ERI"


def test_homoglyph_inside_latin_word():
    # Answer 499 contains a Cyrillic "а" inside the Latin word "ega".
    assert normalize("Yangi egа ta'minlovchi") == "yangi ega ta'minlovchi"


def test_hyphens_and_punctuation():
    assert normalize("ID-karta (yangi), ota-ona; my.gov.uz!") == "id karta yangi ota ona my gov uz"


def test_stray_apostrophes_removed():
    assert normalize("'qo'shimcha' ma'lumot'") == "qo'shimcha ma'lumot"


def test_strip_apostrophes_option():
    assert normalize("Ma'lumot o'zgartirish", strip_apostrophes=True) == "malumot ozgartirish"


def test_entity_tokens_survive_cleaning():
    assert clean_text("LAW_pq_4567 qarori, ORG_dxm-ga!") == "LAW_pq_4567 qarori ORG_dxm ga"


@pytest.mark.parametrize(
    "text",
    ["Паспорт ID-karta (yangi)!", "Ma‘lumot  o`zgartirish", "LAW_pq_1 va ORG_dxm", "  "],
)
def test_normalize_is_idempotent(text):
    once = normalize(text)
    assert normalize(once) == once
