import pytest

from app.core.normalize import normalize
from app.core.synonyms import QueryExpander, SynonymGroup


@pytest.fixture
def expander():
    return QueryExpander(
        [
            SynonymGroup("haydovchilik guvohnomasi", ("prava", "voditelskiy prava")),
            SynonymGroup("elektr", ("svet",)),
            SynonymGroup("gaz ta'minoti", ("gaz",)),
        ]
    )


@pytest.mark.parametrize("query", ["prava olish", "pravamni yo'qotdim", "Права олиш", "PRAVA ga"])
def test_variants_with_suffixes_and_scripts(expander, query):
    assert expander.expand(normalize(query)).endswith("haydovchilik guvohnomasi")


def test_whole_words_only(expander):
    assert expander.expand("gazeta o'qiyman") == "gazeta o'qiyman"
    assert expander.expand("gazga to'lov") == "gazga to'lov gaz ta'minoti"


def test_multiple_groups_and_no_duplicates(expander):
    assert (
        expander.expand("svet va prava prava")
        == "svet va prava prava haydovchilik guvohnomasi elektr"
    )
    assert expander.expand("haydovchilik guvohnomasi prava") == "haydovchilik guvohnomasi prava"


def test_no_match_returns_text_unchanged(expander):
    assert expander.expand("soliq qarzi") == "soliq qarzi"


def test_yaml_loading_and_validation(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text("- {canonical: ta'til, variants: [otpusk]}\n", encoding="utf-8")
    assert QueryExpander.from_yaml(good).expand("otpusk") == "otpusk ta'til"
    assert QueryExpander.from_yaml(tmp_path / "missing.yaml").groups == []
    bad = tmp_path / "bad.yaml"
    bad.write_text("- {canonical: ta'til}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs 'canonical' and 'variants'"):
        QueryExpander.from_yaml(bad)
