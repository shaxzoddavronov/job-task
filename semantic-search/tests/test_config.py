from app.core.config import Settings


def test_defaults_load():
    settings = Settings(_env_file=None)
    assert settings.eval_seed == 42
    assert settings.data_source in {"postgres", "file"}


def test_bot_token_prefers_env(tmp_path):
    token_file = tmp_path / "token.txt"
    token_file.write_text("from-file\n")
    settings = Settings(_env_file=None, bot_token="from-env", bot_token_file=token_file)
    assert settings.resolve_bot_token() == "from-env"


def test_bot_token_falls_back_to_file(tmp_path):
    token_file = tmp_path / "token.txt"
    token_file.write_text("  from-file\n")
    settings = Settings(_env_file=None, bot_token=None, bot_token_file=token_file)
    assert settings.resolve_bot_token() == "from-file"


def test_bot_token_missing(tmp_path):
    settings = Settings(_env_file=None, bot_token=None, bot_token_file=tmp_path / "nope.txt")
    assert settings.resolve_bot_token() is None
