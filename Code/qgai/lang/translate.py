import json
from functools import lru_cache
from pathlib import Path

__all__ = ["translate", "detranslate"]
LANG_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=8)
def _load_language(lang: str) -> dict[str, str]:
    path = LANG_DIR / f"{lang}.lang"
    with path.open(encoding="utf-8") as language_file:
        return json.load(language_file)


def translate(vocab: str, lang: str = "zh-cn") -> str | None:
    return _load_language(lang).get(vocab)


def detranslate(vocab: str, lang: str = "zh-cn") -> str | None:
    language = _load_language(lang)
    return {translated: source for source, translated in language.items()}.get(vocab)
