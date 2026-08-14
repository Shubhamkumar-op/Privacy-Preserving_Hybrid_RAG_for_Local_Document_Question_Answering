import argostranslate.translate


def translate_to_hindi(text: str) -> str:
    languages = argostranslate.translate.get_installed_languages()
    source = next((lang for lang in languages if lang.code == "en"), None)
    target = next((lang for lang in languages if lang.code == "hi"), None)
    if source is None or target is None:
        raise RuntimeError("English-to-Hindi Argos translation model is not installed.")
    return source.get_translation(target).translate(text)
