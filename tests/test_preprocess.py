from dd.preprocess import clean_text, detect_lang, tokenize


def test_removes_url_mention_html():
    out = clean_text("Hej @jan &amp; <b>wszyscy</b> https://x.com/a?b=1 www.test.pl ok")
    assert "@" not in out and "http" not in out and "www" not in out and "<b>" not in out
    assert "&" in out and "wszyscy" in out


def test_non_string_and_empty():
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_tokenize_keeps_polish_diacritics():
    assert tokenize("Żółć, ŁÓDŹ!") == ["żółć", "łódź"]


def test_detect_lang():
    assert detect_lang("Nie wiem, czy to się uda, ale ja próbuję") == "pl"
    assert detect_lang("I think that it is the end of my story") == "en"
    assert detect_lang("") == "unk"
    assert detect_lang("12345 !!!") == "unk"
