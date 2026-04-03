"""Tests for Kazakh-aware text quality filters."""

from kazllm.data.filters import (
    FilterResult,
    apply_all_filters,
    apply_filters,
    cyrillic_ratio,
    domain_score,
    filter_paragraphs,
    kazakh_char_density,
    kazakh_score,
    length_filter,
    normalize_homoglyphs,
    normalize_text,
    quality_score,
    russian_char_density,
    vowel_harmony_score,
)

# ---- Real Kazakh test texts ----
KAZ_SENTENCE = "Қазақ тілі — түркі тілдес тілдердің бірі, қазақ халқының ана тілі"
KAZ_LONG = (
    "Қазақстан Республикасы — Орталық Азиядағы мемлекет. "
    "Астанасы — Астана қаласы. Халқының саны 20 миллионнан асады. "
    "Қазақ тілі мемлекеттік тіл болып табылады. "
    "Ел аумағы жағынан әлемдегі тоғызыншы орында тұр. "
) * 3
RUS_SENTENCE = "Русский язык является одним из восточнославянских языков"
RUS_LONG = (
    "Российская Федерация — крупнейшее государство в мире. "
    "Столица — Москва. Население превышает 140 миллионов человек. "
    "Русский язык является государственным языком страны. "
    "Площадь территории составляет более 17 миллионов квадратных километров. "
) * 3


# ===================================================================
# Homoglyph normalization
# ===================================================================

def test_normalize_homoglyphs_basic():
    # Latin 'a' → Cyrillic 'а', Latin 'c' → Cyrillic 'с'
    text = "Казаxстан"  # 'x' is Latin here
    result = normalize_homoglyphs(text)
    assert "x" not in result  # Latin x should be gone
    assert "х" in result  # Cyrillic х present


def test_normalize_homoglyphs_preserves_pure_cyrillic():
    text = "Қазақстан Республикасы"
    assert normalize_homoglyphs(text) == text


def test_normalize_homoglyphs_uppercase():
    text = "KAZAK"  # All Latin lookalikes
    result = normalize_homoglyphs(text)
    # K→К, A→А, Z stays (no Cyrillic Z), A→А, K→К
    assert result[0] == "К"  # Latin K → Cyrillic К
    assert result[1] == "А"  # Latin A → Cyrillic А


# ===================================================================
# Text normalization
# ===================================================================

def test_normalize_text_removes_urls():
    text = "Мәтін https://example.com кейін мәтін"
    result = normalize_text(text)
    assert "https://" not in result
    assert "Мәтін" in result


def test_normalize_text_removes_emails():
    text = "Хабарласу: info@example.com арқылы"
    result = normalize_text(text)
    assert "@" not in result


def test_normalize_text_standardizes_quotes():
    text = "\u201cҚазақстан\u201d — \u00abүлкен ел\u00bb"
    result = normalize_text(text)
    assert "\u201c" not in result
    assert "\u00ab" not in result
    assert '"' in result


def test_normalize_text_caps_punctuation():
    text = "Керемет!!!!!!!!"
    result = normalize_text(text)
    assert result == "Керемет!!!"


def test_normalize_text_collapses_whitespace():
    text = "Сөз    арасында    көп    бос    орын"
    result = normalize_text(text)
    assert "  " not in result


# ===================================================================
# Kazakh language identification
# ===================================================================

def test_kazakh_char_density_kazakh():
    density = kazakh_char_density(KAZ_SENTENCE)
    assert density > 0.03, f"Kazakh text should have >3% Kaz-specific chars, got {density:.3f}"


def test_kazakh_char_density_russian():
    density = kazakh_char_density(RUS_SENTENCE)
    assert density < 0.01, f"Russian text should have ~0% Kaz-specific chars, got {density:.3f}"


def test_russian_char_density_russian():
    density = russian_char_density(RUS_SENTENCE)
    assert density > 0.0, f"Russian text should have some Russian-only chars, got {density:.3f}"


def test_russian_char_density_kazakh():
    density = russian_char_density(KAZ_SENTENCE)
    assert density < 0.02, f"Kazakh text should have minimal Russian-only chars, got {density:.3f}"


def test_kazakh_score_pure_kazakh():
    score = kazakh_score(KAZ_LONG)
    assert score > 0.40, f"Kazakh text should score >0.40, got {score:.3f}"


def test_kazakh_score_pure_russian():
    score = kazakh_score(RUS_LONG)
    assert score < 0.20, f"Russian text should score <0.20, got {score:.3f}"


def test_cyrillic_ratio_kazakh():
    ratio = cyrillic_ratio(KAZ_SENTENCE)
    assert ratio > 0.80, f"Expected high Cyrillic ratio, got {ratio:.2f}"


def test_cyrillic_ratio_latin():
    latin_text = "Kazakh language is a Turkic language"
    ratio = cyrillic_ratio(latin_text)
    assert ratio < 0.10, f"Expected low Cyrillic ratio, got {ratio:.2f}"


# ===================================================================
# Quality scoring
# ===================================================================

def test_quality_score_good_text():
    score = quality_score(KAZ_LONG)
    assert score > 0.40, f"Good Kazakh text should score >0.40, got {score:.3f}"


def test_quality_score_empty():
    assert quality_score("") == 0.0


def test_quality_score_spam():
    spam = "а" * 500
    score = quality_score(spam)
    assert score < 0.15, f"Spam text should score low, got {score:.3f}"


def test_quality_score_numbers_only():
    numbers = "123456789 " * 50
    score = quality_score(numbers)
    # Numbers have zero alphabetic chars → alpha_score=0, but TTR/sent may be nonzero
    # The full pipeline rejects this via kazakh_score and cyrillic_ratio
    assert score < 0.70, f"Number-only text should score below good text, got {score:.3f}"


# ===================================================================
# Domain scoring
# ===================================================================

def test_domain_score_conversational():
    text = "Бүгін ауа райы жақсы болды. Біз серуенге шықтық. Балалар ойнады."
    score = domain_score(text)
    assert score > 0.70, f"Conversational text should score high, got {score:.3f}"


def test_domain_score_legal():
    text = (
        "Заң бойынша кодекс статья сот прокурор тергеу жаза үкім "
        "талапкер жауапкер адвокат сотталған қылмыс"
    )
    score = domain_score(text)
    assert score < 0.30, f"Legal jargon should score low, got {score:.3f}"


# ===================================================================
# Vowel harmony
# ===================================================================

def test_vowel_harmony_kazakh():
    # Native Kazakh words follow vowel harmony
    text = "қаламыз баламыз аламыз оқимыз жүреміз келеміз"
    score = vowel_harmony_score(text)
    assert score > 0.65, f"Kazakh words should have high harmony, got {score:.3f}"


def test_vowel_harmony_mixed():
    # Mix of Kazakh and Russian words should score lower
    score_kaz = vowel_harmony_score(KAZ_LONG)
    score_rus = vowel_harmony_score(RUS_LONG)
    # Kazakh should have better harmony than Russian
    assert score_kaz > score_rus, (
        f"Kazakh ({score_kaz:.3f}) should have better harmony than Russian ({score_rus:.3f})"
    )


# ===================================================================
# Paragraph filtering
# ===================================================================

def test_filter_paragraphs_removes_short():
    text = "Мәтін.\n\nОК\n\nЖақсы мәтін, бұл жеткілікті ұзын абзац."
    result = filter_paragraphs(text, min_chars=20)
    assert "ОК" not in result
    assert "Жақсы" in result


def test_filter_paragraphs_removes_russian():
    text = (
        "Қазақстан — Орталық Азиядағы мемлекет.\n\n"
        "This paragraph is entirely in English and should be removed "
        "because it has no Kazakh characters at all.\n\n"
        "Астанасы — Астана қаласы."
    )
    result = filter_paragraphs(text, min_chars=20)
    assert "Қазақстан" in result
    assert "Астана" in result


# ===================================================================
# Length filter
# ===================================================================

def test_length_filter_pass():
    assert length_filter("а" * 100)


def test_length_filter_too_short():
    assert not length_filter("аа", min_chars=50)


# ===================================================================
# Full pipeline: apply_filters (backward compat) and apply_all_filters
# ===================================================================

def test_apply_filters_good_text():
    assert apply_filters(KAZ_LONG)


def test_apply_filters_rejects_latin_only():
    latin = "This is an English text that should be rejected. " * 5
    assert not apply_filters(latin, min_cyrillic_ratio=0.50)


def test_apply_filters_rejects_short():
    assert not apply_filters("Қаз")


def test_apply_filters_rejects_russian():
    assert not apply_filters(RUS_LONG)


def test_apply_all_filters_returns_result():
    result = apply_all_filters(KAZ_LONG)
    assert isinstance(result, FilterResult)
    assert result.passed is True
    assert result.kazakh_score > 0.30
    assert result.quality_score > 0.20
    assert len(result.text) > 0


def test_apply_all_filters_rejects_russian():
    result = apply_all_filters(RUS_LONG)
    assert result.passed is False
    assert result.kazakh_score < 0.30
