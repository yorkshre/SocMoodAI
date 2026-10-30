# Wskaźniki językowe depresji w mediach społecznościowych (PL/EN) — szkielet badawczy

Cel: sprawdzić, czy interpretowalne cechy psycholingwistyczne (zaimki 1. os. l. poj.,
słownictwo absolutystyczne, aktywność nocna) poprawiają klasyfikację ponad samym tekstem (TF-IDF),
osobno dla polskiego i angielskiego.

**To nie jest narzędzie diagnostyczne.** Wynik oznacza wzorce językowe, nie diagnozę.

## Uruchomienie
    pip install -r requirements.txt
    pytest
    python scripts/run_baseline.py --data data/posts.csv --per-lang

## Format danych (`data/posts.csv`)
| kolumna   | opis                                                    |
|-----------|---------------------------------------------------------|
| user_id   | zanonimizowany identyfikator (stały dla użytkownika)    |
| timestamp | data i godzina (w czasie lokalnym użytkownika, jeśli to możliwe) |
| text      | treść wpisu                                             |
| label     | 0/1, jedna wartość na użytkownika                       |
| lang      | opcjonalnie `pl` / `en` (inaczej wykrywane heurystyką)  |

## Model transformatorowy
    pip install -r requirements-transformer.txt
    python scripts/train_transformer.py --train-data data/en.csv \
        --eval-data pl=data/pl.csv --model xlm-roberta-base --epochs 3 --out runs/xlmr_en
Podział użytkowników train/val/test, chunkowanie tekstu, agregacja wyników do poziomu użytkownika,
próg wybierany na walidacji, zapis `results.json` i `splits.json`. Zero-shot EN→PL przez `--eval-data`.
Dla samego angielskiego: `--model roberta-base`; dla samego polskiego: `--model allegro/herbert-base-cased`.

## Co jest w środku
- `preprocess.py` — czyszczenie (URL, @wzmianki, HTML), tokenizacja, prosta detekcja języka
- `lexicons.py` — słowniki PL/EN (punkt wyjścia, do weryfikacji z literaturą)
- `features.py` — cechy na wpis i agregacja do poziomu użytkownika
- `baseline.py` — TF-IDF / cechy / oba, walidacja krzyżowa **po użytkownikach**, ablacja
- `transformer.py` — fine-tuning, chunkowanie, agregacja, dobór progu, metryki
- `tests/` — testy jednostkowe

## Zasady metodologiczne
- Podział po użytkownikach, nigdy po wpisach (wyciek danych).
- Stałe ziarno losowości, konfiguracja w kodzie/skryptach, raportuj wszystkie metryki.
- Raportuj wyniki osobno dla PL i EN; nie mieszaj bez uzasadnienia.
- Przed pracą na danych: zgoda komisji etycznej / zgodność z umową o dostępie do zbioru; bez publikowania surowych wpisów.

## Następne kroki
1. Zdobyć dane (patrz `data/README.md`).
2. Zastąpić `detect_lang` modelem (fastText/langdetect) i rozszerzyć słowniki.
3. Dodać dostrojony transformer: HerBERT (PL), RoBERTa (EN) lub XLM-R (oba).
4. Dodać analizę czasową (trend, persystencja) i metryki eRisk (ERDE, latency-weighted F1).
5. Interpretowalność (wagi cech, SHAP) i analiza błędów.
