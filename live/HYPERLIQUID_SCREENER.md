# HIP-4 Paper-Screener

`hyperliquid_screener.py` ist eine reine Offline-Analysebibliothek. Sie erhält bereits aufgezeichnete HIP-4-Metadaten, Preis-Kontexte und L2-Bücher als Python-Daten und führt selbst keinerlei Netzwerkzugriffe, Wallet-Aktionen, Orders oder Trading-Runtime-Aufrufe aus. Der bestehende read-only Testnet-Client bleibt davon getrennt.

## Verwendung mit aufgezeichneten Snapshots

```python
from datetime import datetime, timedelta, timezone

from live.hyperliquid_screener import (
    analyze_binary_outcome,
    filter_outcomes,
    parse_outcome_metadata,
)

metadata = parse_outcome_metadata(outcome_record)
near_term_btc = filter_outcomes(
    [metadata],
    underlying="BTC",
    min_remaining=timedelta(minutes=15),
    max_remaining=timedelta(hours=6),
    now=recorded_snapshot_time,
)

metrics = analyze_binary_outcome(
    outcome_record,
    side_price_records,  # Form von get_outcome_prices(); side 0 = Yes, 1 = No
    books={0: yes_l2_snapshot, 1: no_l2_snapshot},
    depth_levels=5,
)
print(metrics.yes.implied_probability, metrics.parity_delta)
```

`parse_outcome_metadata()` extrahiert `class`, `underlying`, `expiry`, `targetPrice` und `period` in `OutcomeMetadata`; unbekannte Attribute bleiben in `extra_attributes` erhalten. `targetPrice` wird als `Decimal` gelesen. `analyze_binary_outcome()` akzeptiert ausschließlich explizit als `class:priceBinary` markierte Outcomes.

Das Feld `expiry` bleibt unverändert als API-Text erhalten. Das Format `YYYYMMDD-HHMM` und ISO-Zeitstempel (UTC-Suffix `Z` oder `z`) werden zu `expires_at` geparst; naive Zeitstempel verwenden standardmäßig UTC. Restlaufzeitfilter erfordern einen deterministischen `now`-Wert mit passender Zeitzone. Auch `now` allein aktiviert die Ablaufprüfung: abgelaufene Outcomes (`remaining <= 0`) und nicht parsebare oder fehlende Expiries werden dann standardmäßig ausgeschlossen. `include_expired=True` erlaubt abgelaufene Outcomes; zusammen mit einem nicht-negativen `min_remaining` ist es widersprüchlich und wird mit `ValueError` abgewiesen. Ein negatives Minimum kann als Untergrenze für die Dauer seit Ablauf dienen.

## Kennzahlen

- Pro Seite wird `midPx` bevorzugt. Fehlt es, verwendet der Screener den Mittelwert aus bestem Bid und Ask. Ohne beides bleibt die implizite Wahrscheinlichkeit `None`.
- `midPx` darf als Randwahrscheinlichkeit `0` oder `1` sein. Ein Preis in einem L2-Orderbuch muss dagegen strikt größer als `0` und höchstens `1` sein; ein API-Level mit `px: "0.00"` wird abgelehnt.
- `parity_delta` beziehungsweise `overround` ist `P(Yes) + P(No) - 1`; ein positiver Wert bedeutet, dass die Summe über 1 liegt. Fehlt eine Seitenwahrscheinlichkeit, bleibt auch die Abweichung `None`.
- `executable_overround` verwendet separat die Best-Asks: `Yes.best_ask + No.best_ask - 1`. Ohne beide Ask-Preise bleibt es `None`; fehlende Seiten werden nicht mit null ergänzt.
- `absolute_spread` ist `best_ask - best_bid` in Preis-Einheiten; ein negatives Ergebnis wird als gekreuztes Buch mit `is_crossed` markiert. `relative_spread` ist Spread geteilt durch BBO-Mittelwert (Ratio, nicht Prozent).
- Bid-/Ask-Tiefe ist die Summe `px * sz` über die ersten N API-Levels. Preise und Größen bleiben `Decimal`; Bids müssen strikt absteigend und Asks strikt aufsteigend (jeweils bestes Level zuerst) sortiert sein. Unsortierte Levels oder doppelte Preise werden mit `ValueError` abgewiesen, damit Top-N-Tiefe nicht still falsch berechnet wird.
- `is_empty_book` kennzeichnet ein vorhandenes Buch mit zwei leeren Seiten; `is_one_sided` kennzeichnet ein Buch mit nur einer Seite. `book_available=False` unterscheidet einen fehlenden Snapshot davon.

## Tests

Alle Screener-Tests laufen ohne Netzwerkzugriff:

```bash
python -m pytest -q tests/test_hyperliquid_screener.py
```
