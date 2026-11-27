# Aura-Bot: Review- und Weiterarbeits-Handoff

- **Stand:** 27. November 2026
- **Repository:** [`nobody55555/aura-bot`](https://github.com/nobody55555/aura-bot)
- **HyperLiQ-Ausgangspunkt:** Branch `handoff/aura-bot-2026-11-25`, Commit [`a7f6dcf1c4249cb945d96ba20a78c2509411d250`](https://github.com/nobody55555/aura-bot/commit/a7f6dcf1c4249cb945d96ba20a78c2509411d250)
- **Ursprüngliche Basis laut Handoff vom 25.11.:** `feature/live-foundation` bei `7353e69b1ef5cc958e335040dadf70d9e1305eea`
- **Vorgesehener unabhängiger Review-Branch:** `review/aura-grok-2026-11-27` (auf `a7f6dcf` aufbauend)

> Dieses Dokument beschreibt Code- und Teststatus, keine Handelsfreigabe. Die Tatsache, dass Live-Ausführung implementiert ist, ist weder eine Empfehlung, sie zu aktivieren, noch ein Nachweis von Profitabilität oder sicherer Börsenausführung.

## 1. Schnellüberblick

- `AURA_MODE` fällt in `config.py` weiterhin auf **`paper`** zurück. Live wird nicht durch den Default aktiviert.
- Wird `AURA_MODE=live` ausdrücklich gesetzt und sind die erforderlichen Zugangsdaten vorhanden, kann ein zulässiges Kaufsignal nun tatsächlich über `OrderManager` an den konfigurierten CCXT-Broker übermittelt werden. Die frühere explizite Blockade in `main.py` wurde entfernt, weil sie fälschlich als dauerhafte Anforderung behandelt worden war.
- Die vorhandenen Vorabprüfungen vor einem Live-Signal bleiben bestehen: aktive SafetyMonitor-Prüfung, Marktdaten-Freshness, positives `USDT.total`, Börsenuhr-/Clock-Drift-Prüfung, Equity-/Risikoprüfungen und persistierter Kill-Switch. Für bestätigte oder teilweise gefüllte Entries fordert der `OrderManager` weiterhin SL- und TP-Schutz an.
- Die vier vom vorangegangenen Review priorisierten Robustheitskorrekturen sind enthalten: Fail-closed für beschädigte Kill-Switch-Werte, positive Bid-/Ask-Größen, eindeutige naive DST-Ortszeiten und numerische Validierung des read-only Hyperliquid-Testnet-Kontexts.
- Der gespeicherte Kill-Switch wird sowohl von `SafetyMonitor` als auch vom `OrderManager` gelesen. Beide akzeptieren nur die Strings `"0"` und `"1"`; unbekannte Werte halten den Prozess/Order-Pfad an.
- Die Änderungen wurden mit Fakes/offline getestet; es wurden keine echten Börsenorders gesendet und keine Live-/Testnet-Abfragen ausgeführt.

## 2. Welche Änderung ist von wem bzw. aus welchem Stand?

Der gepushte HyperLiQ-Arbeitsstand endet bei Commit `a7f6dcf1c4249cb945d96ba20a78c2509411d250` auf `handoff/aura-bot-2026-11-25`. Der exakte HyperLiQ-Code ist am zuverlässigsten über diesen GitHub-Commit und dessen Branch-Historie zu prüfen; ein manuell nachgebildeter Patch wäre eine unnötige zweite Quelle. Für einen vollständigen Branchvergleich mit der damaligen Basis kann lokal verwendet werden:

```bash
git diff 7353e69b1ef5cc958e335040dadf70d9e1305eea..a7f6dcf1c4249cb945d96ba20a78c2509411d250
```

Der vorgesehene Review-Branch startet direkt bei `a7f6dcf`. Damit lassen sich zwei Reviews klar trennen:

- **HyperLiQ-Meilenstein:** ursprüngliche Basis `7353e69` bis `a7f6dcf`.
- **Folgekorrekturen und Klarstellung Live:** `a7f6dcf` bis zum Head des Review-Branches. Dazu gehören die vier Review-Fixes, die Live-Pfad-Aktivierung (bei weiterem Paper-Default) und die aktuelle Dokumentation.

Die neue Hand-off-Datei ist die aktuelle Quelle für den Zustand nach diesen Arbeiten. `HANDOFF_AURA_2026-11-25.md` und `LIVE_IMPLEMENTATION_PLAN.md` bleiben als historische Momentaufnahmen erhalten und tragen jetzt einen Hinweis, dass ihre früheren Statusangaben überholt sind.

## 3. Paper- und Live-Verhalten

### Paper-Modus

- Ohne `AURA_MODE` wird `Settings.mode` in `config.py` auf `paper` gesetzt.
- Es werden keine Börsenschlüssel für Paper benötigt.
- `main.py` erstellt `PaperBroker`; für jede Runde wird eine simulierte Paper-Ausführung über den `OrderManager` versucht.
- Der PaperBroker erhält vor der Paper-Ausführung den letzten Marktpreis, damit seine Simulation diesen Wert nutzen kann.

### Expliziter Live-Modus

- `AURA_MODE=live` ist eine bewusste Konfigurationsentscheidung; `config.py` verlangt Exchange-Credentials. `main.py` instanziiert `CcxtBroker` mit `sandbox=False`.
- Das ist ein echter Produktionspfad, kein bloßer Dry-Run: Ein passendes Signal kann eine reale Börsenorder auslösen. Nur mit echtem Exchange-Konto, geeigneten Zugangsdaten und nach Prüfung der Börseneinstellungen aktivieren.
- Vor der Signalauswertung werden Marktdaten-Freshness, `USDT.total`-Equity, RiskManager-Equity, Exchange-Zeit und Clock-Drift geprüft. Fehlende/ungültige Equity bzw. Zeitdaten führen zum Halt statt zu einem Orderversuch. Ein bereits aktiver oder nicht lesbarer/beschädigter Kill-Switch blockiert Orders.
- Aktuell ist der Runtime-Einstieg **long-only**: ein `buy`-Signal, `risk.can_trade()` und positive berechnete Größe sind erforderlich. Der gemeinsame Aufruf geht an `OrderManager.create_entry_order(...)` mit Market-Order und `place_protection=True`.
- Für einen gefüllten Long-Entry berechnet `OrderManager` standardmäßig einen separaten Stop-Market bei 2 % unter Fill und ein separates Take-Profit-Limit bei 6 % über Fill. Das sind gegenwärtige Implementierungswerte, keine empirisch validierten Empfehlungen.
- SL und TP sind **keine OCO-Gruppe**: der jeweils andere Auftrag wird nicht automatisch als atomarer OCO-Partner aufgehoben. Die konkrete CCXT-/Binance-Parameterabbildung und die reale Reservierungs-/Fill-/Cancel-Semantik sind explizit offene Reviewpunkte.
- Bei Fehlern im Schutzorderpfad versucht der `OrderManager` zu halten und neue Entries zu stoppen. Das macht einen bereits akzeptierten Entry nicht rückgängig und garantiert keine Börsenausführung oder Verlustobergrenze.

**Nicht Teil dieses Pfades:** `live/hyperliquid_testnet.py` ist weiterhin ein separater read-only Info-Client für explizite Hyperliquid-Testnet-Abfragen. Er ist nicht an die Aura-Trading-Runtime angeschlossen und kann keine Testnet- oder Mainnet-Orders senden.

## 4. Codeänderungen in diesem Review-Branch

| Datei(en) | Änderung / Review-Intention |
|---|---|
| `live/safety.py`, `tests/test_safety.py` | Persistierte Kill-Switch-Werte streng auf `"0"`/`"1"` begrenzen; beschädigte Werte beim Neustart fail-closed halten und Wiederaufnahme ohne explizite Operatorfreigabe ablehnen. Tests für `"true"`, `"2"` und `""`. |
| `live/order_manager.py`, `tests/test_order_manager.py` | Dieselbe strenge Wertvalidierung am zweiten, direkten Order-Einstiegspunkt absichern. Neustart mit beschädigten gespeicherten Werten muss Order-Erstellung blockieren. |
| `live/hyperliquid_screener.py`, `tests/test_hyperliquid_screener.py` | Naive lokale Expiries durch beide `fold`-Kandidaten und UTC-Roundtrip prüfen; nicht existente und doppeldeutige Ortszeiten ablehnen. Zürcher 25.10.2026 02:30 (doppeldeutig) und 29.03.2026 02:30 (nicht existent) getestet. Explizite Offsets und feste Zeitzonen bleiben erlaubt. Book-Größen `<= 0` auf Bid und Ask ablehnen. |
| `live/hyperliquid_testnet.py`, `tests/test_hyperliquid_testnet.py` | Optionale Kontextwerte vor Rückgabe validieren: nur endliche numerische Werte; binäre Preisfelder (`markPx`, `midPx`, `prevDayPx`) in `[0, 1]`; Volumenfelder (`dayNtlVlm`, `dayBaseVlm`) `>= 0`; `None` weiter zulassen. |
| `main.py`, `tests/test_main_safety.py` | Frühere unbedingte Live-Order-Blockade entfernt. Paper- und Live-Ausführung nutzen nach ihren jeweiligen Vorarbeiten den `OrderManager`; für Live wird vor Submit ausdrücklich gewarnt. Offline-Test prüft, dass ein gültiges Live-Kaufsignal nach Vorabprüfungen den `OrderManager` mit den erwarteten Orderparametern erreicht. |
| `README.md`, `HANDOFF_AURA_2026-11-25.md`, `LIVE_IMPLEMENTATION_PLAN.md`, dieses Dokument | Paper-Default und expliziten Live-Pfad korrekt beschreiben; frühere historische Dokumente als überholt markieren; Tests, Risiken, Grenzen und Review-Aufgaben zentral zusammenfassen. |

## 5. Test- und Prüfstatus

Nach den Codeänderungen lief die vollständige Suite lokal/offline:

```bash
PYTHONPATH=. .venv/bin/pytest -q
# 190 passed in 1.53s
```

Die drei später ergänzten OrderManager-Kill-Switch-Regressionen decken `"true"`, `"2"` und `""` am direkten Orderpfad ab. Die Laufzeit-/Live-Route wird mit Fake-Analyzer, Fake-Exchange-Uhr, Fake-Broker und aufgezeichnetem OrderManager-Aufruf getestet. Das beweist nur, dass Runtime-Code die Aufträge nach den geprüften Vorbedingungen weiterreicht; es ist **kein** Test der CCXT-/Binance-API gegen echte Konten oder eine Sandbox.

Der Ruff-Abgleich über die zehn geänderten Python-Dateien ergab **88 Diagnosen im Basis-Commit und 88 im Arbeitsstand; keine neuen Diagnosekategorien**. Das ist ein Vergleich mit dem bestehenden Repository-Stand, kein sauberer Ruff-Lauf: vorhandene Lint-Befunde wurden nicht im Rahmen dieses Patches global umformatiert. Black meldet vier HIP-4-Screener-/Testdateien als unverändert formatiert. Sechs geänderte Safety-/Runtime-Dateien scheitern auch im unveränderten Basis-Commit am Black-Check; breite Bestandsformatierung wurde vermieden.

Vor Veröffentlichung sind erneut auszuführen:

```bash
PYTHONPATH=. python -m pytest -q
git diff --check
```

Der ursprüngliche Run vor der Live-Klarstellung hatte 187 bestandene Tests. Die aktuelle Zahl **190** enthält die drei zusätzlichen direkten OrderManager-Kill-Switch-Fälle. Für jeden späteren Code-Edit ist der Testlauf neu zu wiederholen; alte Zähler nicht als Nachweis für neue Edits übernehmen.

## 6. Wichtige offene Reviewpunkte (bitte nicht als erledigt interpretieren)

### Live-Ausführung / Börsensemantik

1. Verifiziere in `live/broker/ccxt_broker.py` die echte Parametermapping-Kompatibilität für den konkret unterstützten Exchange (der Default ist Binance). Insbesondere Stop-Market- und Limit-Schutzorders, `stopPrice`, `stopLossPrice`, Client-Order-IDs, Rundung, Minimum-Notional und Ablehnungsfälle müssen gegen die gewählte Marktart geprüft werden.
2. Prüfe das separate, nicht-OCO SL-/TP-Verhalten nach vollständigem und partiellem Fill, unmittelbar gefülltem Limit, Neustart/Reconciliation, Cancel und einem einzelnen ausgeführten Exit. Ein Schutzorderfehler kann eine bestehende Position ungeschützt lassen.
3. Prüfe, ob verfügbare/gebundene Assets, bestehende offene Positionen, Gebühren, Spread, Slippage, Liquidität, API-Rechte, Rate-Limits und Börsenstörungen im Sizing und in der Recovery angemessen behandelt sind.
4. Es existiert in diesem Nachweis keine echte Exchange-Integrationstest-Ausführung. Bitte keine realen Schlüssel in Issues, Prompts, Tests oder Doku committen.

### Risiko- und Paper-Qualität

- Der Runtime-Sizing-Stop ist `letzter Kurs - 2 × ATR`, während der tatsächlich berechnete OrderManager-Stop aktuell standardmäßig 2 % unter dem Entry-Fill liegt. Diese Diskrepanz kann das Positionsrisiko anders machen als das Sizing suggeriert; vor Risikobehauptungen harmonisieren.
- Der `PaperBroker` bildet im dokumentierten älteren Handoff BTC-Bestand/unrealisiertes P&L nicht vollständig in `USDT.total`-Equity ab. Equity-basierte Tagesverlust-/Drawdown-Aussagen können daher bei offenen Positionen unvollständig sein.
- Es liegt in diesem Repository kein reproduzierbarer Aura-100-Trade-Backtest oder Out-of-Sample-Nachweis vor. Tests sind Funktionsnachweise, keine Profitabilitätsevidenz.
- Ein Startwert von ungefähr 50 CHF ist ein Ziel/Plan, keine validierte Mindestgröße. Fees, Mindestnotional, Rundungen und verfügbare BTC/USDT-Paare können kleine Konten ungeeignet machen.

### Vier Robustheitskorrekturen

- Für Kill-Switch-Änderungen stets beide Leser prüfen: `SafetyMonitor` und `OrderManager`; unbekannte Werte dürfen an keinem Order-Einstiegspunkt als `"0"` behandelt werden.
- DST-Tests kontrollieren naive Ortszeiten; explizite UTC-Offsets/feste Offsets sollen nicht durch die Ablehnung lokaler DST-Gaps beeinträchtigt werden.
- Der Testnet-Client darf read-only bleiben. Schema-/Numerikvalidierung erweitert weder seine Netzwerkberechtigungen noch eine Order-Funktion.

## 7. Review-Checkliste für Grok

Bitte den Code im GitHub-Review-Branch gegen seinen Eltern-Commit prüfen und die Befunde nach **Sicherheit / Korrektheit / Regression / Dokumentation** priorisieren. Für jedes Finding bitte Pfad/Zeile, auslösende Eingabe oder Ablauf, Auswirkung und konkrete Abhilfe angeben. Besonders erwünscht:

- Bestätigen, dass `paper` ohne Umgebungsvariable Standard bleibt und `live` weder default noch durch eine versteckte Gate-Bedingung wirkungslos ist.
- Verfolgen: `Settings.validate` → `build_components` → Runtime-Preflight → Signal/Risk → `OrderManager` → `CcxtBroker` → SL/TP-/Reconciliation-Lifecycle.
- Prüfen, dass die neuen OrderManager- und SafetyMonitor-Tests denselben beschädigten DB-Zustand fail-closed abdecken.
- Exchange-spezifische Live-Semantik des separaten SL/TP ohne reale Orderausführung kritisch bewerten; keine Behauptung auf Basis bloßer Unit-Tests.
- Kontrollieren, dass Zurich DST-Gap/Overlap, explizite Offsets, bid/ask Nullgrößen und Testnet-Preis-/Volumenbereiche vollständig getestet sind.
- Auf veraltete gegenteilige Aussagen in Markdown-Dateien oder Kommentaren achten. Die zwei alten Statusdokumente sind jetzt als historisch gekennzeichnet.
- Keine Aussage zu Rendite, Profitabilität oder einer sicheren Live-Freigabe aus den bestandenen Tests ableiten.

## 8. Anleitung für neue Threads und lokale Reproduktion

1. Repository öffnen und `git status --short`, `git log --oneline --decorate -8` sowie `git diff --stat` prüfen. Vorhandene Änderungen nicht überschreiben oder stillschweigend verwerfen.
2. Dieses Handoff, `README.md` und bei HIP-4-Fragen `live/HYPERLIQUID_SCREENER.md` lesen. Das historische August-Plan-Dokument und das Handoff vom 25.11. sind Snapshots, keine aktuellen Betriebsanweisungen.
3. Mit isolierter Python-Umgebung installieren und lokal testen:

   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   python -m pip install -r requirements.txt
   PYTHONPATH=. python -m pytest -q
   ```

4. Für die Review-Patch-Aufteilung `a7f6dcf` als Basis verwenden. HyperLiQ-Historie und spätere Korrekturen getrennt betrachten (siehe Abschnitt 2); keine Git-Historie umschreiben.
5. Live-Code in Unit-Tests ausschließlich mit Fakes/Mocks prüfen. Keine echten Credentials, keine echten Börsenorders und keine Netzwerkabfrage zur Validierung dieses Hand-offs.
6. Nach Befunden Dokumentation und Tests gemeinsam aktualisieren; bei neuen Testläufen immer Befehl, Laufzeit, Pass/Fail und relevante Einschränkungen festhalten.

## 9. Quellen im Repository

- Aktueller Überblick: [`README.md`](../README.md)
- Historischer 25.11.-Status: [`HANDOFF_AURA_2026-11-25.md`](../HANDOFF_AURA_2026-11-25.md)
- Historischer Implementierungsplan: [`LIVE_IMPLEMENTATION_PLAN.md`](../LIVE_IMPLEMENTATION_PLAN.md)
- HIP-4-Screener und read-only Grenzen: [`live/HYPERLIQUID_SCREENER.md`](../live/HYPERLIQUID_SCREENER.md)
- Ursprung des HyperLiQ-Stands: [Commit `a7f6dcf`](https://github.com/nobody55555/aura-bot/commit/a7f6dcf1c4249cb945d96ba20a78c2509411d250)
