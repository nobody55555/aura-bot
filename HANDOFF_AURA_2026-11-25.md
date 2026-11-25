# Aura-Bot-Meilenstein: Safety/Risk und HIP-4-Screener

- **Stand:** 25. November 2026
- **Arbeitsbranch:** `handoff/aura-bot-2026-11-25`
- **Ausgangspunkt:** `feature/live-foundation` bei `7353e69b1ef5cc958e335040dadf70d9e1305eea`

## Was dieser Meilenstein festhält

Dieser Branch bündelt den aktuellen Arbeitsstand des Python-Aura-Bots: ein BTC/USDT-Projekt mit Paper- und Live-Runtime, Signal-/Risiko-/Order-Komponenten und einem separaten, strikt lesenden Hyperliquid-Testnet-HIP-4-Screener. Das Ziel dieser Übergabe ist, die Sicherheits- und Analysearbeit fortsetzen zu können, ohne lange Testgespräche erneut lesen zu müssen. Sie ist **kein** Leistungsnachweis und **keine** Freigabe zum Handel mit echtem Geld.

Im Runtime-Pfad wurden Fail-closed-Prüfungen für Equity, Uhrzeit, veraltete Daten, Kill-Switch, Wiederaufnahme und Order-Schutz ergänzt beziehungsweise erweitert. Paper ist der Standard. Der Live-Orderpfad bleibt in `main.py` ausdrücklich gesperrt. Daneben parst `live/hyperliquid_screener.py` aufgezeichnete HIP-4-Daten offline und deterministisch mit `Decimal`; `live/hyperliquid_testnet.py` ist davon getrennt und erlaubt nur explizite, lesende Testnet-Info-Abfragen. Der Screener ist nicht in die Aura-Handelsruntime verdrahtet. Die Risk-/Runtime-Berechnungen verwenden dagegen weiterhin `float`; „Decimal“ gilt nicht pauschal für den gesamten Aura-Bot.

## Validiert durch Codeprüfung und Offline-Tests

- **Paper-Default:** `Settings.mode` fällt ohne `AURA_MODE` auf `paper` zurück. Der isolierte Default-Check unten wurde mit entferntem `AURA_MODE` ausgeführt und bestätigte `paper`.
- **Live-Order-Gate:** Ein neuer Offline-Regressionstest führt ein gültiges Live-Kaufsignal durch die vorgelagerten Prüfungen und bestätigt, dass `OrderManager` nicht aufgerufen wird; stattdessen wird „Live order blocked“ gemeldet. Der Test verwendet ausschließlich Fakes und erzeugt keine Brokerverbindung oder Order.
- **Fail-closed- und Lifecycle-Funktion:** Tests prüfen unter anderem fehlende/ungültige Equity und Uhrzeit, persistente Halts, Operator-Resume-Preflight, Reconciliation sowie Order- und Schutzorder-Verhalten. Das ist funktionale Evidenz für die getesteten Pfade, kein Nachweis realer Börsenfills oder Profitabilität.
- **HIP-4-Funktionen:** Offline-Tests prüfen Metadaten/Expiry, Filter, Quote-Fallbacks, Parität, Orderbuch-Tiefe, fehlende Werte und ungültige Inputs. Testnet-Transporttests verwenden gemockte HTTP-Verbindungen; es wurden bei dieser Übergabe **keine** Testnet- oder Mainnet-Abfragen ausgeführt.
- Die Tests sind deterministisch und offline; sie beweisen weder Marktverhalten noch Ausführungsqualität.

### Exakte aktuelle Testläufe

| Befehl | Umfang | Ergebnis |
|---|---|---|
| `python -m pytest -q tests/test_safety.py tests/test_order_manager.py tests/test_main_safety.py tests/test_risk.py` | Safety, Order-Lifecycle, Runtime-Gates, Risk | **81 passed in 1.77s** |
| `python -m pytest -q tests/test_hyperliquid_screener.py tests/test_hyperliquid_testnet.py` | HIP-4-Offline-Screener und gemockter read-only Testnet-Client | **89 passed in 0.14s** |
| `python -m pytest -q` | Gesamte Aura-Suite | **170 passed in 1.97s** |
| `python -m pytest -q tests/test_main_safety.py::test_live_buy_signal_is_blocked_before_order_manager` | Isolierter neuer Live-Order-Gate-Test | **1 passed in 0.90s** |
| `env -u AURA_MODE python -c 'from config import Settings; assert Settings().mode == "paper", Settings().mode; print("default-mode guard: paper")'` | Default ohne Umgebungs-Override | Ausgabe: `default-mode guard: paper`; Exit-Code 0 |

Die gezielten Suites überschneiden sich mit der Gesamtsuite; ihre Counts dürfen daher nicht addiert werden. Vor dieser Übergabe waren in den vorangegangenen Arbeitsschritten **169/169** Gesamttests und **80/80** Safety/Runtime-Tests berichtet worden. Eine frühere, kleinere Vor-Erweiterungs-Suite hatte **119/119 in 1.67s**. Diese historischen Zahlen sind nicht dieselbe Testsuite wie die aktuelle; der eine zusätzliche Guard-Test erklärt den Wechsel von 169 auf 170.

Beim Entwickeln des neuen Guard-Tests schlug zunächst ein Entwurf mit einer unrealistisch alten Fake-Uhr fehl, weil der Safety-Code den Zeitstempel korrekt ablehnte. Der Test-Fixture-Zeitpunkt wurde auf eine plausible Zeit verschoben; der abschließende isolierte Test, die fokussierten Suites und die Gesamtsuite liefen wie oben bestanden. Am Anwendungscode wurde dafür nichts geändert.

## Risikowerte und tatsächliche Semantik — nicht empirisch validiert

Die folgenden Werte stehen derzeit als Konfiguration/Codegrenzen im Projekt. Sie sind **Vorschläge beziehungsweise Implementierungsparameter, keine von Andreas bestätigte Risikofreigabe und keine durch Aura-Backtests validierten Limits**:

- `risk_per_trade` hat aktuell den Default 1 %. `RiskManager.position_size()` berechnet die Stückzahl als Risikobudget geteilt durch den absoluten Abstand zwischen Entry und Stop, begrenzt zusätzlich durch `max_position_pct` (derzeit 25 % des Equity-Werts).
- `main.py` setzt für die Größenberechnung beim Kauf `stop = letzter Kurs - 2 × ATR`.
- `OrderManager` legt den tatsächlichen Long-Schutzstop standardmäßig **2 % unter dem Entry-Fill** an (Short: 2 % darüber); Take-Profit ist standardmäßig 6 % vom Entry entfernt. Dieser Schutzstop wird nicht aus dem ATR-Stop des Sizing-Pfads übernommen. Damit können berechnetes Positionsrisiko und tatsächlicher Schutzabstand auseinanderlaufen; vor einer Performance- oder Risikobewertung müssen beide Pfade vereinheitlicht und getestet werden.
- `daily_loss_limit` steht auf 2 %, `max_drawdown` auf 5 %; die Codevalidierung verhindert eine Lockerung über diese Caps. Dass die Grenzwerte im Code existieren und Grenzfalltests bestehen, beweist **nicht**, dass sie sinnvoll, genehmigt oder in realen Marktbedingungen wirksam sind.

Es gibt **keine Aura-Backtest- oder Trade-Performance-Evidenz**, die eine Verlustobergrenze von 1 % je Trade, 2 % täglichen Verlust oder 5 % maximalen Drawdown belegt. 170 bestandene Tests sind ausschließlich funktionale Evidenz.

## Kritische offene Equity-Lücke im PaperBroker

`PaperBroker` verbucht den USDT-Cashbestand, führt aber keine BTC-Bestände und keinen Marktwert beziehungsweise unrealisierte BTC-P&L. Der Runtime-Equity-Check `_checked_total_equity()` liest ausschließlich `balances["USDT"]["total"]`; `SafetyMonitor` und `RiskManager` erhalten folglich USDT-only Equity. Bei offener BTC-Exponierung bilden Tagesverlust- und Drawdown-Prüfungen damit nicht das vollständige Konto ab. Paper-Ergebnisse können so keine belastbare Aussage über die vorgeschlagenen Risikolimits liefern.

**Empfohlene Reihenfolge vor jeder Performanceaussage:** (1) Paper-Bestände und unrealisiertes P&L markieren und in Equity einbeziehen; (2) Sizing-Stop und tatsächlich platzierte Schutzorder angleichen; (3) Gebühren, Spread, Slippage, Gaps, Teilfills, überlappende Positionen und Tages-/Zeitzonenregeln in einem deterministischen Offline-Backtest abbilden; (4) erst danach Out-of-Sample und die von Andreas festgelegten Paper-/Testnet-Green-Kriterien auswerten. Bis diese Nachweise vorliegen, bleibt Live-Orderplatzierung gesperrt.

## 100-Trade-Szenario: was vorhanden ist und was fehlt

In den Aura-Dateien liegt kein reproduzierbarer 100-Trade-Backtest mit benannter Trade-Sequenz oder vollständigem Parametersatz. Es werden daher **keine** Aura-Winrate, Rendite oder Drawdownzahl aus einem vermeintlichen 100-Trade-Szenario behauptet.

Die separat liegende PaperDesk-Dokumentation nennt mindestens 100 geschlossene Trades als eine ihrer eigenen Stichprobenhürden. Das ist eine Mindestanzahl, kein Aura-Test und kein Ergebnis für genau 100 Trades. Für eine zukünftige Aura-Analyse müssen mindestens Startkapital, Datenzeitraum und Symbol, genaue Trade-/R-Ergebnisse oder Signale, Compounding, Stop-/Fill-Semantik, gleichzeitige Positionen, Gebühren/Spread/Slippage, Gap-Behandlung, Tagesreset/Zeitzone und Equity-/Drawdown-Messung explizit festgelegt werden. Die derzeitige Differenz zwischen 2×ATR-Sizing-Stop und 2-%-OrderManager-Stop muss vorher aufgelöst werden.

## PaperDesk-Ergebnisse — separates Projekt, nicht übertragbar

Die folgenden Zahlen stammen ausschließlich aus `/workspace/hyperliquid-paperdesk/results/direction_comparison.json`, nicht aus Aura. Die vorhandene PaperDesk-README beschreibt einen historischen 5-Minuten-Paper-Prototyp über 6 Märkte, 17.354 Tage Daten und jeweils 4,999 Kerzen je Markt. Annahmen dort: 100 USD Startkapital, 0.25 % rechnerisches Risiko je Trade, höchstens fünf offene Positionen, höchstens 1× Bruttoexposure, Stop-Abstand 1.5× ATR, Ziel 1.5R sowie 5 bp Gebühren und 5 bp Slippage je Seite. Funding, Marktimpact, Liquidationen und News sind nicht modelliert; ein Stop ist ausdrücklich keine harte Verlustobergrenze.

| PaperDesk-Variante | Geschlossene Trades | Netto-P&L | Profit Factor | Max. Drawdown | Endkapital |
|---|---:|---:|---:|---:|---:|
| Trend-Breakout | 1,526 | −64.39 USD | 0.452 | 64.56 % | 35.61 USD |
| Umgekehrte Breakout-Richtung | 1,771 | −54.79 USD | 0.581 | 55.38 % | 45.21 USD |

Beide PaperDesk-Berichte lauten `NOT_GREEN`; sie verfehlen unter anderem positive Nettoerträge, Profit-Factor 1.20 und max. 10 % Drawdown sowie die geforderte 90-Tage-Historie, unabhängigen Out-of-Sample-Nachweis und 30 Tage Forward-Paper-Trading. Das sind **PaperDesk-only**-Resultate und können weder auf Aura übertragen noch als Beleg für Aura-Risikowerte oder als Live-Freigabe genutzt werden. Die PaperDesk-Dateien selbst gehören nicht zu diesem Branch.

## HIP-4-Screener: nächste technische Hinweise

- Die Analyse nimmt bereits aufgezeichnete Daten entgegen und bleibt offline; fehlende Quote-/Book-Werte bleiben unbekannt (`None`), statt stillschweigend geraten zu werden. `markPx` ist keine Wahrscheinlichkeit und kein Fallback für fehlendes `midPx`/Book-Mid.
- Der getrennte HTTP-Client ist auf kanonische Hyperliquid-Testnet-Endpoints und erlaubte read-only Info-Payloads beschränkt. Setze `HL_ENV=testnet` explizit, wenn eine künftige, ausdrücklich autorisierte Abfrage vorgesehen ist; keine Signier-, Wallet- oder Order-Funktion gehört in diesen Pfad.
- Bei wiederkehrenden `priceBinary`-Märkten können Outcome-IDs mit dem Expiry-Zyklus wechseln (bekannte Beobachtung: 03:00 UTC-Kohorten). Änderungen daher nach Expiry einordnen, nicht pauschal als instabile IDs interpretieren.

## Fortsetzen und Tests ausführen

Im geklonten Repository:

```bash
git fetch origin handoff/aura-bot-2026-11-25
git switch --track origin/handoff/aura-bot-2026-11-25
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Fokussierte Läufe stehen in der Testtabelle oben. Das Ausführen der App ist **kein** Backtest und liefert keinen Performance-Nachweis. Für einen reinen Paper-Start (mit öffentlichem OHLCV-Datenabruf, aber ohne echte Orders):

```bash
AURA_MODE=paper python main.py
```

Nicht `AURA_MODE=live` verwenden, um dieses Handoff zu validieren. Der Live-Orderpfad bleibt gesperrt, aber der Live-Modus baut einen nicht-sandboxed CCXT-Broker auf und kann Start-/Reconciliation-Aufrufe auslösen. Es wurden hier weder Live- noch Testnet-Aufrufe oder Trades ausgeführt. Keine Änderung an dieser Sperre vor Erfüllung und ausdrücklicher Prüfung von Andreas’ bestehenden Paper-/Testnet-Green-Kriterien; diese Tests allein erfüllen sie nicht.

## Dateien im Meilenstein

- **Safety/Risk/Runtime:** `.env.example`, `config.py`, `risk.py`, `main.py`, `live/exceptions.py`, `live/order_manager.py`, `live/safety.py`, `live/broker/base.py`, `live/broker/ccxt_broker.py`, `live/broker/paper.py`, `requirements.txt`, relevante Runtime-/Risk-Tests.
- **HIP-4:** `live/hyperliquid_testnet.py`, `live/hyperliquid_screener.py`, `live/HYPERLIQUID_SCREENER.md`, zugehörige Offline-Tests und die HIP-4-Ergänzung in `README.md`.
- **Handoff:** dieses Dokument.

`__pycache__/`-Verzeichnisse und `*.pyc` sind vom Python-Interpreter erzeugte Bytecode-Caches, keine Quelldateien. Sie werden bei Bedarf regeneriert und sind für Git nicht erforderlich; sie sind aus diesem Branch ausgeschlossen. Die getrennte PaperDesk-Repository-/Ergebnisdatei ist ebenfalls nicht Teil des Stagings.
