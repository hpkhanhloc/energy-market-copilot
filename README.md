# Energy Market Copilot

A small local tool that takes an energy analyst from *"something odd happened in the Finnish
day-ahead market"* to *"here is what happened, the numbers behind it, and the drivers worth
checking"*. Facts come from code. Hypotheses are labelled as hypotheses. Nothing is called a cause.

## Setup and run

```bash
make setup                      # uv sync + git hooks (needs uv, installs Python 3.14.7)
cp .env.example .env            # add FINGRID_API_KEY, ENTSOE_API_KEY, and an LLM key
make app                        # chat UI at http://localhost:8501
uv run python cli.py scan 2023-12-08 2024-01-08          # list abnormal episodes
uv run python cli.py investigate 2024-01-05T19:00         # explain one hour (Helsinki time)
uv run python cli.py investigate 2024-01-05T19:00 --no-llm --charts out/
make check                      # format, lint, type-check, tests (what CI runs)
make eval                       # score the chat router on 28 golden prompts (real model)
```

The LLM is optional. `COPILOT_MODEL` picks any provider Pydantic AI knows
(`anthropic:claude-sonnet-5`, `openai:gpt-5`, `ollama:...`). Without a key you get the same
report written by code. The ENTSO-E API answers in 20 to 40 seconds per call; the first look at
a new month takes a minute or two, then everything is cached per calendar month under
`data/cache/`. Two demo months (Dec 2023, Jan 2024) are worth warming before a demo:
`uv run python scripts/warm_cache.py 2023-12-01 2024-02-01`.

**Using the chat.** Three starter chips cover the demo (the 1,896 EUR/MWh spike on 5 Jan 2024
19:00, the negative night of 16 Dec 2023, a scan of 8 Dec to 8 Jan); they run plain code, no
LLM. Or type: *"what happened on 5 Jan 2024 at 19:00"*, *"anything odd in December 2023?"*,
then follow up with *"the hour before that"*, *"what is residual load?"*, *"why does the report
say imports do not support?"*. A scan answers with a table; clicking a row opens the
investigation. Off-topic questions get a one-line "I only do the Finnish day-ahead price".

## Who it is for, what it does, what it leaves out

**First user:** a market analyst on the Energy Market Team who today opens Fingrid, ENTSO-E and a
spreadsheet by hand every time the desk asks "why was Friday evening so expensive?".

**Use case:** the Finnish **day-ahead (spot) price**. Picked because it is the number everyone
looks at, the data is public and complete, and its drivers are well understood (demand, wind,
nuclear, imports, neighbours), so a first version can be judged right or wrong. Two investigations
the tool handles end to end: **5 Jan 2024 19:00** (1,896 EUR/MWh, a cold snap) and the night of
**16 to 17 Dec 2023** (13 hours at or below 0 EUR/MWh, a windy weekend).

**Deliberately left out:** imbalance and balancing prices (the data is fetched and cached from
Fingrid, but no drivers yet), outages (ENTSO-E outage feed not wired), weather as a direct input,
intraday, and anything beyond Finland's four borders. No cloud, no auth, no styling.

## Data, signals, and how the system reasons

**Sources:** only the two named in the brief. ENTSO-E gives the day-ahead price for Finland and
SE1, SE3, EE, NO4, actual load and its forecast, generation by type, day-ahead wind forecast, and
physical flows per border. Fingrid gives real-time wind, nuclear, hydro, production, consumption,
consumption forecast and the imbalance price. Everything is resampled to hourly means on a UTC
index and shown in Helsinki time.

**Spotting an event (plain code):** every hour is compared with the same hour of day over the
previous 28 days using a median and MAD (a robust standard deviation). An hour is abnormal when
its robust z is at least 4 *and* it moved at least 50 EUR/MWh; a price at or below zero is always
an event. Adjacent abnormal hours merge into one episode. A user can also ask about any hour; if
it is not abnormal the report says so and analyses it anyway.

**Checking drivers (plain code):** each driver is one function that returns a number, its
baseline, and one of three verdicts: *supports*, *does not support*, *not enough data*. Support
means the series moved in the direction that would push the price the way it went, by robust
z >= 2 or by 20% of baseline. Drivers: day-ahead wind forecast (what the auction actually saw),
actual wind, nuclear, load, imports from Sweden (judged on SE1+SE3 because Estonia often flips
direction and hides a Nordic shortfall in the total), neighbouring prices (regional move versus
Finland alone), and residual load (load minus wind minus nuclear). Missing data becomes "not
enough data", never a guess.

**Where the LLM is used, and where it is not:** the LLM never sees raw data, never picks the
event, never runs a check, and has no tools. It is called three times at most, each with a typed
output and a code guard behind it:

1. *Routing.* Your words plus a small context block (today, the cached date range, the last hour
   and range on screen) become one of `Scan`, `Investigate`, `Ask`, `Reply`. Code re-checks every
   date (60-day cap, data reach, no future). Any model failure becomes a `Reply`.
2. *Narrative.* The deterministic facts text becomes a typed object with separate `facts`,
   `hypotheses` and `insufficient` lists. Every number is checked against the facts; causal words
   ("caused", "because", "due to") are rejected; if either trips, the code-written narrative is
   shown instead.
3. *Follow-up answers.* A question about the report on screen is answered from the facts text
   with the same two guards. Term definitions are labelled "general knowledge, not from your
   data". Hours the report does not cover get "not in the current report, ask me to investigate
   it", never a guess.

Every call writes one line to `data/logs/llm.jsonl` (kind, guard hit, fallback used, latency),
and the sidebar shows the last call. No agents: the evidence has to be identical run to run, or
you cannot measure whether it improves.

## How I would know it is getting better

**Product side:** does the analyst accept the top hypothesis without opening another tool
(thumbs up/down per investigation, and how often they add a driver we missed); time from
question to written note; share of investigations that end in "not enough data" and which
series was missing.

**System side:** a regression set of labelled events with expected verdicts (below); detection
precision on a month of history (flagged hours an analyst agrees with) and recall on known
events; routing accuracy on the golden prompt set (`make eval`: 28 prompts, 96% with
`claude-sonnet-5`, the one miss is a bare "explain the 5th" that should ask back); the share of
LLM calls that hit a guard, read straight from `data/logs/llm.jsonl`; p50 latency per call
(about 2.4 s routing, 6 s narrative); cache hit rate and seconds per investigation.

**Concrete test cases (in `tests/`, run offline on a fixture):**
1. 2024-01-05 19:00 is the top event of Dec 8 to Jan 8; load, Swedish imports and neighbouring
   prices support; nuclear and wind do not.
2. 2023-12-16 19:00 to 12-17 07:00 is detected as a negative-price episode; wind forecast, actual
   wind and residual load support; nuclear does not.
3. A quiet hour returns `flagged = False` and the report says so before analysing it.
4. A synthetic spike with normal drivers everywhere gives zero supporting drivers (no false
   stories).
5. A narrative containing a number that is not in the facts is rejected.
6. A narrative or answer that says "caused" or "because" is rejected.
7. A 61-day scan, a date before the cached range, or a future hour never reaches the data layer.
8. A follow-up answer with an invented number falls back to a fixed safe text.

## What I would build next

1. **Outages driver (ENTSO-E A80/A77):** the one signal an analyst always checks that the tool
   cannot yet, and the most common reason for "nuclear looks fine but the price is high".
2. **Imbalance price use case:** the data is already cached; the same driver pattern applies with
   actual-versus-forecast wind and activated mFRR from Fingrid as the key checks.
3. **Feedback capture in the app:** one click per investigation to store "right / wrong / missing
   driver". Without it the product metrics above cannot be measured.
