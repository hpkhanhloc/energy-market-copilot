# Energy Market Copilot

A small local tool for one question: *"something odd happened in the Finnish day-ahead price,
what was it and what may have driven it?"* It answers with the numbers, the drivers the evidence
supports, and the ones it does not. Facts come from code. Hypotheses are labelled as hypotheses.
Nothing is called a cause.

## What it is for

**First user:** a market analyst who today opens Fingrid, ENTSO-E and a spreadsheet by hand every
time the desk asks "why was Friday evening so expensive?".

**Use case:** the Finnish **day-ahead (spot) price**. It is the number everyone looks at, the data
is public and complete, and its drivers are well understood (demand, wind, nuclear, imports,
neighbours), so a first version can be judged right or wrong.

**Two investigations work end to end:** **5 Jan 2024 19:00** (1,896 EUR/MWh, a cold snap) and
the night of **16 to 17 Dec 2023** (13 hours at or below 0 EUR/MWh, a windy weekend).

**Left out on purpose:** imbalance and balancing prices (fetched and cached, no drivers yet),
outages, weather as a direct input, intraday, anything beyond Finland's four borders. No cloud,
no auth, no styling.

## Try it

```bash
make setup                      # uv sync + git hooks (needs uv, installs Python 3.14.7)
cp .env.example .env            # add ENTSOE_API_KEY, FINGRID_API_KEY, and an LLM key
make warm                       # download the two demo months into data/cache (~2 min)
make app                        # chat UI at http://localhost:8501
uv run python cli.py investigate 2024-01-05T19:00           # same thing without the UI
make check                      # format, lint, type-check, tests (what CI runs)
```

**Why `make warm`.** The tool does not ship with any data. It downloads market data from two
public APIs (next section) and keeps a copy under `data/cache/`, one file per series per month.
That folder is not in git, so a fresh clone starts empty. Without `make warm`, the first
question you ask about a month fetches it on the spot, and that takes a minute or two (about 20
API calls, the ENTSO-E ones 20 to 40 seconds each). `make warm` does that fetch once, up front,
for December 2023 and January 2024, the months the demo cases live in. After that every demo
question answers in under a second and needs no network. For other months, run
`uv run python scripts/warm_cache.py <start> <end>` or ask about them and wait.

**The LLM is optional.** `COPILOT_MODEL` picks any provider Pydantic AI knows
(`anthropic:claude-sonnet-5`, `openai:gpt-5`, `ollama:...`). Without a key you get the same
report, written by code.

**In the chat.** Three starter chips run the demo without the LLM: the 5 Jan 2024 spike, the
16 Dec 2023 negative night, and a scan of 8 Dec to 8 Jan. Or type *"what happened on 5 Jan 2024
at 19:00"*, *"anything odd in December 2023?"*, then *"the hour before that"* or *"what is
residual load?"*. A scan gives a table; clicking a row opens the investigation.

## How it works

In the chat, one LLM call first reads your words and turns them into a typed request (which
hour, which range, or a question). Then plain code runs the pipeline: fetch data, find the
abnormal hours, check each driver, write the facts. The LLM comes back only at the end, to phrase
those facts and answer follow-ups. Starter chips, table-row clicks and the CLI skip the first
call entirely.

### Two data sources, and what each one is for

Both are named in the brief, and they answer different questions.

**ENTSO-E is the main source.** It is the only one of the two with the day-ahead price, and the
only one with anything across a border. From it: the day-ahead price for Finland and its four
neighbours (SE1, SE3, EE, NO4), load and load forecast, generation by type, the day-ahead wind
forecast, and physical flows on each border. Every driver check reads ENTSO-E first.

**Fingrid is the Finnish grid operator's own real-time measurements.** From it: wind, nuclear,
hydro, total production and consumption, the consumption forecast, and the imbalance price. It
serves three purposes today:

- *Fallback.* When ENTSO-E's generation-by-type feed has no wind or nuclear for the window (a
  failed call, or a gap in the feed), the wind, nuclear and residual-load checks use the Fingrid
  series instead. The chart legend says which one was used.
- *Charts.* The Fingrid series are plotted next to the ENTSO-E ones so the analyst can see
  whether the two agree.
- *Next use case.* The imbalance price is already cached for the imbalance investigation listed
  under "what I would build next".

Everything is hourly, stored in UTC, shown in Helsinki time. A source that is missing (no key,
API down) leaves its columns out and the drivers that need them say "not enough data".

### Spotting an event (plain code)

- **Baseline:** each hour is compared with the same Helsinki hour on recent days of the same
  type (weekday or weekend) over the last four weeks, using a median and a robust spread.
- **An hour is abnormal when any one of these holds:**
  - it sits at least 4 robust standard deviations from that baseline *and* moved at least
    50 EUR/MWh;
  - it is at or below 0 EUR/MWh;
  - it jumped at least 100 EUR/MWh from the hour before, beyond the usual daily shape.
- **Episodes:** neighbouring abnormal hours merge into one episode, ranked by size and length.
- **Any hour:** ask about one and the report says whether it is abnormal before analysing it.
- Full rules, the reason behind each threshold, and the backtest:
  [docs/DETECTION.md](docs/DETECTION.md).

### Checking drivers (plain code)

- **One function per driver.** Each returns the value during the event, its baseline, and one
  of three verdicts: *supports*, *does not support*, *not enough data*.
- **"Supports" means:** the series moved in the direction that would push the price the way it
  went, by a clear margin over its own normal spread.
- **Drivers checked:** day-ahead wind forecast, actual wind, nuclear, load, imports from Sweden,
  neighbouring prices, and residual load (load minus wind minus nuclear).
- **Missing data** becomes "not enough data", never a guess.

### Where the LLM comes in (three calls, all guarded)

The LLM never sees raw data and has no tools. Code writes a short text, the LLM reads it and
returns a typed object, code checks the object. Three calls:

1. **Routing** (chat only). Reads your message plus a little context: today's date, which dates
   have data, the last hour or range you looked at, the last few chat lines. Returns `Scan`,
   `Investigate`, `Ask` or `Reply` with the dates filled in. Code then checks the dates: inside
   the data, not in the future, at most a year per scan (60 days if it needs a download). Bad
   output becomes a plain `Reply`.
2. **Narrative.** Reads the facts text that code wrote: price, baseline, each driver's value and
   verdict. Returns `facts`, `hypotheses`, `insufficient` and a short summary. Code checks that
   every number exists in the facts text and that no causal word ("caused", "because", "due to")
   is used. If a check fails, or there is no LLM key, the code-written narrative is shown.
3. **Follow-up.** Reads the same facts text, the last few chat lines, and your question. Returns
   an answer. Same number and causal-word checks; a failed check gives a fixed safe reply.
   Answers from the model's own knowledge are labelled "general knowledge, not from your data".

Every call logs one line to `data/logs/llm.jsonl`: which call, which guard fired, latency.
No agents, no tool use, so the evidence is the same on every run.

## How I would know it is getting better

**Measured today** (manual runs, not in CI):

- Detector: `make backtest` runs it over all 35 cached months, offline. The two press-verified
  events (5 Jan 2024 spike, 24 Nov 2023 bid error) rank 2nd and 27th of 601 episodes. Changing
  the z threshold from 3 to 5 moves the episode count by 14%, so it is not on a cliff. Against
  the textbook mean-plus-two-std rule, ours catches a 264 EUR/MWh hour the naive rule misses,
  because an earlier spike had inflated the naive std. Details in
  [docs/DETECTION.md](docs/DETECTION.md).
- Routing: `make eval` scores the intent call on a fixed prompt set. 28 prompts, 96%.
- Tests in `tests/` pin both demo investigations, a quiet hour, a synthetic spike with normal
  drivers, and every guard.

**Logged today, not yet counted:** every LLM call writes one line to `data/logs/llm.jsonl`
(which call, which guard fired, whether fallback text was shown, latency). A small script over
that file would give guard rate and latency per call type.

**Not measurable yet, needs new work:**

- Product: does the analyst accept the top hypothesis without opening another tool; time from
  question to written note; share of investigations that end in "not enough data" and which
  series was missing. All need the feedback button and a log of investigations run (next
  steps, item 3).
- Detection accuracy: how many flagged hours are truly odd, and how many odd hours we miss.
  Needs a human-labelled list of hours. We have two confirmed events, not a list.
- Driver verdicts: are *supports* / *does not support* right? Tests check two events. Needs
  20 to 30 events with the expected verdict per driver.
- Cache hit rate: how often data comes from disk instead of the API. Nothing counts it yet.

## What I would build next

1. **Outages driver (ENTSO-E):** the one signal an analyst always checks, and the usual reason
   for "nuclear looks fine but the price is high".
2. **Imbalance price use case:** data already cached from Fingrid; same driver pattern,
   starting with forecast-versus-actual wind and how much backup power Fingrid had to switch
   on.
3. **Feedback capture in the app:** one click per investigation for "right / wrong / missing
   driver". Without it the product metrics above cannot be measured.
