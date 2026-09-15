# How a question flows through the code

Four diagrams: the chat loop, fetching data, scanning a range, investigating one hour.
Function names are the real ones, so you can grep them.

## 1. Chat loop (`app.py`, `views.py`)

Starter chips and table-row clicks skip the LLM and go straight to `handle_intent`.

```mermaid
flowchart TD
    U[User types text] --> PI["parse_intent (LLM call 1)<br/>copilot/intent.py"]
    PI --> GI["guard_intent<br/>dates inside data, not future, scan cap"]
    GI -->|Scan| RS["run_scan<br/>views.py"]
    GI -->|Investigate| RI["run_investigate<br/>views.py"]
    GI -->|Ask| AQ["answer_question (LLM call 3)<br/>copilot/chat.py"]
    GI -->|Reply or guard failed| TX[plain text reply]
    C[Starter chip] --> GI
    R[Click table row] --> RI
    RS --> T[events table in chat]
    RI --> REP[report + charts in chat]
    AQ --> TX
```

## 2. Fetching data (`copilot/data/`)

Every fetch, whether for a scan or an investigation, goes through `build_market_frame`.

```mermaid
flowchart TD
    LW["load_window(start, end)<br/>copilot/investigate.py"] --> BF["build_market_frame<br/>copilot/data/frame.py"]
    BF --> J["_jobs: one job per series<br/>13 ENTSO-E + 4 Fingrid"]
    J --> P[ThreadPoolExecutor, all jobs in parallel]
    P --> E["EntsoeSource<br/>price FI + SE1 SE3 EE NO4, load, load forecast,<br/>generation by type, 4 border flows, wind forecast"]
    P --> F["FingridSource<br/>wind, wind forecast,<br/>nuclear, consumption"]
    E --> CR["cached_range<br/>copilot/data/cache.py"]
    F --> CR
    CR --> M{"parquet for this month<br/>on disk and fresh?<br/>(old months never expire,<br/>current month does)"}
    M -->|yes| RD[read data/cache/*.parquet]
    M -->|no| API["call API (XML or JSON),<br/>convert to hourly table,<br/>save as parquet"]
    RD --> JN[join all columns on one hourly UTC index]
    API --> JN
    JN --> MF["MarketFrame<br/>data: hourly table<br/>missing: series that failed"]
```

A job that fails after 3 tries is not an error. Its column is left out and its name goes into
`missing`, so the driver that needs it says "not enough data".

## 3. Scan a range (`scan` in `copilot/investigate.py`)

```mermaid
flowchart TD
    MF[MarketFrame] --> PR["price_fi column only"]
    PR --> SC["score_prices<br/>copilot/detect.py"]
    SC --> BL["baseline per hour:<br/>same Helsinki hour, same day type,<br/>last 4 weeks, median + robust spread"]
    BL --> RU{"any rule hits?<br/>z >= 4 and move >= 50<br/>or price <= 0<br/>or ramp >= 100 from previous hour"}
    RU -->|yes| FL[hour flagged]
    RU -->|no| NF[hour normal]
    FL --> GR["_groups: merge neighbouring<br/>flagged hours into one Event"]
    GR --> RK["rank by severity,<br/>drop events before the asked range"]
    RK --> TOP["top 5 Events"]
```

## 4. Investigate one hour (`investigate_at` in `copilot/investigate.py`)

```mermaid
flowchart TD
    H["hour asked"] --> WF["window_for: 30 days before, 2 after"]
    WF --> LW[load_window, diagram 2]
    LW --> EA["event_at<br/>copilot/detect.py<br/>Event anchored on that hour,<br/>flagged or not"]
    EA --> RA["run_all<br/>copilot/drivers/checks.py"]
    RA --> D1[wind_forecast]
    RA --> D2[wind_actual]
    RA --> D3[nuclear]
    RA --> D4[load]
    RA --> D5[imports]
    RA --> D6[neighbour_prices]
    RA --> D7[residual_load]
    D1 & D2 & D3 & D4 & D5 & D6 & D7 --> CB["compare_to_baseline<br/>copilot/drivers/base.py<br/>value in event vs own same-hour baseline"]
    CB --> V{"verdict"}
    V --> S[supports]
    V --> N[does not support]
    V --> I[not enough data]
    S & N & I --> INV["Investigation<br/>event + frame + DriverResult list"]
    INV --> RF["render_facts<br/>copilot/report.py<br/>markdown with every number"]
    INV --> PL["plots.py charts"]
    RF --> NA["narrate (LLM call 2)<br/>copilot/llm.py"]
    NA --> G{"guards:<br/>every number in facts?<br/>no causal words?"}
    G -->|pass| OUT[LLM narrative shown]
    G -->|fail or no key| FB["fallback_narrative<br/>code-written, shown instead"]
```

`supports` means the series moved in the direction that pushes the price the way it went, by a
clear margin over its own normal spread. Details of the rules and thresholds:
[DETECTION.md](DETECTION.md).
