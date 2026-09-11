# How events are detected, and how that was checked

Companion to the README. Everything here is plain code in `copilot/detect.py`,
`copilot/baseline.py` and `copilot/backtest.py`.

## The baseline

Every hour is compared with the same *Helsinki local* hour on recent days of the same type,
weekday or weekend. The comparison uses a median and MAD (a robust standard deviation) so one
earlier spike does not hide the next one.

- Local hours, not UTC: the daily price shape follows the wall clock. A UTC bucket shifts by an
  hour at every DST switch and would print a month of fake events twice a year.
- Weekday and weekend apart: Sunday midday is nothing like Tuesday midday.
- Lookback 28 days: exactly four weeks, so always 20 weekdays and 8 weekend days. 30 days would
  hold 8 or 10 weekend days depending on where you start.
- Fewer than 7 same-type days of history (3 for weekends): no baseline, and the report says so
  instead of printing a number.

### The formulas (`copilot/baseline.py`)

For one hour, take the values of the same local hour on the prior same-type days in the
lookback. Call them `x_1 ... x_n`.

```
median  = middle value of x_1 ... x_n
MAD     = median of |x_i - median|            (median absolute deviation)
sigma   = 1.4826 * MAD                        (MAD scaled to a std-dev equivalent)
z       = (value - median) / max(sigma, floor)
```

In words:

- **median** is the baseline: "normal price for this hour". The middle value, so one extreme day
  does not drag it.
- **MAD** is the spread: how far a typical day sits from the median. Also a median, so one
  extreme day does not inflate it.
- **sigma** is MAD rescaled. For bell-shaped data, 1.4826 × MAD equals the standard deviation,
  so z reads like a textbook z-score.
- **z** is how many spreads the hour sits from normal. z = 0 is normal, z = +4 is four spreads
  above, negative is below.
- **floor** is the smallest spread we will divide by. A very quiet hour can have a spread near
  zero, and then a 6 EUR move becomes z = 60. The floor is a quarter of the typical spread of
  the whole series (median sigma over all hours in the window), never below 1. On the demo month
  the typical price spread is about 43 EUR/MWh, so the floor is about 11. A fixed floor of 1
  was tried first and turned an ordinary 60 EUR/MWh drop into z = -61.

Why median and MAD instead of mean and standard deviation: mean and std count every value with
full weight, so a single 1,896 EUR/MWh hour inflates std for the next four weeks and hides a
264 EUR/MWh hour behind it. Median and MAD ignore that one hour. The backtest below shows the
difference on real data.

The same function scores every driver series (wind, load, nuclear...), so "wind z = -2.1" in a
report means the same thing: wind sat 2.1 of its own typical spreads below its usual level for
that hour.

## The rules

Every hour gets a baseline (normal price for that hour) and a z (how many usual fluctuations it
sits from that baseline). An hour is abnormal if **any one** of three rules hits.

**Rule 1: far from normal.** The hour is at least 4 usual fluctuations from its baseline
(z >= 4 or z <= -4), *and* the move is big in euros too.

- Spike (price above normal): the move must be at least 50 EUR/MWh.
- Crash (price below normal): 50 EUR/MWh also works, *or* the price lost at least half its
  baseline and at least 10 EUR/MWh. Why the second option: a spike can go up forever, but a
  crash cannot go below zero. 40 to 3 EUR/MWh is only a 37 EUR move, yet it is a 93% collapse.
  The 10 EUR minimum stops "half the baseline" from meaning "half of nothing" when normal is
  already tiny.

**Rule 2: zero or below.** Price at or below 0 EUR/MWh. Always flagged, even with no history.
Negative prices are always worth explaining.

**Rule 3: sudden jump.** The price moved at least 100 EUR/MWh from the previous hour, *more than
the baseline itself moves* between those two hours, *and* the move took the price further away
from normal.

- "More than the baseline moves": every evening the baseline rises, say 40 to 90. A price going
  40 to 90 is not a jump, it follows the shape. 40 to 200 is.
- "Further away from normal": the hour after a spike falls just as steeply, but that is the
  return to normal, so it is not a second event.
- This rule catches jumps that Rule 1 misses, for example 10 to 150 EUR/MWh when both hours are
  still inside their own usual fluctuation.

**Episodes.** Flagged hours next to each other merge into one episode. One normal hour in
between does not break the episode. Episodes are ranked by peak |z| times the square root of
the length in hours, so a long event beats a one-hour blip but a violent short one still ranks
high. Every report also states the steepest hour-to-hour move in the event's direction, for
example +696 EUR/MWh from 15:00 to 16:00 on the cold-snap day.

## Backtest: `make backtest`

Runs the detector over every cached month of Finnish price (October 2023 to August 2026, 25,584
hours, 11 seconds, parquet only). Four things to read before trusting the thresholds.

**Events per month.** 601 episodes, median 15 a month, 13.7% of hours abnormal. That is a lot, and
it is honest: Finland since 2024 runs a low-price regime with near-zero baselines and frequent
jumps to 150 to 400 EUR/MWh. The tool leans on ranking rather than on a short list.

**Known events in the ranked list.**

| Event | Rank of 601 |
|---|---|
| 2024-01-05 cold snap, 1,896 EUR/MWh | 2 |
| 2023-11-24 bid error, -500 EUR/MWh | 27 |
| 2023-12-17 windy night, 13 h at or below 0 | 232 |

Rank 1 is a 39-hour September 2024 episode at 393 EUR/MWh against an 8 EUR/MWh baseline. The windy
night ranks low because a night at zero in a low-price regime has a small z. Inside its own scan
window it still shows.

**Threshold sweep.** Episodes for z 3, 4, 5 with a 50 EUR/MWh gate: 663, 601, 568. Absolute gate
30, 50, 80 EUR/MWh at z 4: 632, 601, 539. Ramp gate 80, 100, 150: 620, 601, 591. A gentle slope, no
cliff, so z 4 / 50 EUR/MWh is not a lucky pick.

**Against the textbook rule** (same UTC hour, trailing 30-day mean, flag beyond 2 std): 980 hours
flagged by both, 2,434 only by us, 927 only by the naive rule. Hours only we flag include 264
EUR/MWh against an 8 EUR/MWh baseline: an earlier spike had inflated the naive std until nothing
looked odd. Hours only the naive rule flags are 49 to 69 EUR/MWh on a near-zero baseline, real but
under the 50 EUR/MWh gate. That is the case for median and MAD.

## Known events and where they come from

The assignment names no events. These were found by scanning the data, then checked against the
public record where one exists.

- **2024-01-05 19:00, 1,896 EUR/MWh.** Press-verified. Day average 890.54 EUR/MWh, a Finnish record
  ([Yle](https://yle.fi/a/74-20067868)). The government asked people to save electricity that
  Friday ([Finnish Government](https://valtioneuvosto.fi/en/-/1410877/save-electricity-on-friday-spot-prices-are-up-to-20-times-higher-than-normal)).
  Fingrid reported the winter's record consumption
  ([Fingrid](https://www.fingrid.fi/en/news/news/2024/cold-snap-leads-to-record-electricity-consumption-for-this-winter/)).
- **2023-11-24 15:00 to 23:00, -500 EUR/MWh.** Press-verified. A trader's kW-to-MW bid error sold
  5,787 MW that did not exist; nine hours at the Nord Pool floor
  ([Fingrid](https://www.fingrid.fi/en/news/news/2023/a-peculiar-situation-in-the-electricity-market-on-friday---the-price-does-not-guide-production-and-consumption-correctly/),
  [Nord Pool](https://www.nordpoolgroup.com/en/trading/Operational-Message-List/2023/11/regarding-market-results-in-finland-for-the-delivery-day-24.11.2023-20231123155100/),
  [Montel](https://montelnews.com/news/1531760/bidding-error-sends-finnish-spot-to-eur--500mwh-price-floor)).
  Useful because it is a market error, not weather: the detector must catch the shape without
  knowing the story.
- **2023-12-16 19:00 to 12-17 07:00, 13 h at or below 0 EUR/MWh.** Data-verified only. No news
  item found; 2023 had 467 such hours, so one windy night is routine
  ([Finnish Energy, 2023 statistics](https://energia.fi/wp-content/uploads/2024/01/Electricity-price-statistics-2023.pdf)).
  Backed by Fingrid actual and forecast wind in the cache, which the driver checks mark as
  supporting.
