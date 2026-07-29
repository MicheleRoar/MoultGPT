# Per-field trait-extraction breakdown (real 500-item run)

Source: `results_scored.csv`, main condition, single+combo items only (negative/hallucination items are a separate question, see report.md). Models: mistral-small-latest, mistral-medium-latest. Combo items already contribute one row per sub-field in the source CSV, so per-field counts here merge single- and combo-derived rows for the same field naturally.

**Read the `n` column before the percentages.** Many fields have very few sampled items (some as low as 1) -- a field flagged with n < 3 is not a statistically meaningful per-field estimate, just a single (or handful of) real data point(s). Fields are sorted by correct rate (descending) within each model, matching the earlier pilot's presentation style.


## mistral-small-latest

| Field | n | correct | ambiguous | incorrect | abstained | other | small sample? |
|---|---|---|---|---|---|---|---|
| Resulting named moulting configurations | 12 | 3 | 0 | 0 | 9 | 0 |  |
| Life history style | 21 | 0 | 0 | 0 | 21 | 0 |  |
| Intraspecific variability in moulting mode | 15 | 0 | 0 | 0 | 15 | 0 |  |
| Consumption of exuviae | 14 | 0 | 3 | 0 | 11 | 0 |  |
| Moulting phase | 14 | 0 | 0 | 0 | 14 | 0 |  |
| Observed ontogenetic stage | 12 | 0 | 0 | 0 | 12 | 0 |  |
| Life mode | 12 | 0 | 0 | 0 | 12 | 0 |  |
| Adult stage moulting | 12 | 0 | 3 | 0 | 9 | 0 |  |
| Segment addition mode | 11 | 0 | 0 | 0 | 11 | 0 |  |
| Estimated # moult stages | 11 | 0 | 0 | 0 | 11 | 0 |  |
| # body segments in adult individuals | 11 | 0 | 0 | 0 | 11 | 0 |  |
| Other behaviours associated with moulting | 11 | 0 | 0 | 0 | 11 | 0 |  |
| Post-moult cuticle calcification event | 10 | 0 | 0 | 0 | 10 | 0 |  |
| Observed # total moult stages | 9 | 0 | 0 | 0 | 9 | 0 |  |
| Number of juvenile moults | 9 | 0 | 0 | 0 | 9 | 0 |  |
| Position exuviae found in | 9 | 0 | 0 | 0 | 9 | 0 |  |
| Ontogenetic stage period (in days) | 9 | 0 | 0 | 0 | 9 | 0 |  |
| Location of moulting suture | 9 | 0 | 0 | 0 | 9 | 0 |  |
| Pre-moult period (in days) | 8 | 0 | 3 | 0 | 5 | 0 |  |
| Direction of egress during moulting | 8 | 0 | 3 | 0 | 5 | 0 |  |
| Number of major morphological transitions | 8 | 0 | 0 | 0 | 8 | 0 |  |
| Cephalic suture location | 8 | 0 | 0 | 0 | 8 | 0 |  |
| Average body length (in mm) | 7 | 0 | 0 | 0 | 7 | 0 |  |
| # body segments per moult stage | 4 | 0 | 0 | 0 | 4 | 0 |  |
| Sex | 4 | 0 | 0 | 0 | 4 | 0 |  |
| Reproductive state | 3 | 0 | 0 | 0 | 3 | 0 |  |

## mistral-medium-latest

| Field | n | correct | ambiguous | incorrect | abstained | other | small sample? |
|---|---|---|---|---|---|---|---|
| Resulting named moulting configurations | 12 | 6 | 0 | 0 | 6 | 0 |  |
| Post-moult cuticle calcification event | 10 | 3 | 0 | 0 | 7 | 0 |  |
| Cephalic suture location | 8 | 1 | 0 | 0 | 7 | 0 |  |
| Life history style | 21 | 0 | 0 | 0 | 21 | 0 |  |
| Intraspecific variability in moulting mode | 15 | 0 | 1 | 0 | 14 | 0 |  |
| Consumption of exuviae | 14 | 0 | 4 | 0 | 10 | 0 |  |
| Moulting phase | 14 | 0 | 0 | 0 | 14 | 0 |  |
| Observed ontogenetic stage | 12 | 0 | 4 | 0 | 8 | 0 |  |
| Life mode | 12 | 0 | 0 | 0 | 12 | 0 |  |
| Adult stage moulting | 12 | 0 | 3 | 0 | 9 | 0 |  |
| Segment addition mode | 11 | 0 | 6 | 0 | 5 | 0 |  |
| Estimated # moult stages | 11 | 0 | 0 | 0 | 11 | 0 |  |
| # body segments in adult individuals | 11 | 0 | 0 | 0 | 11 | 0 |  |
| Other behaviours associated with moulting | 11 | 0 | 3 | 0 | 8 | 0 |  |
| Observed # total moult stages | 9 | 0 | 0 | 0 | 9 | 0 |  |
| Number of juvenile moults | 9 | 0 | 0 | 0 | 9 | 0 |  |
| Position exuviae found in | 9 | 0 | 3 | 0 | 6 | 0 |  |
| Ontogenetic stage period (in days) | 9 | 0 | 0 | 0 | 9 | 0 |  |
| Location of moulting suture | 9 | 0 | 3 | 0 | 6 | 0 |  |
| Pre-moult period (in days) | 8 | 0 | 7 | 0 | 1 | 0 |  |
| Direction of egress during moulting | 8 | 0 | 3 | 0 | 5 | 0 |  |
| Number of major morphological transitions | 8 | 0 | 0 | 0 | 8 | 0 |  |
| Average body length (in mm) | 7 | 0 | 0 | 0 | 7 | 0 |  |
| # body segments per moult stage | 4 | 0 | 0 | 0 | 4 | 0 |  |
| Sex | 4 | 0 | 0 | 0 | 4 | 0 |  |
| Reproductive state | 3 | 0 | 0 | 0 | 3 | 0 |  |

## Overall (all fields combined, per model)

| Model | fields | items | correct | ambiguous | incorrect | abstained | fields >=50% correct | fields with 0 correct |
|---|---|---|---|---|---|---|---|---|
| mistral-small-latest | 26 | 261 | 3 (1.1%) | 12 | 0 | 246 | 0/26 | 25/26 |
| mistral-medium-latest | 26 | 261 | 10 (3.8%) | 37 | 0 | 214 | 1/26 | 23/26 |
