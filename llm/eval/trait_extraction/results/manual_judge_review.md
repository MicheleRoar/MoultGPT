# Manual re-review of judge-flagged disagreements (500-item run)

The automated LLM judge (`mistral-medium-latest`) classified **0 of 102**
`disagreement` rows as `incorrect` — every single one came back `ambiguous`.
That is a red flag on its own (a judge that never says "incorrect" is not
discriminating), so before trusting that number this document is an
independent manual re-review of every underlying disagreement, done by
reading the gold value, the model's prediction, and the judge's own stated
reasoning side by side.

The 102 raw rows collapse to **19 distinct (paper, field, model) clusters**
— the rest are the same disagreement recurring across the 5 rotated
question phrasings (`phrasing_round`) and, for combo items, the
`combo_split_ok=False` fallback re-scoring the same answer against each
sub-field. Verdicts below are per cluster, not per row.

## Verdict: judge was too lenient — should be `incorrect`, not `ambiguous` (8/19)

| Paper | Field | Model(s) | Gold | Predicted | Why this looks wrong, not ambiguous |
|---|---|---|---|---|---|
| 6 | Direction of egress during moulting | small, medium, large (all 3) | `Anterior` | "posterior separation [...]" | The model's own answer says the OPPOSITE direction from gold. The judge's justification ("evidence describes both posterior and anterior disarticulation as possible") does not establish that "Anterior" is actually the correct answer among the two — this is the single clearest case of judge over-leniency in the run. |
| 3 | Life history style | large | `direct development` | `Terrestrial` | Different dimension (habitat vs. developmental mode). The judge's own reasoning states *"Evidence supports terrestrial adaptation but not direct development"* and then labels it `ambiguous` anyway — the verdict contradicts the judge's own stated reasoning. |
| 18 | Life mode | large | `surficial` | "Active, feeding immediately after emergence" | Different dimension (activity vs. habitat position). Judge's justification is "does not explicitly contradict" — absence of contradiction is not evidence of support, and the judge applied that weaker bar here. |
| 18 | Reproductive state | large | `Pre-reproductive; Reproductive, partenogenesis` | `Oviparous` | Parthenogenesis (asexual) and oviparity (egg-laying mode) are not the same axis; an oviparous species can reproduce sexually or asexually. Predicting "oviparous" does not confirm or contradict the gold parthenogenesis claim. |
| 5 | Position exuviae found in | medium, large | `Prone` (body posture) | Lists of substrate/geographic locations (deep-water, sediment, inside rocks, ...) | Gold asks about body ORIENTATION when found; the model answered with WHERE (substrate/location), a different question. |
| 20 | Pre-moult period (in days) | medium, large | `14-90` | "8.6 days (mean, larvae, rainy season)" / "Mean 8.6 days; Range 5–16 days" | Predicted range barely overlaps gold (14–16 only); looks like the model surfaced a different, narrower subpopulation statistic (larvae, one season) rather than the paper's overall reported range that MoultDB curators used. |
| 2, 3 | Intraspecific variability in moulting mode | large (both papers) | `0` | "Size-class and reproductive-status dependent variation" / "Greater variation in females" | Same pattern recurring on two independent papers: gold `0` (read as "no variability") vs. a real, specific variability description in the evidence. Either the model is contradicting real gold data, or `0` is coded ambiguously in the source spreadsheet for this field — worth a manual spreadsheet check either way (see Limitations). |
| 21 | Observed ontogenetic stage | medium, large | `Adult; Larva 5; Larva 6; Larva 7` (specific observed stages for this record) | "eight stadia" / "eight stadia (instars)" (a TOTAL stage count for the species) | Category mismatch: the field asks which stage(s) were observed in THIS annotation record, the model answers with the species' total developmental stage count — a real fact, just not the one asked for. The judge's own reasoning ("does not explicitly confirm the system's phrasing as the observed stage") supports `incorrect`, not `ambiguous`. |

## Verdict: agree, genuinely ambiguous / plausibly correct (7/19)

These are real paraphrase or tolerance gaps in the automated string-matching
scorer, not model reasoning errors — good candidates for a future scoring
improvement (morphological stemming, numeric range containment) rather than
evidence against the model.

| Paper | Field | Model(s) | Gold | Predicted | Why this is plausibly correct |
|---|---|---|---|---|---|
| 46, 8 | Segment addition mode | medium, large | `hemianamorphosis` | `hemianamorphic` / "Intermittent segment addition..." | Adjectival form of the same noun / a descriptive restatement of the same process. |
| 4 | Pre-moult period (in days) | small, medium, large | `5-13` | `12` / `12 days` / "9.5 ± 2.98 days (min 5, max 13)" | `12` is within the gold range; large's answer restates the same mean/range as gold verbatim. A numeric-range-containment matcher would score this correct. |
| 44 | Other behaviours associated with moulting | medium | `mass moulting` | `en masse` | Same concept; French/English phrasing variant of the same term. |
| 20 | Consumption of exuviae | small, medium | `yes` | "They devoured their exuviae" / "After moulting, they devoured their exuviae" | Direct paraphrase of "yes." |
| 20 | Adult stage moulting | small, medium | `yes` | "Adults of S. assiniensis moult [...]" | Confirms the core claim (adults do moult) for at least one taxon in the paper. |
| 8 | Position exuviae found in | large | `Prone` | "dorsal-up attitude" | Anatomically equivalent: an organism lying prone (front/ventral side down) has its dorsal side up — this is the same posture in more precise terminology, not a different one. |
| 26 | Consumption of exuviae (combo) | large | `partial` | `True` | Directionally consistent (some consumption occurs) even though it loses the "partial, not universal" nuance gold captures — reasonable partial credit. |

## Verdict: requires domain-expert review, not confidently adjudicated here (2/19)

| Paper | Field | Model(s) | Gold | Predicted | Why this needs a specialist |
|---|---|---|---|---|---|
| 7 | Location of moulting suture | medium, large | `cephalon; dorsal; ventral; cephalothoracic joint` (repeated) | "facial and rostral sutures" | "Facial suture" and "rostral suture" are real, specific trilobite cephalic anatomy terms, distinct from the gold list's terms — whether they denote the same structures requires trilobite-specific morphological expertise beyond what this review can determine. |
| 7 | Other behaviours associated with moulting | large | `enrollment` | "dorsal flexure (tipping over of the lower cephalic unit)" | "Enrollment" (rolling into a defensive ball) and "dorsal flexure" (a specific tipping motion) may or may not describe the same behaviour in trilobite ecdysis literature. |

## Bottom line

- The judge's 0% `incorrect` rate on this run should NOT be read as "the
  model was never wrong" — roughly 8/19 (42%) of the underlying disagreement
  clusters manually reviewed here look like real errors that a stricter
  judge (or a domain expert) would likely have called `incorrect`.
- At the same time, roughly 7/19 (37%) are genuine scorer-tolerance gaps
  (paraphrase, adjectival form, numeric range containment) where the
  *automated matching*, not the model, is what's wrong — these should
  eventually be `correct`, not `disagreement` at all.
- Net effect: the reported "0 incorrect, all ambiguous" number understates
  real errors and simultaneously understates real correct answers, in
  different rows. Neither the raw `disagreement` count nor the judge's
  ambiguous/incorrect split should be reported as authoritative without
  this caveat.
- This is a real, if small-n (19 clusters), independent check on the
  automated LLM-judge methodology — exactly the kind of limitation this
  project's evaluation work has tried to surface rather than hide
  throughout.
