# Recall-loss diagnostic: is sentence-selection discarding evidence the model would use?

For every main-condition item where `mistral-large-latest` did NOT get it right (label `abstained` or `disagreement`, item_type single/combo), this checks whether each gold value appears (normalized substring match, a looser check than the paper's real scorer) in the paper's full text versus in the ~13-20 sentences actually selected and shown to the model. Source: `results_scored.csv`.

Rows checked: 231 (of 231 non-correct rows; 0 skipped for missing paper text, 0 skipped for empty/degenerate gold value).

| Category | n | % of checked | Interpretation |
|---|---|---|---|
| Gold value NOT in full text (verbatim) | 165 | 71.4% | Expected for inferred/paraphrased MoultDB annotations (e.g. "direct development"); not a selection bug. |
| Gold value in full text, NOT in selected evidence | 38 | 16.5% | **Sentence-selection dropped it before the model ever saw it** -- a real, fixable recall bottleneck. |
| Gold value in full text AND in selected evidence | 28 | 12.1% | The model had the evidence and still got it wrong/abstained -- a genuine model-capability or prompt-format issue, not retrieval. |

## Examples: lost in sentence selection

- paper 4, field "Pre-moult period (in days)": gold=`5-13`, label=disagreement, predicted=`9.5 ± 2.98 days (min 5, max 13)`
- paper 4, field "Pre-moult period (in days)": gold=`5-13`, label=disagreement, predicted=`9.5 ± 2.98 days (min 5, max 13)`
- paper 4, field "Pre-moult period (in days)": gold=`5-13`, label=disagreement, predicted=`9.5 ± 2.98 days (min 5, max 13)`
- paper 18, field "Reproductive state": gold=`Pre-reproductive; Reproductive, partenogenesis`, label=abstained, predicted=`Not mentioned`
- paper 18, field "Average body length (in mm)": gold=`3.5`, label=abstained, predicted=`Not mentioned`
- paper 18, field "Reproductive state": gold=`Pre-reproductive; Reproductive, partenogenesis`, label=disagreement, predicted=`Oviparous, short between-laying period`
- paper 18, field "Average body length (in mm)": gold=`3.5`, label=abstained, predicted=`Not mentioned`
- paper 18, field "Reproductive state": gold=`Pre-reproductive; Reproductive, partenogenesis`, label=disagreement, predicted=`Oviparous`
- paper 18, field "Average body length (in mm)": gold=`3.5`, label=abstained, predicted=`Not mentioned`
- paper 18, field "Observed ontogenetic stage": gold=`Adult; Larva 2; Larva 3; Larva 4; Larva 5; Larva 6`, label=abstained, predicted=`Observed ontogenetic stage: juvenile (stage I to IV); Life history style: Not mentioned`
- paper 5, field "Direction of egress during moulting": gold=`Anterior`, label=abstained, predicted=`Not mentioned`
- paper 5, field "Direction of egress during moulting": gold=`Anterior`, label=abstained, predicted=`Not mentioned`
- paper 5, field "Direction of egress during moulting": gold=`Anterior`, label=abstained, predicted=`Not mentioned`
- paper 5, field "Observed ontogenetic stage": gold=`Holaspid`, label=abstained, predicted=`Observed ontogenetic stage: Not mentioned, Direction of egress during moulting: Not mentioned`
- paper 5, field "Direction of egress during moulting": gold=`Anterior`, label=abstained, predicted=`Observed ontogenetic stage: Not mentioned, Direction of egress during moulting: Not mentioned`

## Examples: evidence present, model still wrong

- paper 21, field "Observed ontogenetic stage": gold=`Adult; Larva 5; Larva 6; Larva 7`, label=disagreement, predicted=`eight stadia (instars)`
- paper 21, field "Observed ontogenetic stage": gold=`Adult; Larva 5; Larva 6; Larva 7`, label=disagreement, predicted=`eight stadia (instars)`
- paper 21, field "Observed ontogenetic stage": gold=`Adult; Larva 5; Larva 6; Larva 7`, label=disagreement, predicted=`eight stadia (instars)`
- paper 21, field "Observed ontogenetic stage": gold=`Adult; Larva 5; Larva 6; Larva 7`, label=disagreement, predicted=`eight stadia (instars)`
- paper 4, field "Moulting phase": gold=`biphasic`, label=abstained, predicted=`Not mentioned`
- paper 20, field "Other behaviours associated with moulting": gold=`Moulting nest`, label=abstained, predicted=`Not mentioned`
- paper 24, field "Sex": gold=`Female`, label=abstained, predicted=`Not mentioned`
- paper 24, field "Sex": gold=`Female`, label=abstained, predicted=`Not mentioned`

## Examples: gold not verbatim in full text

- paper 2, field "Observed ontogenetic stage": gold=`sexually mature adult stage`, label=abstained, predicted=`Not mentioned`
- paper 2, field "Intraspecific variability in moulting mode": gold=`0`, label=abstained, predicted=`Not mentioned`
- paper 2, field "Consumption of exuviae": gold=`T`, label=abstained, predicted=`Not mentioned`
- paper 2, field "Post-moult cuticle calcification event": gold=`T`, label=abstained, predicted=`Not mentioned`
- paper 2, field "Intraspecific variability in moulting mode": gold=`0`, label=disagreement, predicted=`Size-class and reproductive-status dependent variation`
- paper 2, field "Consumption of exuviae": gold=`T`, label=abstained, predicted=`Not mentioned`
- paper 2, field "Post-moult cuticle calcification event": gold=`T`, label=abstained, predicted=`Not mentioned`
- paper 2, field "Intraspecific variability in moulting mode": gold=`0`, label=disagreement, predicted=`Size-class and reproductive-status dependent variation`
