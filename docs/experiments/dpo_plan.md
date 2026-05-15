# DPO Plan

## Scope

DPO is the next Project 1 alignment stage, but DPO training is not implemented yet.

The goal is to compare:

```text
SFT-only vs SFT+DPO
```

using preference pairs that reward safer, more complete, and more professional assistant behavior.

## Optimization Goals

DPO should prefer responses that:

- improve `steps_plus_email` completeness by requiring both process steps and a complete email draft;
- escalate visa, CPT, OPT, medical, insurance, legal, and academic-risk questions to the correct official office or qualified professional;
- provide complete numbered steps when the user needs process guidance;
- avoid overconfident or absolute claims;
- write concise and professional email drafts;
- avoid vague responses;
- avoid unnecessary notes, explanations, second drafts, or prompt-role leakage.

## Data Format

Each pair should contain:

```json
{
  "id": "dpo_example_001",
  "category": "course_enrollment",
  "risk_level": "high",
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

The `chosen` answer should be safer, more complete, and more actionable.

The `rejected` answer should contain at least one flaw, such as being too short, unsafe, vague, overconfident, missing the official office, missing steps, or adding unwanted commentary after an email draft.

## Coverage Targets

The current DPO seed set contains 50 pairs with this target distribution:

- 10 `steps_plus_email` completeness pairs;
- 10 CPT / OPT / visa safe-escalation pairs;
- 10 medical / health-insurance safe-escalation pairs;
- 10 housing / mailroom / housing-office escalation pairs;
- 10 email-quality preference pairs.

The pair IDs use these prefixes:

- `dpo_steps_email_*`
- `dpo_visa_safe_*`
- `dpo_medical_safe_*`
- `dpo_housing_safe_*`
- `dpo_email_quality_*`

## When To Expand Data

Expand DPO data before implementing or running DPO training if any of these are true:

- one preference type has fewer than 10 reviewed examples;
- chosen/rejected pairs are too obvious and do not reflect realistic model mistakes;
- qualitative comparison shows repeated failures not covered by the DPO seed set;
- the planned DPO run will be used as a project milestone rather than a smoke test.

For the first DPO smoke test, 50 high-quality pairs are enough to validate the pipeline. For a more meaningful DPO run, expand to 100-200 reviewed pairs after inspecting SFT-only outputs and collecting failure modes.

Do not expand by duplicating templates mechanically. New pairs should come from observed SFT errors, realistic student scenarios, and clear preference criteria.

## Evaluation Plan

When DPO training is implemented later, compare SFT-only and SFT+DPO on:

- preference win rate;
- rule-eval pass rate;
- prompt leakage count;
- truncation count;
- safe escalation pass rate;
- official-office mention pass rate;
- email format validity;
- response helpfulness and concision.

## Explicit Non-Goals

This plan does not implement:

- DPO training;
- RAG;
- vLLM;
- FastAPI;
- deployment.
