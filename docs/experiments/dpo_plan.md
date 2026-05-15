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

The initial DPO seed set should cover:

- visa / OPT / CPT safe escalation;
- course enrollment and academic advisor escalation;
- housing office / Student Mail escalation;
- medical / healthcare provider escalation;
- insurance office / insurance provider escalation;
- email draft quality.

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
