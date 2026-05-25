# DPO Data Audit

## Summary

- DPO seed count: 157
- Train split: 126
- Eval split: 31
- Eval ratio: 0.2
- Seed: 42

## Category Distribution

| Category | Count |
|---|---:|
| `academic` | 20 |
| `course_enrollment` | 48 |
| `email_drafting` | 8 |
| `health_insurance` | 42 |
| `housing` | 39 |

## Risk Distribution

| Risk Level | Count |
|---|---:|
| `high` | 65 |
| `low` | 32 |
| `medium` | 60 |

## Prefix Distribution

| Prefix | Count |
|---|---:|
| `dpo_email_quality` | 30 |
| `dpo_housing_safe` | 28 |
| `dpo_medical_safe` | 33 |
| `dpo_steps_email` | 28 |
| `dpo_visa_safe` | 38 |

## Quality Statistics

- Average chosen word count: 67.52
- Average rejected word count: 46.94
- Average chosen/rejected length ratio: 2.19
- Chosen missing official office when required: 1
- Chosen bad marker count: 0

## Common Chosen Patterns

- Numbered steps for process guidance and safe escalation.
- Official office or qualified professional referral for medium/high-risk issues.
- No absolute guarantees for visa, medical, insurance, housing-contract, or academic-risk scenarios.
- Email drafts include `Subject:`, `Dear`, and a polite closing.
- `steps_plus_email` pairs include both process steps and an email draft.

## Common Rejected Flaws

| Flaw | Count |
|---|---:|
| `extra_notes` | 11 |
| `missing_steps` | 44 |
| `no_official_office` | 64 |
| `overconfident_claim` | 10 |
| `poor_email_format` | 37 |
| `too_short` | 43 |

## Known Limitations

- This is still seed preference data, not a production-scale preference dataset.
- Some rejected answers are intentionally obvious to make the first DPO pipeline easy to validate.
- The next expansion should use observed SFT-only failures rather than more template duplication.
- The audit does not replace human review of preference quality.
- DPO smoke-test training is implemented separately in `scripts/train_dpo.py`.
