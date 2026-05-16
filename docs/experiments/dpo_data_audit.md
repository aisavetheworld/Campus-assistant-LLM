# DPO Data Audit

## Summary

- DPO seed count: 61
- Train split: 49
- Eval split: 12
- Eval ratio: 0.2
- Seed: 42

## Category Distribution

| Category | Count |
|---|---:|
| `course_enrollment` | 17 |
| `email_drafting` | 8 |
| `health_insurance` | 20 |
| `housing` | 16 |

## Risk Distribution

| Risk Level | Count |
|---|---:|
| `high` | 25 |
| `low` | 10 |
| `medium` | 26 |

## Prefix Distribution

| Prefix | Count |
|---|---:|
| `dpo_email_quality` | 12 |
| `dpo_housing_safe` | 10 |
| `dpo_medical_safe` | 13 |
| `dpo_steps_email` | 12 |
| `dpo_visa_safe` | 14 |

## Quality Statistics

- Average chosen word count: 62.44
- Average rejected word count: 22.46
- Average chosen/rejected length ratio: 3.85
- Chosen missing official office when required: 0
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
| `extra_notes` | 7 |
| `missing_steps` | 38 |
| `no_official_office` | 36 |
| `overconfident_claim` | 5 |
| `poor_email_format` | 14 |
| `too_short` | 43 |

## Known Limitations

- This is still seed preference data, not a production-scale preference dataset.
- Some rejected answers are intentionally obvious to make the first DPO pipeline easy to validate.
- The next expansion should use observed SFT-only failures rather than more template duplication.
- The audit does not replace human review of preference quality.
- DPO smoke-test training is implemented separately in `scripts/train_dpo.py`.
