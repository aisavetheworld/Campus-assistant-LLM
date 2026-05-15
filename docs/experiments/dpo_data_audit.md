# DPO Data Audit

## Summary

- DPO seed count: 50
- Train split: 40
- Eval split: 10
- Eval ratio: 0.2
- Seed: 42

## Category Distribution

| Category | Count |
|---|---:|
| `course_enrollment` | 13 |
| `email_drafting` | 6 |
| `health_insurance` | 15 |
| `housing` | 16 |

## Risk Distribution

| Risk Level | Count |
|---|---:|
| `high` | 21 |
| `low` | 8 |
| `medium` | 21 |

## Prefix Distribution

| Prefix | Count |
|---|---:|
| `dpo_email_quality` | 10 |
| `dpo_housing_safe` | 10 |
| `dpo_medical_safe` | 10 |
| `dpo_steps_email` | 10 |
| `dpo_visa_safe` | 10 |

## Quality Statistics

- Average chosen word count: 60.88
- Average rejected word count: 38.18
- Average chosen/rejected length ratio: 1.67
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
| `extra_notes` | 6 |
| `missing_steps` | 7 |
| `no_official_office` | 35 |
| `overconfident_claim` | 4 |
| `poor_email_format` | 9 |

## Known Limitations

- This is still seed preference data, not a production-scale preference dataset.
- Some rejected answers are intentionally obvious to make the first DPO pipeline easy to validate.
- The next expansion should use observed SFT-only failures rather than more template duplication.
- The audit does not replace human review of preference quality.
- DPO training is intentionally not implemented yet.
