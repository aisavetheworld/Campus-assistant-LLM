# Experiment Log

This log records Project 1 SFT experiments for the International Student Campus Assistant. It focuses on reproducible observations from the minimal LoRA SFT loop.

## SFT v1

- Data: initial 32 SFT seed samples.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Method: LoRA SFT.
- Epochs: 1.
- Eval samples: 12.
- Total checks: 52.
- Passed checks: 37.
- Eval pass rate: 71.15%.

Findings:

- Email outputs often missed `Subject:`.
- Some email outputs missed greeting or closing fields.
- Some high-risk answers did not mention the relevant official office.
- Some evaluation checks were mismatched with `output_format`, especially ordinary guidance being checked for email closing.
- Base/SFT comparison showed that the first training loop was runnable but behavior was not stable enough.

## SFT v2

- Data: 38 SFT seed samples with targeted email and safe-escalation fixes.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Method: LoRA SFT.
- Epochs: 1.
- Eval samples: 12.
- Total checks: 53.
- Passed checks: 41.
- Eval pass rate: 77.36%.

Improvements:

- Pass rate improved from 71.15% to 77.36%.
- General housing guidance check mismatch was fixed.
- Health insurance waiver guidance passed all checks.
- Email samples now more consistently include formal structure in the training data.
- Evaluation now checks `no_extra_notes` for email drafts to catch postscript-style commentary.

Remaining failures:

- Email outputs still sometimes include `Note:` or `Ensure to replace...` after the email.
- `Subject:` is still unstable in one email case.
- Visa/enrollment high-risk prompts still sometimes miss `international student office`.
- Medical safe-boundary output can be too short and may miss `healthcare provider` or `student health center`.
- Insurance claim output can miss numbered steps.

Conclusion:

SFT v2 improved the rule pass rate from 71.15% to 77.36%, mainly by fixing evaluation-schema mismatch and improving general guidance. Remaining failures are concentrated in email postscript suppression and high-risk official-office escalation.

Next:

- Build SFT v3 targeted data focused on email postscript suppression.
- Add more high-risk CPT/OPT/enrollment examples that explicitly mention `international student office`.
- Add more medical safe-boundary examples that explicitly mention `healthcare provider` and `student health center`.
- Add more insurance claim examples with numbered steps and explicit `insurance office` or `insurance provider` referral.

## SFT v3

- Data: 102 SFT seed samples with expanded targeted email and safe-escalation data.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Method: LoRA SFT.
- Epochs: 1.
- Eval samples: 38.
- Total checks: 198.
- Passed checks: 156.
- Eval pass rate: 78.79%.

Evaluation note:

- SFT v3 used a larger and stricter eval set than SFT v2, so the pass rate is not a perfectly same-distribution comparison.
- The v3 eval set added more email postscript checks, more high-risk course/enrollment prompts, more medical boundary prompts, and a new `mentions_academic_office` check.

Improvements:

- Email format became more stable overall.
- `Subject:` failure was reduced to one eval case.
- Health insurance and medical-safety cases performed better than course/enrollment high-risk cases.
- Several insurance claim and medical boundary prompts passed all checks under the stricter eval.

Remaining failures:

- Email outputs still often include postscript-style commentary, causing `no_extra_notes` failures.
- Course/enrollment high-risk prompts remain weak, especially for `international student office`, `mentions_academic_office`, `has_steps`, and `not_too_short`.
- Some housing safe-escalation outputs failed `mentions_official_office`; this may be partly a rule vocabulary issue if the model uses terms such as residence life, RA, or campus housing staff.
- Some medical high-risk outputs are still too short or miss explicit healthcare-provider wording.

Conclusion:

SFT v3 held a 78.79% pass rate on a substantially larger and stricter 38-sample eval set. This suggests the v3 data direction is useful, but remaining failures are concentrated in email postscript suppression and high-risk course/enrollment escalation.

Next:

- Inspect failed v3 responses before adding more data.
- For email failures, add constraints that the answer must end immediately after `Best regards, [Your Name]`.
- For course/enrollment failures, add short high-risk examples whose first steps explicitly mention `international student office` or `academic advisor / department / registrar` depending on the risk type.
- Review `mentions_official_office` vocabulary for housing-related synonyms before assuming every housing failure is a model failure.
