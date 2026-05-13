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
