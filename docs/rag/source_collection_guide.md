# Source Collection Guide

## What Counts as an Acceptable Source

Use only official sources. An acceptable source is one that:

- Is published directly by UCSD, UC San Diego Health, or the UC Office of the President
- Has a `.ucsd.edu`, `.ucop.edu`, or equivalent official domain
- Covers administrative policies, procedures, deadlines, or office contacts
- Would be referenced by an academic advisor, housing coordinator, registrar, ISEO advisor, or SHS staff member as authoritative

Examples of acceptable source types:

- Official UCSD webpage (housing.ucsd.edu, registrar.ucsd.edu, blink.ucsd.edu)
- UC SHIP plan documents (PDF from ucship.com or insurance.ucop.edu)
- ISEO official advising pages (ispo.ucsd.edu or equivalent)
- Student Health Services official pages
- Registrar official calendar pages

## What Not to Include

Do not use any of the following as factual sources:

- Reddit posts (r/UCSD, r/college, r/f1visa, etc.)
- Student blog posts, Medium articles, or personal websites
- Facebook groups, Discord servers, or WeChat group summaries
- RateMyProfessors or similar peer review sites
- Older cached pages that may no longer reflect current policy
- News articles interpreting policy (use the primary source instead)
- Any page that does not clearly identify the publisher as a UCSD office

**Why Reddit and forums are not usable:** Forum posts reflect personal experiences, not official policy. They frequently contain outdated information, school-specific misinterpretations, or generalizations that do not apply to UCSD. Using them as sources could cause the model to give confidently wrong answers on visa-status or insurance-claim questions, which carry real legal and financial consequences.

## How to Store Source Metadata

Every source should have an entry in `data/rag/ucsd_sources.json` following the schema:

```json
{
  "source_id": "ucsd_<category>_<topic>_<seq>",
  "school": "UCSD",
  "category": "<one of: housing, course_enrollment, health_insurance, international_students>",
  "title": "<page title or document name>",
  "url": "<full URL, leave empty string if not yet collected>",
  "source_type": "<official_webpage | official_pdf | official_form>",
  "date_collected": "<YYYY-MM-DD, leave empty if not yet collected>",
  "usable_for": ["<keyword1>", "<keyword2>"],
  "notes": "<one to two sentences on what this source covers and any caveats>"
}
```

## How to Record Collection Date

When you fetch or manually copy a source page, record today's date in `date_collected` as `YYYY-MM-DD`. This matters because UCSD policy pages update every quarter or year. If a chunk is retrieved and the collection date is more than one year old, flag it as potentially stale in the answer.

## How to Store Source Text

1. Create a plain text file at `data/rag/raw_sources/<source_id>.txt`
2. Paste or save the main body text from the source page
3. Remove navigation bars, footer links, and repeated boilerplate
4. Keep headers, numbered lists, bullet points, and table content — these are structurally important
5. Do not rephrase or summarize the content at this stage; store the original text

## Handling PDFs

For PDF documents (e.g., UC SHIP plan documents):

1. Download the PDF to `data/rag/raw_sources/<source_id>.pdf`
2. Extract text with `pdfminer.six` or `pypdf`
3. Save the extracted text as `data/rag/raw_sources/<source_id>.txt`
4. Note in `ucsd_sources.json` whether the PDF is an annual document that needs re-downloading

## Avoiding Private or Sensitive Student Information

Do not copy or store:

- Student names, student IDs, or email addresses from any example
- Medical record content, even redacted examples
- Any content that was only accessible after logging into TritonLink or an official student portal
- Visa numbers, passport numbers, or SEVIS numbers (from any context)

If a source page contains example student data for illustration, paraphrase the structure and omit any identifying values when storing the text. The goal is to capture official policy text, not personal data.

## Minimum Quality Bar Before Adding a Source

Before saving a source:

- Confirm the URL shows a `.ucsd.edu` or equivalent official domain
- Confirm the page has a visible last-updated date or publication attribution
- Confirm the content is relevant to at least one `usable_for` category that appears in the eval set
- Check that the text is not purely navigation (menus, site maps without substantive content)
