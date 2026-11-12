# GOLD AI — Gold Market Prediction

An experimental project exploring machine-learning approaches to historical gold-market (XAUUSD) data. The repository currently contains a ZIP archive and this guide; the Python source and data dependencies have not been independently reviewed here.

## Repository contents

- `GOLD_AI_RESEARCH_CLEAN (1).zip` — project archive. Extract it locally to inspect its contents before running anything.
- `README.md` — project overview and onboarding notes.

## Getting started

1. Download or clone this repository.
2. Extract the ZIP archive into a new local folder.
3. Review the extracted files, especially its own README, dependency files, and data-source instructions.
4. Use a separate virtual environment and install only dependencies specified by the extracted project.
5. Do not run scripts until you understand their inputs and any network or trading actions they perform.

There is no verified command to train or launch the project from the files currently visible at the repository root. Add the source files, dependency manifest, data description, and reproducible commands to the repository to make it directly usable.

## Data, credentials, and safety

Market data providers and API credentials are not documented in the root files. Keep credentials in local environment variables or an untracked `.env` file; never commit them. Historical model predictions are not financial advice, do not guarantee future results, and should not be used to place trades without independent validation and risk controls.

## Verification

No test command can be confirmed from the root contents. Record the environment, data period, evaluation method, and actual test results before publishing performance claims.


## Inspect the bundled research archive

Run `python inspect_archive.py` to validate the ZIP and list its members without extracting or executing project files. Run `python extract_archive.py` to extract only allow-listed text/data files into the ignored `archive_extracted/` directory for manual review. The CI workflow publishes the extracted review files as a temporary artifact; it does not execute the archived application.


The review branch safely extracts allow-listed source files into `GOLD_AI_RESEARCH/` for inspection. The extraction workflow does not execute any archived code.


The extracted research package now includes a Yahoo-provenance guard and credential-redaction regression tests; see the GOLD AI test workflow for current validation.
