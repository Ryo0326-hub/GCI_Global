# Final sprint review and handoff

[[00 Dashboard]] · [[07 How the Submission Works]] · [[04 Submission Tracker]]

A fresh Astra reviewer examined the whole branch from `6b0f323` through `4ca67bc`. The reviewer independently verified the final run's artifact hashes, package contents, executed source/config/notebook correspondence, Drive readbacks, test ID order and all 61,500 valid probabilities. No Critical findings or behaviors declined for review were reported.

## Findings addressed in one fix pass

1. **Interrupted export:** direct overwrite could truncate the current CSV. Promotion now writes a temporary sibling, checks its SHA-256, then replaces the current file. JSON metadata is also written and verified before replacement. Three export-failure cases and two metadata-interruption cases failed before the fix and pass afterward; existing candidate bytes are preserved.
2. **Late GPU failure:** command-line XGBoost previously rejected CPU fallback only after fitting a fold. The shared pipeline now checks every requested accelerator with a bounded native probe before any full fold starts. The post-fit checks remain. Three simulated unavailable-backend cases failed before the fix and pass afterward. A real unavailable-CUDA check on the Mac rejected the request in 0.027 seconds, before any full fold.
3. **Stale audit status:** the reviewer initially graded the decision-log opening Minor. It was treated as Important because a future worker could interpret it as permission to tune on the already inspected audit. The current champion now says the audit is closed; earlier results are explicitly historical. The dashboard's stale validation sentence was also corrected. Human prose was inspected directly rather than given a text-matching test.

The complete suite passes **38 tests on Python 3.12 and 3.13**. The generated notebook passes schema/source checks and a fresh-process end-to-end fixture workflow on both versions, including tuning, three model families, exact saved inference and matching CSV/ZIP export. Earlier actual Colab Python 3.13.16 evidence remains in [[research/SPRINT_VERIFICATION_2026-11-17]]. These post-review checks are local; they are not a claim that competition models were trained again in Colab. Detailed hashes and checks are in `SPRINT_REVIEW_VERIFICATION_2026-11-17.json`.

## Decisions made during execution

1. Used a new `codex/honours-sprint` branch in the existing vault to preserve active Obsidian paths, environments and run artifacts. Cost: less filesystem isolation than a separate checkout; explicit staging protected unrelated user settings.
2. Added two small feature groups inspired by public solutions and one GOSS comparison in response to Ryo's research request. Cost: bounded extra experiment time, with no guaranteed improvement.
3. Added three deeper derived-feature groups and XGBoost in response to the request for new categories and model diversity. Cost: additional bounded training and one tested dependency; no promise of 0.80 AUC.
4. Preserved each ensemble member's validated feature groups and training limits rather than forcing a shared set. Cost: additional configuration logic, covered by tests; split and audit identities remain shared.
5. Preserved the delivered S17 v2 artifacts while applying export/GPU safeguards to the current pipeline and notebook. Cost: the current notebook and executed S17 v2 notebook are different versions; this distinction is explicit below and in the runbook.

**Deferred minors: none.** The one initially Minor documentation finding was regraded and addressed. There was no second review; regression tests and the green full suite verify the one fix pass.

## Immutable S17 v2 delivery and current notebook

The delivered [S17 v2 CSV](https://drive.google.com/file/d/1NNzbcs0UK3ZUbuHh6GaBXzwLUlYFHHMe/view) and [matching ZIP](https://drive.google.com/file/d/1ir2pw5r823k55DYfxxTIXifON-keQ_jv/view) are unchanged. Their recorded source, configuration and manifests still describe the code actually executed. No predictions were altered, no competition models were retrained, and the historical source hash was not rewritten.

The [executed S17 v2 notebook](https://colab.research.google.com/drive/1zRbaJKregEsrMBcxkpNW4ni8Oe24X6-J) is separately archived and also included in the matching ZIP. The [current notebook](https://colab.research.google.com/drive/1sAPtD9Kmf3DiXiyFaLinhfOHHTcuGYZR) includes the later safeguards for future runs. Both notebook uploads were downloaded again and matched their respective local bytes. The model backup was also downloaded and hash-verified.

No rerun is needed to submit S17 v2. Ryo still needs to upload its CSV and matching ZIP, wait for scoring, record the public result, and ensure the intended candidate is the last submission before the deadline. Reserved audit AUC **0.764923** is not a public score or evidence of achieving the **0.80** target.
