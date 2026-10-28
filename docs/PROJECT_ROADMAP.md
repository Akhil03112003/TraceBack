# TraceBack project roadmap

TraceBack is a developer-facing log anomaly investigation workflow.

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | Dataset validation and baseline evaluation | In progress |
| Phase 2 | Spring Boot foundation, database, and ingestion API | Not started |
| Phase 3 | Parsing, normalization, evidence IDs, grouping, and deterministic detection | Not started |
| Phase 4 | Evidence-grounded AI investigation with structured outputs and validation | Not started |
| Phase 5 | React frontend and end-to-end workflow | Not started |
| Phase 6 | Benchmarking, hardening, Docker, CI, and documentation | Not started |

Phase 1 Step 1, dataset inspection and validation, is complete according to
the existing validation report. Its validation artifacts were originally
created in the source dataset directory rather than this workspace.

Phase 1 Step 2, the majority-class baseline, is complete after its
implementation, tests, actual-dataset execution, and output verification.
Phase 1 Step 3, the deterministic rule-based baseline, is complete after its
implementation, tests, actual-dataset execution, and output verification.
Phase 1 Step 4, the Isolation Forest machine-learning baseline, is complete
after its implementation, tests, actual-dataset execution, and output
verification. Phase 1 Step 5, comparative evaluation and final reporting, is
complete after its implementation, tests, artifact generation, and
independent verification. Phase 1 is complete. Phase 2 remains not started.
