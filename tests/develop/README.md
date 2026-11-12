# Evaluator subprocess evidence

The generated-scorer cases use `tests._evaluator_process.run_evaluator`.
It preserves `capture_output=True`, text decoding, the three-second timeout
and `subprocess.TimeoutExpired`. It does not retry a failed child. Python's
existing `subprocess.run` still kills and reaps that child on timeout.

`SCITEX_DATASET_EVALUATOR_IMPORT_TRACE=1` enables Python's `-X importtime`
for these children. The ordinary CI caller enables this diagnostic; it does
not change workers, coverage or the deadline. A successful scorer keeps its
normal stdout. A timeout prints an `EVALUATOR_TIMEOUT` JSON record containing
elapsed time, the parent worker PID, sampled CPU/I/O/memory pressure and up
to twelve import-timing rows. Where accessible, it also records the process's
cgroup-v2 CPU/memory limits and throttling counters, without the cgroup path.
Only standard-library roots and `coverage`
are named; other module names become `external`. Arguments, paths, stdout
and arbitrary child stderr are not added to that record.

The pressure sample is taken after the timeout. It does not prove what caused
an earlier failure. The PID identifies the parent test worker, not the child.
Elapsed time includes process creation and cleanup around the unchanged
three-second communication budget. These diagnostics do not establish that
the historical native failures are fixed or qualify a release.

Controls in `test_evaluator_process.py` use the real generated scorer and
owned sleeping child processes to check scoring, opt-in tracing, the timeout
contract and redaction. Existing scorer assertions stay in
`tests/scitex_dataset/ai_for_science/test__standardize.py`.
