#!/usr/bin/env python
"""CLI for the offline S1-REV1 r3 evidence closeout (see cp_disr.analysis.s1_evidence_closeout).

Subcommands: inventory | budget | branches | contracts | relations | assemble | verify
Exit codes: 0 OK, 2 INPUT_MISSING, 3 ANALYSIS_INCOMPLETE, 4 VERIFY_FAILED, 5 ZERO_CALL_VIOLATION.
"""
import sys

from cp_disr.analysis.s1_evidence_closeout import main

if __name__ == "__main__":
    sys.exit(main())
