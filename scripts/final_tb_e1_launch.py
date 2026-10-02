#!/usr/bin/env python
"""Thin CLI for the R-TB-E-1 (ELASTIC-01) launch (CP-DISR-TB-E1-ELASTIC-01).

Subcommands: check-config | register | elastic-ledger | rebind | evaluator-check | release | train | guard | status.
There is no default 'all'.  Importing this file never imports torch or builds an environment.
"""
from __future__ import annotations

import sys

from cp_disr import final_tb_e1

if __name__ == "__main__":
    sys.exit(final_tb_e1.main())
