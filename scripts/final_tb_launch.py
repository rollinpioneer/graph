#!/usr/bin/env python
"""Thin CLI for the final T_B launch (CP-DISR-NEXT-ENGINEERING-FIRST-TB-1).

Subcommands: check | register | release | smoke | train | summarize | status.  There is no default 'all'.
Importing this file never imports torch or builds an environment.
"""
from __future__ import annotations

import sys

from cp_disr import final_tb

if __name__ == "__main__":
    sys.exit(final_tb.main())
