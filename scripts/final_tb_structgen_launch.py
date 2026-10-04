#!/usr/bin/env python
"""Thin CLI for CP-DISR-TB-STRUCT-GEN-V1.

Subcommands: register | release | train | supervise | guard | status.  There is no default 'all'.
Importing this file never imports torch or builds an environment.
"""
from __future__ import annotations

import sys

from cp_disr import final_tb_structgen

if __name__ == "__main__":
    sys.exit(final_tb_structgen.main())
