#!/usr/bin/env python
"""Thin CLI for CP-DISR-C1-MECH-CONFIRM-V1.

Subcommands: register | release | train | guard | status.  There is no default 'all' and no supervisor.
Importing this file never imports torch or builds an environment.
"""
from __future__ import annotations

import sys

from cp_disr import final_tb_c1

if __name__ == "__main__":
    sys.exit(final_tb_c1.main())
