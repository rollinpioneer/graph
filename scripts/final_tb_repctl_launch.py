#!/usr/bin/env python
"""Thin CLI for CP-DISR-TB-REP-CONTROLS-01.

Subcommands: register | release | train | supervise | status | smoke | verify-prep.  There is no default 'all'.
Importing this file never imports torch or builds an environment.
"""
from __future__ import annotations

import sys

from cp_disr import final_tb_repctl

if __name__ == "__main__":
    sys.exit(final_tb_repctl.main())
