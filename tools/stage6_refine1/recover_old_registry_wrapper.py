#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from tools.stage6_refine1.recover_frozen_registries import main as recover_main
if __name__=='__main__':
    # Kept as a named R1 entry point for the round command log.
    recover_main()
