# PathGraph P2B-D1 graph-increment diagnostics

Frozen, read-only diagnostics for the P2B result at commit
`0f26660cde901aa32b0cfc032fc75ec01b31687b`.

The package never trains, updates an optimizer, performs back-propagation, selects a
checkpoint, or modifies P2B inputs. Large transition tables remain under the external
diagnostic root; Git contains only code, summaries, indices, hashes, and the report.

Run with `python -B -m d1.cli run-all ...` using the frozen P2B and P2A packages on
`PYTHONPATH`.
