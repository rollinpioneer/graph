# S1-REV1 code diff

Authorized continuation change:

- bound `run_provider_call` to production `DashScopeProvider`, frozen request validation, redacted attempt logging, parser/admission, exact cache handling, and the single retry budget;
- added exact public scene-ID binding checks so missing or mismatched discovery inputs stop as `STOPPED_DISCOVERY_INPUT_BINDING`;
- preserved the S1-REV1 budget and downstream stop/verify accounting.
