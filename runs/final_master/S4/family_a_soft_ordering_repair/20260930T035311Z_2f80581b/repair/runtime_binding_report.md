# Family A runtime binding repair

The generic RuntimeBundle.start_case() now has optional verifier_factory and evaluator_factory fields. Defaults remain FactVerifier and TaskEvaluator; D0/T_A/T_B/T_C do not opt into the fields.

T_P_SO_MVP binds TPSOMVPFactVerifier and TPSOMVPTaskEvaluator through those factories. The task verifier first calls production FactVerifier, leaves all non-Open records unchanged, and applies only the public Open invariant when the measured Open value is UNKNOWN. FALSE is fail-closed. It checks the public manifest, reset identity, lid_closed=false, and registered contract effects. It does not call hidden_truth().

Offline repair tests cover bindings, invariant and B_PLAN progression with the real template.
