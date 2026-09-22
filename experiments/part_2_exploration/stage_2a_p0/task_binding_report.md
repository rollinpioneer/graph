# Task binding report

## T_A
Real multi-object shared-prerequisite task. Signed goals: Inside(target, container) AND Inside(second_object, container). Reward only when both first become true.

## T_C
Intermediate relocation. Signed goal: Inside(target, container) only. Interferer starts on the target-to-container segment at 0.038 m from target. Relocation uses PICK(interferer)+PLACE_BUFFER, not MOVE. Interferer-at-buffer reward is 0.

{
  "interference": {
    "target_xy": [
      -0.12,
      -0.1
    ],
    "interferer_xy": [
      -0.0893566101574832,
      -0.07752818078215434
    ],
    "container_xy": [
      0.18,
      0.12
    ],
    "target_interferer_distance": 0.03799999999999999,
    "interferer_fraction_along_target_to_container": 0.10214463280838935,
    "geometrically_between": true,
    "rule": "second_xy = target + 0.038 * unit(container-target)",
    "pick_target_without_clearing": {
      "exit": "NORMAL_TERMINATION",
      "held_target": "UNKNOWN",
      "duration": 3.8999999999996815
    }
  },
  "prior_mechanism_evaluable": {
    "T_A": false,
    "T_C": true
  }
}
