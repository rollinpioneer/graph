# Fixed-rule attack

Stability order = the base item with the larger top-over-base COM margin (centred drop, offset bound 0.028 m). Rules compared with it on all 6 pairs.

| rule | agreements / 6 |
|---|---|
| larger_footprint_first | 6 |
| smaller_footprint_first | 0 |
| heavier_first | 6 |
| lighter_first | 0 |
| taller_first | 6 |
| shorter_first | 0 |
| larger_top_plateau_first | 6 |
| smaller_top_plateau_first | 0 |
| higher_com_first | 6 |
| lower_com_first | 0 |

Pinchable pairs: bbq_sauce|chocolate_pudding, bbq_sauce|cream_cheese, chocolate_pudding|cream_cheese

Rules that match the stability order on every pinchable pair (or on none, i.e. reversed): larger_footprint_first, smaller_footprint_first, heavier_first, lighter_first, taller_first, shorter_first, larger_top_plateau_first, smaller_top_plateau_first, higher_com_first, lower_com_first

Pair with size/stability agreement: True; pair with a conflict: False

| pair | stability first | larger first | gap (m) | class |
|---|---|---|---|---|
| bbq_sauce|butter | bbq_sauce | bbq_sauce | 0.0102 | HELPFUL |
| bbq_sauce|chocolate_pudding | bbq_sauce | bbq_sauce | 0.0068 | HELPFUL |
| bbq_sauce|cream_cheese | bbq_sauce | bbq_sauce | 0.0087 | HELPFUL |
| butter|chocolate_pudding | chocolate_pudding | chocolate_pudding | 0.0034 | NEUTRAL |
| butter|cream_cheese | cream_cheese | cream_cheese | 0.0016 | NEUTRAL |
| chocolate_pudding|cream_cheese | chocolate_pudding | chocolate_pudding | 0.0018 | NEUTRAL |

Result: single_rule_sufficient = True; pass = False
