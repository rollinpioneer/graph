# Annotation manual v1

## Node intervals
A node is a stable semantic state. Start/end require five stable frames; transitions remain edge intervals.

## Edge intervals
Forward/alternative/failure/recovery/stagnation are mutually explicit. Alternative means a legal order only. Recovery must follow a state-supported failure.

## Failure and recovery
Failure onset is the earliest frame supported by state/intervention evidence; episode midpoint is forbidden. Recovery complete is the first restored node stable for five frames.

## Attempt/revisit
Attempt index increments only after a failed attempt group restarts; adjacent jitter is de-bounced.

## Within-node progress
Use anchors 0/.25/.5/.75/1.0 and interpolate between anchors.
