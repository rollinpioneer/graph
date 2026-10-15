# Computational evaluation complete; visual review and final report pending

All4608model-frame predictions and18native scene AP evaluations passed. Development AP51.40379252%;fixed Seen subset54.70872391/official64.10353196%;fixed Novel subset40.12994793/official47.08921875%. The fixed two-scene means are not full Seen/Novel split results. All24selected-case replays matched original predictions. Twelve derived figures and all quantitative case observations are now saved.

Important diagnostic limitation:scene188has110unique RGB/depth/seg/meta observations and188frames109..255repeat one observation;scene189has171unique observations with170..255repeated. Fresh1285file per-scene CRC lists match independent official ZIP indexes. Native256frame AP and frozen selection remain unchanged. The three selected188frames represent one view with different sampled point clouds. See DIAGNOSTIC_OBSERVATION_AUDIT.json. Final interpretation and visual QA follow.
