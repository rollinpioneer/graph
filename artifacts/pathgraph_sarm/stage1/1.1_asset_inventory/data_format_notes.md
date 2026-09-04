# Data format notes

The scan uses the existing CUPID rollout manifest CSVs as the episode index. Each row points to one complete pickle rollout and retains its first/last timestep, success label, video path (in the source manifest), observation shape, and action shape. No video frames were decoded. `episode_file` is the complete-history source; semantic subtask, SARM, timestamp, and explicit recovery fields are absent. The existing robomimic HDF5 loader is documented in `CUPID/repo` and can be scanned by the same CLI when added to `data_roots`.

Episode boundary: manifest `first_timestep..last_timestep`, requiring first timestep 0 and an existing episode file for `has_full_episode_history=true`.
