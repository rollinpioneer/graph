# Gradient clipping logging contract

The formal `src/cp_disr/torch_rl.py::PPO.update` path now records, per actual optimizer step:

- `pre_clip_global_grad_norm`: independently recomputed before clipping;
- `post_clip_global_grad_norm`: independently recomputed after `clip_grad_norm_`;
- `clip_threshold`: frozen at 0.5;
- `clip_triggered`: pre norm greater than threshold;
- `optimizer_step_id`: incremented only after the optimizer step;
- legacy `grad_norm` is retained as the pre-clip value for compatibility.

The measurement helper only reads detached gradients and does not change gradients, RNG or optimizer state. It includes every parameter with a gradient, including a legal tail batch. No no-clip training run was created. The production-path unit test verified distinct fields and the post <= pre invariant.
