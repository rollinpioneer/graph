"""Re-export one saved-data figure with short labels; no scientific recomputation.

The original signal_analysis.auc_figure function is copied at runtime and only
its four axis-label texts and footer are changed. The original PNG and CSVs
remain byte-identical. No bootstrap, checkpoint, model query, or simulation.
"""
from core import ROOT, guard, sha, write
import inspect
import time
import signal_analysis as source_module


def main():
    guard()
    started=time.time()
    csv_path=ROOT/'SIGNAL_BY_DISTANCE.csv'
    original=ROOT/'figures/SIGNAL_AUROC_BY_DISTANCE.png'
    destination=ROOT/'figures/SIGNAL_AUROC_BY_DISTANCE_v2.png'
    assert csv_path.exists() and original.exists()
    assert not destination.exists(), 'VISUAL_EXPORT_EXISTS'
    input_hash=sha(csv_path);original_hash=sha(original)
    source=inspect.getsource(source_module.auc_figure)
    replacements={
        "('WP_log_AUROC','Waypoint / raw log score')":"('WP_log_AUROC','WP log AUROC')",
        "('WP_clipped_AUROC','Waypoint / clipped weight')":"('WP_clipped_AUROC','WP clipped AUROC')",
        "('AWR_log_AUROC','AWR / raw log score')":"('AWR_log_AUROC','AWR log AUROC')",
        "('AWR_clipped_AUROC','AWR / clipped weight')":"('AWR_clipped_AUROC','AWR clipped AUROC')",
        "if seed==0:ax.set_ylabel(label+'\\nAUROC: positive vs negative graph progress')":"if seed==0:ax.set_ylabel(label)",
        "Bands: 95% original-trajectory bootstrap intervals (500 draws). Missing/insufficient classes stay NA; neutral progress is excluded from AUROC.\\nScores concern graph progress only, not ground-truth value or causal supervision quality.":
        "AUROC: positive versus negative graph-distance progress; neutral samples are excluded.\\nBands: 95% original-trajectory bootstrap CIs (500 draws). Missing or insufficient classes stay NA.\\nScores are graph-progress proxies, not ground-truth value or causal supervision quality.",
    }
    for before,after in replacements.items():
        assert source.count(before)==1, ('SOURCE_TEXT_NOT_FOUND_ONCE',before)
        source=source.replace(before,after)
    namespace=dict(vars(source_module))
    exec(compile(source,'saved_data_visual_export_v2','exec'),namespace)
    namespace['auc_figure'](source_module.read_csv(csv_path),destination)
    guard()
    assert sha(csv_path)==input_hash and sha(original)==original_hash, 'SCIENTIFIC_OR_ORIGINAL_VISUAL_CHANGED'
    write(ROOT/'receipts/SIGNAL_VISUAL_EXPORT_V2.json',dict(status='COMPLETED_PURE_VISUAL_REPAIR',
        source_function='signal_analysis.auc_figure',source_file_sha256=sha(source_module.__file__),
        input=dict(path=str(csv_path),sha256=input_hash),
        original_figure=dict(path=str(original),sha256=original_hash,preserved=True),
        output=dict(path=str(destination),sha256=sha(destination)),
        changes='Shorten four y-axis labels; move positive-versus-negative definition to footer.',
        new_model_queries=0,new_bootstrap_draws=0,new_physical_steps=0,new_parameter_updates=0,
        seconds=time.time()-started))
    print('SIGNAL_VISUAL_EXPORT_V2_COMPLETE',flush=True)


if __name__=='__main__':
    main()
