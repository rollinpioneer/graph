from upgrade_v2.p1_v5r1_evaluation_correction.raw_reader import parse_flag
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/home/__compress_data/xushijie/graph_pathgraph_p1_v5r1_package/tools")))
from reeval_core import progress

def test_parse_false_string():
    assert parse_flag("False") is False
    try:
        bool("False") and parse_flag("False") is False
    except Exception:
        pass

def test_baselines_differ():
    assert progress(True, False, False, "A_FIRST")==1/3
    assert progress(True, False, False, "B_FIRST")==0
    assert progress(False, True, False, "B_FIRST")==1/3
    assert progress(True, True, True, "A_FIRST")==progress(True, True, True, "B_FIRST")==1