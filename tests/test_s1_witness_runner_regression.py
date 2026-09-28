from pathlib import Path

def test_runner_separates_controller_and_task_success():
    text=Path("src/cp_disr/analysis/s1_revision_resume.py").read_text()
    assert '"controller_exit": execution.controller_exit' in text
    assert '"task_success": bool(task.success)' in text
    assert 'candidate = plan.plan[0]' in text

def test_runner_consumes_plan_and_checks_mask():
    text=Path("src/cp_disr/analysis/s1_revision_resume.py").read_text()
    assert 'CURRENT_MASK_REJECTED' in text
    assert 'plan.status in ("NO_PLAN", "SEARCH_TIMEOUT")' in text
    assert 'bundle.executor.execute(candidate' in text
