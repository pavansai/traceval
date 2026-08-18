from pathlib import Path

import pytest

from traceval.tasks import build_fixture, compute_task_hash, load_task
from traceval.tasks.schema import TaskValidationError

EXAMPLE_TASK_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "tasks" / "example_search_task"
)
REPO_TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"
FEEDBACK_FORM_TASK_DIR = REPO_TASKS_DIR / "feedback_form"


def test_load_task_parses_fields() -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    assert task.id == "example_search_task"
    assert task.seed == 12345
    assert task.goal == 'Search for "playwright" and view the result.'
    assert task.environment.kind == "browser"
    assert task.environment.config["fixture_file"] == "fixture.html"
    assert {s.kind for s in task.scorers} == {"exact_match", "rubric"}
    assert task.expected == "found: playwright"
    assert task.requires_live_judge is False


def test_load_task_rejects_missing_goal(tmp_path: Path) -> None:
    (tmp_path / "task.yaml").write_text(
        "id: goalless_task\nseed: 1\nenvironment:\n  kind: fake\n  config: {}\n"
    )
    with pytest.raises(TaskValidationError, match="goal must be a non-empty"):
        load_task(tmp_path)


def test_load_task_rejects_blank_goal(tmp_path: Path) -> None:
    (tmp_path / "task.yaml").write_text(
        "id: goalless_task\nseed: 1\ngoal: '   '\nenvironment:\n  kind: fake\n  config: {}\n"
    )
    with pytest.raises(TaskValidationError, match="goal must be a non-empty"):
        load_task(tmp_path)


def test_load_task_parses_requires_live_judge() -> None:
    task = load_task(FEEDBACK_FORM_TASK_DIR)
    assert task.requires_live_judge is True


def test_build_fixture_resolves_paths() -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    fixture = build_fixture(task)
    assert fixture.path("fixture.html") == EXAMPLE_TASK_DIR / "fixture.html"
    assert fixture.path("scripted_trajectory.jsonl").exists()


def test_task_hash_is_stable() -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    h1 = compute_task_hash(task)
    h2 = compute_task_hash(task)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_task_hash_changes_with_fixture_content(tmp_path: Path) -> None:
    import shutil

    task_dir = tmp_path / "example_search_task"
    shutil.copytree(EXAMPLE_TASK_DIR, task_dir)
    task = load_task(task_dir)
    original_hash = compute_task_hash(task)

    fixture_html = task_dir / "fixture.html"
    fixture_html.write_text(fixture_html.read_text() + "<!-- changed -->")

    changed_hash = compute_task_hash(task)
    assert changed_hash != original_hash


def _write_task_yaml(task_dir: Path, *, observe_selectors: list[str], target: str) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text(
        "id: broken_task\n"
        "seed: 1\n"
        "goal: irrelevant to this validation\n"
        "environment:\n"
        "  kind: browser\n"
        "  config:\n"
        f"    fixture_file: fixture.html\n"
        f"    observe_selectors: {observe_selectors!r}\n"
        "scorers:\n"
        "  - kind: exact_match\n"
        "    config:\n"
        f"      target: {target!r}\n"
        "      expected: whatever\n"
    )


def test_load_task_rejects_exact_match_target_not_observed(tmp_path: Path) -> None:
    _write_task_yaml(tmp_path, observe_selectors=["#username"], target="#status")
    with pytest.raises(TaskValidationError, match="#status.*not in observe_selectors"):
        load_task(tmp_path)


def test_load_task_accepts_exact_match_target_that_is_observed(tmp_path: Path) -> None:
    _write_task_yaml(tmp_path, observe_selectors=["#username", "#status"], target="#status")
    task = load_task(tmp_path)
    assert task.id == "broken_task"


def test_load_task_skips_selector_validation_for_non_browser_environment(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "task.yaml").write_text(
        "id: fake_env_task\n"
        "seed: 1\n"
        "goal: irrelevant to this validation\n"
        "environment:\n"
        "  kind: fake\n"
        "  config: {}\n"
        "scorers:\n"
        "  - kind: exact_match\n"
        "    config:\n"
        "      target: '#nonexistent'\n"
        "      expected: whatever\n"
    )
    task = load_task(tmp_path)
    assert task.id == "fake_env_task"


def test_all_real_tasks_pass_selector_validation() -> None:
    """Regression guard for the whole tasks/ set: every real task must
    remain load-able (not just the one, feedback_form, that surfaced the
    original bug).
    """
    task_dirs = sorted(p for p in REPO_TASKS_DIR.iterdir() if (p / "task.yaml").exists())
    assert len(task_dirs) >= 4
    for task_dir in task_dirs:
        task = load_task(task_dir)
        assert task.format_version == 3
        assert task.goal.strip()
