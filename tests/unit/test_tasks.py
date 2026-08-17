from pathlib import Path

from traceval.tasks import build_fixture, compute_task_hash, load_task

EXAMPLE_TASK_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "tasks" / "example_search_task"
)
FEEDBACK_FORM_TASK_DIR = Path(__file__).resolve().parents[2] / "tasks" / "feedback_form"


def test_load_task_parses_fields() -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    assert task.id == "example_search_task"
    assert task.seed == 12345
    assert task.environment.kind == "browser"
    assert task.environment.config["fixture_file"] == "fixture.html"
    assert {s.kind for s in task.scorers} == {"exact_match", "rubric"}
    assert task.expected == "found: playwright"
    assert task.requires_live_judge is False


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
