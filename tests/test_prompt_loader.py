from reindex import prompt_loader


def test_loads_packaged_prompt():
    text = prompt_loader.load_prompt("conversation-summary")
    assert len(text) > 100


def test_override_dir_wins_per_file(tmp_path):
    (tmp_path / "conversation-summary.md").write_text("OVERRIDE", encoding="utf-8")
    prompt_loader.set_prompts_dir(tmp_path)
    try:
        assert prompt_loader.load_prompt("conversation-summary") == "OVERRIDE"
        # not present in override dir -> falls back to packaged default
        assert prompt_loader.load_prompt("project-aggregate") != "OVERRIDE"
        assert len(prompt_loader.load_prompt("project-aggregate")) > 100
    finally:
        prompt_loader.set_prompts_dir(None)
