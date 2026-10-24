"""Tests for the generate-page output layout (image/ backup, out/<stem>/ JSONs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from generator_backend import (
    generated_jsons,
    generator_output_base,
    prepare_generation_copies,
    unique_copy_target,
)


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "image", tmp_path / "out"


def _write_image(path: Path, payload: bytes = b"png-bytes"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class TestPrepareGenerationCopies:
    def test_backup_and_working_copy_are_created(self, dirs, tmp_path):
        image_dir, output_dir = dirs
        source = _write_image(tmp_path / "sources" / "doro1.png")

        backup, working = prepare_generation_copies(source, image_dir, output_dir)

        assert backup == image_dir / "doro1.png"
        assert backup.read_bytes() == b"png-bytes"
        assert working == output_dir / "doro1" / "doro1.png"
        assert working.read_bytes() == b"png-bytes"

    def test_generator_output_lands_in_the_per_image_folder(self, dirs, tmp_path):
        image_dir, output_dir = dirs
        source = _write_image(tmp_path / "sources" / "doro1.png")
        _, working = prepare_generation_copies(source, image_dir, output_dir)

        # The generator derives its -output base from the input path; with the
        # working copy, checkpoints must land inside out/<stem>/.
        base = generator_output_base(working)
        assert base.parent == output_dir / "doro1"

        final_json = base.with_suffix(".json")
        final_json.write_text("{}", encoding="utf-8")
        checkpoint = base.with_name("doro1.500.json")
        checkpoint.write_text("{}", encoding="utf-8")

        found = {path.name for path in generated_jsons(working)}
        assert {"doro1.json", "doro1.500.json"} <= found

    def test_regenerating_the_same_image_reuses_paths(self, dirs, tmp_path):
        image_dir, output_dir = dirs
        source = _write_image(tmp_path / "sources" / "doro1.png")

        first = prepare_generation_copies(source, image_dir, output_dir)
        second = prepare_generation_copies(source, image_dir, output_dir)

        assert first == second
        assert len(list((image_dir).iterdir())) == 1
        assert len(list((output_dir / "doro1").iterdir())) == 1

    def test_same_name_different_image_gets_a_suffix(self, dirs, tmp_path):
        image_dir, output_dir = dirs
        first = _write_image(tmp_path / "a" / "doro1.png", b"first-image")
        second = _write_image(tmp_path / "b" / "doro1.png", b"second-image")

        backup_one, working_one = prepare_generation_copies(first, image_dir, output_dir)
        backup_two, working_two = prepare_generation_copies(second, image_dir, output_dir)

        assert backup_one != backup_two
        assert working_one != working_two
        assert backup_two.parent == image_dir
        assert working_two.parent == output_dir / "doro1"
        assert backup_one.read_bytes() == b"first-image"
        assert backup_two.read_bytes() == b"second-image"

    def test_source_already_in_target_folder_is_not_copied_onto_itself(self, dirs):
        image_dir, output_dir = dirs
        source = _write_image(image_dir / "doro1.png")

        backup, working = prepare_generation_copies(source, image_dir, output_dir)

        assert backup == source
        assert working == output_dir / "doro1" / "doro1.png"


class TestUniqueCopyTarget:
    def test_missing_target_returned_as_is(self, tmp_path):
        source = _write_image(tmp_path / "src.png")
        assert unique_copy_target(tmp_path / "out.png", source) == tmp_path / "out.png"

    def test_identical_existing_file_is_reused(self, tmp_path):
        source = _write_image(tmp_path / "src.png", b"same")
        target = _write_image(tmp_path / "out.png", b"same")
        assert unique_copy_target(target, source) == target

    def test_different_content_returns_free_sibling(self, tmp_path):
        source = _write_image(tmp_path / "src.png", b"new")
        target = _write_image(tmp_path / "out.png", b"old")
        assert unique_copy_target(target, source) == tmp_path / "out-2.png"
