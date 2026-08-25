"""
Unit tests for Particle_Dataset in particle_loader.py, focused on JEOL SEM
metadata parsing and banner cropping - the machinery that turns a raw JEOL
.tif (with an information banner baked into the bottom of the image) plus
its metadata into a clean, correctly-cropped image for downstream analysis.

Uses the real fixture pair data/train data/A2-25.tif / .txt, copied into a
tmp_path per test. That .tif has no embedded $CM_FORMAT metadata block, so
these tests exercise the .txt-sidecar fallback branch of
_parse_jeol_metadata.
"""
import os
import shutil

import pytest

from particle_loader import Particle_Dataset

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train data")
FIXTURE_TIF = os.path.join(FIXTURE_DIR, "A2-25.tif")
FIXTURE_TXT = os.path.join(FIXTURE_DIR, "A2-25.txt")


@pytest.fixture
def jeol_dataset_dir(tmp_path):
    shutil.copy(FIXTURE_TIF, tmp_path / "A2-25.tif")
    shutil.copy(FIXTURE_TXT, tmp_path / "A2-25.txt")
    return tmp_path


def test_make_dataset_lists_only_tif_files(jeol_dataset_dir):
    dataset = Particle_Dataset(root=jeol_dataset_dir, device="jeol")
    assert len(dataset) == 1
    assert dataset.files == ["A2-25.tif"]


def test_parse_jeol_metadata_falls_back_to_txt_sidecar(jeol_dataset_dir):
    dataset = Particle_Dataset(root=jeol_dataset_dir, device="jeol")
    metadata = dataset._parse_jeol_metadata(str(jeol_dataset_dir / "A2-25.tif"))

    assert metadata["CM_IMAGE_SIZE"] == "1280 960"
    assert metadata["CM_PIXEL_SIZE"] == "1.25nm/pixel"
    # The sidecar's first line ($CM_FORMAT ...) is skipped by the parser, so
    # that key never makes it into the metadata dict - only later lines do.
    assert "CM_FORMAT" not in metadata
    assert metadata["CM_INSTRUMENT"] == "JSM-IT800"


def test_getitem_crops_out_the_jeol_info_banner(jeol_dataset_dir):
    dataset = Particle_Dataset(root=jeol_dataset_dir, device="jeol")
    image, name, metadata = dataset[0]

    assert name == "A2-25.tif"
    # Raw JEOL tif is (1026, 1280, 3); CM_IMAGE_SIZE says the real image
    # (banner excluded) is 1280x960, so crop_banner should cut it down to that.
    assert image.shape == (960, 1280, 3)
    assert metadata["CM_IMAGE_SIZE"] == "1280 960"


def test_crop_banner_is_enabled_automatically_for_jeol_device(jeol_dataset_dir):
    dataset = Particle_Dataset(root=jeol_dataset_dir, device="jeol")
    assert dataset.crop_banner is True
