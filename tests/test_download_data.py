"""Tests for the data download script (fully mocked, no network/disk)."""

from pathlib import Path
from unittest import mock

import pytest

from src.data.download_data import (
    FILES,
    RESOLVE_URL,
    _download_file,
    download_and_extract_data,
)

_SIZES = {
    "train_transaction.csv": 683_351_067,
    "train_identity.csv": 26_529_680,
}


class _FakeResponse:
    """Minimal stand-in for urllib's HTTPResponse used by urlopen."""

    headers = {"Content-Length": "5"}

    def __init__(self, body: bytes = b"hello") -> None:
        self._chunks = [body] if body else []

    def read(self, _n: int) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None


@pytest.fixture
def mocked_fs(tmp_path: Path):
    """Point Path.exists/stat/mkdir at in-memory stubs so no files are written."""
    existing = set()

    def fake_exists(self: Path) -> bool:
        return str(self) in existing

    def fake_stat(self: Path):
        stat = mock.Mock()
        stat.st_size = _SIZES.get(self.name, 0)
        return stat

    with (
        mock.patch("src.data.download_data.Path.exists", fake_exists),
        mock.patch("src.data.download_data.Path.stat", fake_stat),
        mock.patch("src.data.download_data.Path.mkdir", return_value=None),
    ):
        yield existing


def test_download_and_extract_skips_when_data_present(tmp_path: Path, mocked_fs) -> None:
    """Idempotency: existing train_transaction.csv short-circuits download."""
    mocked_fs.add(str(tmp_path / "train_transaction.csv"))

    with mock.patch("src.data.download_data._download_file", return_value=None) as mock_dl:
        download_and_extract_data(tmp_path)

    mock_dl.assert_not_called()


def test_download_and_extract_downloads_and_validates(tmp_path: Path, mocked_fs) -> None:
    """Happy path downloads every file and passes the size check."""
    with mock.patch("src.data.download_data._download_file", return_value=None) as mock_dl:
        download_and_extract_data(tmp_path)

    assert mock_dl.call_count == len(FILES)
    for call in mock_dl.call_args_list:
        url, dest = call.args
        assert url.startswith(RESOLVE_URL)
        assert Path(dest).name in FILES


def test_download_and_extract_raises_on_size_mismatch(tmp_path: Path, mocked_fs) -> None:
    """A corrupted/short file raises OSError."""
    with mock.patch("src.data.download_data._download_file", return_value=None) as mock_dl:

        def bad_stat(self: Path):
            stat = mock.Mock()
            stat.st_size = 10
            return stat

        with mock.patch("src.data.download_data.Path.stat", bad_stat):
            with pytest.raises(OSError, match="size mismatch"):
                download_and_extract_data(tmp_path)

    assert mock_dl.call_count == len(FILES)


def test_download_file_streams_and_reports(tmp_path: Path) -> None:
    """_download_file streams the response body to the destination path."""
    dest = tmp_path / "out.bin"

    with (
        mock.patch("src.data.download_data.urlopen", return_value=_FakeResponse()) as mock_open,
        mock.patch.object(Path, "open", mock.mock_open()) as mock_fh,
    ):
        _download_file(f"{RESOLVE_URL}/out.bin", dest)

    mock_open.assert_called_once()
    handle = mock_fh.return_value.__enter__.return_value
    handle.write.assert_called()
