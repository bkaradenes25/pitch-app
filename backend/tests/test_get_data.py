from datetime import date

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

import get_data  # noqa: E402  (pybaseball is only imported inside main())

OLD_COLUMNS = [c for c in get_data.KEEP if c not in get_data.REQUIRED_NEW]


def write(path, columns):
    pd.DataFrame({c: [1.0] for c in columns}).to_parquet(path, index=False)


def test_missing_file_is_downloaded(tmp_path):
    assert get_data.needs_download(tmp_path / "a.parquet", pd.Timestamp("2026-04-30"), pd.Timestamp("2026-10-01"))


def test_current_month_is_always_refreshed(tmp_path):
    f = tmp_path / "a.parquet"
    write(f, get_data.KEEP)
    assert get_data.needs_download(f, pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-01"))


def test_finished_month_with_every_column_is_skipped(tmp_path):
    f = tmp_path / "a.parquet"
    write(f, get_data.KEEP)
    assert not get_data.needs_download(f, pd.Timestamp("2026-04-30"), pd.Timestamp("2026-10-01"))


def test_finished_month_saved_by_the_old_script_is_downloaded_again(tmp_path):
    f = tmp_path / "a.parquet"
    write(f, OLD_COLUMNS)                      # no velocity / spin / movement columns
    assert get_data.needs_download(f, pd.Timestamp("2026-04-30"), pd.Timestamp("2026-10-01"))


def test_pitch_quality_columns_are_kept():
    assert set(get_data.REQUIRED_NEW) <= set(get_data.KEEP)


def test_refresh_flag_forces_a_download_even_when_the_file_is_complete(tmp_path):
    f = tmp_path / "a.parquet"
    write(f, get_data.KEEP)
    assert get_data.needs_download(f, pd.Timestamp("2026-04-30"), pd.Timestamp("2026-10-01"), force=True)


def test_old_files_without_flight_columns_are_still_accepted_without_refresh(tmp_path):
    f = tmp_path / "a.parquet"
    write(f, [c for c in get_data.KEEP if c not in ("vx0", "vy0", "vz0", "ax", "ay", "az")])
    assert not get_data.needs_download(f, pd.Timestamp("2026-04-30"), pd.Timestamp("2026-10-01"))
