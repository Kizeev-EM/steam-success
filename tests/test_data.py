"""Отрицательные случаи проверки качества порции (ПР 2, требования К1–К10)."""
import pandas as pd
import pytest

from steam_success.config import INCOMING, load_params
from steam_success.data.io import read_games
from steam_success.data.validate import BatchRejected, check_cards

PARAMS = load_params()


@pytest.fixture
def batch(tmp_path):
    """Первые 20 карточек первой порции, прочитанные через read_games."""
    df = pd.read_csv(INCOMING / "batch_01.csv", dtype=str, keep_default_na=False).head(20)
    path = tmp_path / "batch.csv"
    df.to_csv(path, index=False)
    return path, df


def _reload(tmp_path, df):
    path = tmp_path / "broken.csv"
    df.to_csv(path, index=False)
    return read_games(path)


def test_clean_batch_passes(batch):
    path, _ = batch
    out, rep = check_cards(read_games(path), PARAMS)
    assert rep["rows_out"] == 20


@pytest.mark.parametrize("field,value,code", [
    ("platforms", "windows;playstation", "К8"),
    ("genres", "", "К5"),
    ("release_date", "05/03/2017", "К3"),
])
def test_bad_row_is_excluded(batch, tmp_path, field, value, code):
    _, df = batch
    df = df.copy()
    df.loc[0, field] = value
    out, rep = check_cards(_reload(tmp_path, df), PARAMS)
    assert rep[code] == 1
    assert rep["rows_out"] == 19


def test_post_release_field_rejects_batch(batch, tmp_path):
    _, df = batch
    df = df.assign(positive_ratings="10")
    with pytest.raises(BatchRejected, match="К1"):
        check_cards(_reload(tmp_path, df), PARAMS)


def test_duplicate_appid_rejects_batch(batch, tmp_path):
    _, df = batch
    df = pd.concat([df, df.head(1)], ignore_index=True)
    with pytest.raises(BatchRejected, match="К2"):
        check_cards(_reload(tmp_path, df), PARAMS)
