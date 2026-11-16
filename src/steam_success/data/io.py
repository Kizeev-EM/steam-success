"""Чтение файлов набора с едиными типами полей."""
from pathlib import Path

import pandas as pd

LIST_FIELDS = ["platforms", "categories", "genres", "steamspy_tags"]


def read_games(path: Path | str) -> pd.DataFrame:
    """Читает исходный файл, исходную часть или порцию.

    Даты разбираются строго по формату ГГГГ-ММ-ДД: строка в другом формате
    превращается в NaT и затем отсеивается проверкой К3.
    """
    df = pd.read_csv(path, dtype={"name": "string", "developer": "string",
                                  "publisher": "string"})
    df["release_date"] = pd.to_datetime(df["release_date"], format="%Y-%m-%d",
                                        errors="coerce")
    return df


def split_list(s: pd.Series) -> pd.Series:
    """Поле со списком через «;» -> список строк (пустой для пропуска)."""
    return s.fillna("").astype(str).map(lambda v: [x for x in v.split(";") if x])


def owners_lower(owners: pd.Series) -> pd.Series:
    """Нижняя граница диапазона SteamSpy: '20000-50000' -> 20000."""
    return owners.astype(str).str.split("-").str[0].astype(int)


def make_target(owners: pd.Series, threshold: int) -> pd.Series:
    return (owners_lower(owners) >= threshold).astype(int)
