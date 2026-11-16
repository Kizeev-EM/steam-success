"""Схемы pandera для карточки игры и файла меток (требования Таблицы 1.5 ПР 2)."""
import pandera.pandas as pa
from pandera.typing import Series

from steam_success.config import load_params

_Q = load_params()["quality"]
OWNERS_RANGES = [
    "0-20000", "20000-50000", "50000-100000", "100000-200000", "200000-500000",
    "500000-1000000", "1000000-2000000", "2000000-5000000", "5000000-10000000",
    "10000000-20000000", "20000000-50000000", "50000000-100000000",
    "100000000-200000000",
]


def _platforms_ok(s: Series[str]) -> Series[bool]:
    allowed = set(_Q["allowed_platforms"])
    return s.fillna("").map(lambda v: bool(v) and set(v.split(";")) <= allowed)


# Остановочные требования: К2 (appid целый, уникальный, не пустой).
card_stop_schema = pa.DataFrameSchema(
    {"appid": pa.Column(int, unique=True, nullable=False)},
    strict=False, coerce=True,
)

# Построчные требования: нарушение исключает запись (К3, К4, К5, К8).
card_row_checks = {
    "К3": lambda df, dump: df["release_date"].notna() & (df["release_date"] <= dump),
    "К4": lambda df, dump: df["price"] >= 0,
    "К5": lambda df, dump: df["genres"].fillna("").str.len() > 0,
    "К8": lambda df, dump: _platforms_ok(df["platforms"]),
}

labels_schema = pa.DataFrameSchema(
    {
        "appid": pa.Column(int, unique=True, nullable=False),
        "owners": pa.Column(str, pa.Check.isin(OWNERS_RANGES), nullable=False),  # К11
    },
    strict=False, coerce=True,
)
