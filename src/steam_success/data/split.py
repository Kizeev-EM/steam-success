"""Стадия split: разбиение набора по дате выхода на исходную часть и порции.

Порция пишется в два файла: batch_k.csv — только поля карточки, известные в
день релиза; labels_k.csv — число владельцев и всё, что копится после выхода.
В эксплуатации второй файл приходит через 12 месяцев после релиза.
"""
import pandas as pd

from steam_success.config import INCOMING, INTERIM, ROOT, load_params


def split(raw: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict]:
    sp = params["split"]
    raw = raw.copy()
    raw["_date"] = pd.to_datetime(raw["release_date"], format="%Y-%m-%d")
    # Детерминированный порядок: дата выхода, затем appid.
    raw = raw.sort_values(["_date", "appid"], kind="mergesort")
    initial = raw[raw["_date"] <= sp["initial_end"]].drop(columns="_date")
    batches = {}
    for b in sp["batches"]:
        part = raw[(raw["_date"] >= b["start"]) & (raw["_date"] <= b["end"])]
        batches[b["name"]] = part.drop(columns="_date")
    return initial, batches


def main() -> None:
    params = load_params()
    sp = params["split"]
    # Читаем как строки, чтобы записать значения без изменений.
    raw = pd.read_csv(ROOT / params["data"]["raw"], dtype=str, keep_default_na=False)
    initial, batches = split(raw, params)
    INTERIM.mkdir(parents=True, exist_ok=True)
    INCOMING.mkdir(parents=True, exist_ok=True)
    initial.to_csv(INTERIM / "initial.csv", index=False, lineterminator="\n")
    for name, part in batches.items():
        k = name.split("_")[1]
        part[sp["card_fields"]].to_csv(INCOMING / f"{name}.csv", index=False,
                                       lineterminator="\n")
        part[sp["label_fields"]].to_csv(INCOMING / f"labels_{k}.csv", index=False,
                                        lineterminator="\n")
    n_late = (pd.to_datetime(raw["release_date"]) > sp["label_cutoff"]).sum()
    print(f"initial: {len(initial)}; "
          + "; ".join(f"{n}: {len(p)}" for n, p in batches.items())
          + f"; без созревшей метки: {n_late}")


if __name__ == "__main__":
    main()
