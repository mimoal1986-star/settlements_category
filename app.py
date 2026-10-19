import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import os
import json
import base64
import requests
from datetime import datetime, date

R_EARTH = 6371.0

SPRAVOCHNIK_FILE = "spravochnik.csv"
KF_FILE = "kf.csv"
COEFFICIENTS_FILE = "coefficients.json"

DEFAULT_COEFFS = {
    "kf_category": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0},
}

NEW_COLS = [
    "Для НП 3-6 → НП 1, 2 Имя НП",
    "Для НП 3-6 → НП 1, 2 Область",
    "Для НП 3-6 → НП 1, 2 расстояние",
    "Для НП 5-6 → НП 3, 4 Имя НП",
    "Для НП 5-6 → НП 3, 4 Область",
    "Для НП 5-6 → НП 3, 4 расстояние",
    "Для НП 6 → НП 5 Имя НП",
    "Для НП 6 → НП 5 Область",
    "Для НП 6 → НП 5 расстояние",
]

FINAL_COLS = ["Город new", "Область new", "Категория new", "Категория new с КФ доезда"]

GRAD_TO_CAT = {
    "миллионник": 1,
    "от 500 тыс. до 1 млн. человек": 1,
    "от 250 тыс. до 500 тыс. человек": 2,
    "от 100 тыс. до 250 тыс. человек": 3,
    "от 50 тыс. до 100 тыс": 4,
    "от 10 тыс. до 50 тыс": 5,
    "менее 10000": 6,
}

# ==================== СЛОВАРЬ КРУПНЫХ ГОРОДОВ ====================
CITY_COORDS = {
    "москва": (55.7558, 37.6173),
    "санкт-петербург": (59.9343, 30.3351),
    "уфа": (54.7388, 55.9721),
    "тюмень": (57.1522, 65.5272),
    "севастополь": (44.6166, 33.5254),
    "астрахань": (46.3497, 48.0408),
    "сургут": (61.2540, 73.3962),
    "луганск": (48.5670, 39.3170),
    "архангельск": (64.5393, 40.5182),
    "владикавказ": (43.0367, 44.6678),
    "орел": (52.9651, 36.0785),
    "орёл": (52.9651, 36.0785),
    "нижневартовск": (60.9344, 76.5531),
    "зеленоград": (55.9960, 37.2090),
    "королёв": (55.9142, 37.8256),
    "королев": (55.9142, 37.8256),
    "северодвинск": (64.5635, 39.8302),
    "колпино": (59.7500, 30.6000),
    "щёлково": (55.9209, 37.9997),
    "щелково": (55.9209, 37.9997),
    "нефтеюганск": (61.0883, 72.6103),
    "ханты-мансийск": (61.0042, 69.0019),
    "тобольск": (58.2000, 68.2667),
    "петергоф": (59.8833, 29.9000),
    "ишим": (56.1114, 69.4900),
    "сунжа": (43.3167, 45.6833),
    "будённовск": (44.7833, 44.1667),
    "буденновск": (44.7833, 44.1667),
    "ялуторовск": (56.6500, 66.3000),
    "беслан": (43.1833, 44.5333),
    "моздок": (43.7500, 44.6500),
    "ликино-дулёво": (55.7167, 38.9500),
    "ликино-дулево": (55.7167, 38.9500),
    "заводоуковск": (56.5000, 66.5500),
    "озёры": (54.8500, 38.5500),
    "озеры": (54.8500, 38.5500),
    "алагир": (43.0333, 44.2167),
    "пикалёво": (59.5167, 34.1667),
    "пикалево": (59.5167, 34.1667),
    "ардон": (43.1667, 44.3000),
    "дигора": (43.1500, 44.1500),
    "нерчинск": (51.9667, 116.5833),
    "полесск": (54.8667, 21.1000),
    "дмитриев-льговский": (52.1167, 35.0833),
    "новохопёрск": (51.1000, 41.6167),
    "новохоперск": (51.1000, 41.6167),
    "белоозёрский": (55.4667, 38.5667),
    "белоозерский": (55.4667, 38.5667),
    "сириус": (43.4000, 39.9667),
}


# ==================== GITHUB API ====================
def github_get_file(path):
    try:
        cfg = st.secrets["github"]
        token = cfg["token"]
        owner = cfg["owner"]
        repo = cfg["repo"]
        branch = cfg.get("branch", "main")
    except Exception:
        return None, None

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": branch}

    headers_raw = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.raw",
    }
    content = None
    try:
        r = requests.get(url, headers=headers_raw, params=params, timeout=60)
        if r.status_code == 200:
            content = r.content
    except Exception:
        pass

    if content is None:
        return None, None

    sha = None
    try:
        headers_json = {"Authorization": f"token {token}"}
        r2 = requests.get(url, headers=headers_json, params=params, timeout=15)
        if r2.status_code == 200:
            data = r2.json()
            sha = data.get("sha")
    except Exception:
        pass

    return content, sha


def github_put_file(path, content_bytes, commit_message):
    try:
        cfg = st.secrets["github"]
        token = cfg["token"]
        owner = cfg["owner"]
        repo = cfg["repo"]
        branch = cfg.get("branch", "main")
    except Exception as e:
        return False, f"Нет настроек github in secrets: {e}"

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}"}
    params = {"ref": branch}

    sha = None
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if r.status_code == 200:
            sha = r.json().get("sha")
    except Exception:
        pass

    content_b64 = base64.b64encode(content_bytes).decode("utf-8")
    payload = {
        "message": commit_message,
        "content": content_b64,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    try:
        r = requests.put(url, headers=headers, json=payload, timeout=30)
        if r.status_code in (200, 201):
            return True, "OK"
        else:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)


def github_read_csv(path):
    content, _ = github_get_file(path)
    if content is None:
        return None
    try:
        return pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(io.BytesIO(content), encoding="cp1251")
        except Exception:
            return None


# ==================== ГУК ====================
def load_coefficients():
    content, _ = github_get_file(COEFFICIENTS_FILE)
    if content is not None:
        try:
            data = json.loads(content.decode("utf-8"))
            cat = data.get("kf_category", {})
            cat = {int(k): float(v) for k, v in cat.items()}
            return {"kf_category": cat}
        except Exception:
            pass
    return dict(DEFAULT_COEFFS)


# ==================== УТИЛИТЫ ====================
def squeeze(s):
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).strip())


def find_col(df, target):
    if target is None:
        return None
    target_norm = squeeze(target).lower()
    for c in df.columns:
        if squeeze(c).lower() == target_norm:
            return c
    return None


def is_empty(v):
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(v, (datetime, date)):
        return False
    s = str(v).strip()
    return s == "" or s == "-" or s == "—"


def to_float(v):
    if v is None:
        return np.nan
    if isinstance(v, (datetime, date)):
        return np.nan
    try:
        if pd.isna(v):
            return np.nan
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip()
    if s == "" or s == "-" or s == "—":
        return np.nan
    if s.endswith("%"):
        s = s[:-1].strip()
        try:
            return float(s.replace(",", ".")) / 100.0
        except (ValueError, TypeError):
            return np.nan
    s2 = s.replace(" ", "").replace(",", ".")
    try:
        return float(s2)
    except (ValueError, TypeError):
        return np.nan


def population_to_gradation(pop):
    if pop is None or pd.isna(pop):
        return None
    try:
        pop = float(pop)
    except (ValueError, TypeError):
        return None
    if pop >= 1_000_000:
        return "миллионник"
    if pop >= 500_000:
        return "от 500 тыс. до 1 млн. человек"
    if pop >= 250_000:
        return "от 250 тыс. до 500 тыс. человек"
    if pop >= 100_000:
        return "от 100 тыс. до 250 тыс. человек"
    if pop >= 50_000:
        return "от 50 тыс. до 100 тыс"
    if pop >= 10_000:
        return "от 10 тыс. до 50 тыс"
    return "менее 10000"


def gradation_to_category(grad):
    if grad is None or pd.isna(grad):
        return None
    return GRAD_TO_CAT.get(str(grad).strip(), None)


def clean_dataframe(df):
    df = df.copy()
    if "Широта" in df.columns:
        df["Широта"] = df["Широта"].apply(to_float).astype(float)
    if "Долгота" in df.columns:
        df["Долгота"] = df["Долгота"].apply(to_float).astype(float)
    if "Население" in df.columns:
        df["Население"] = df["Население"].apply(to_float)
    if "match_status" in df.columns:
        df = df.drop(columns=["match_status"])
    return df


def remove_duplicates(user_df):
    df = user_df.copy()
    df["_cat_filled"] = df["Категория"].apply(lambda v: not is_empty(v))

    groups_seen = {}
    for idx, row in df.iterrows():
        key = (squeeze(row["Город"]).lower(), squeeze(row["Область"]).lower())
        groups_seen.setdefault(key, []).append(idx)

    keep_set = set()
    for key, indices in groups_seen.items():
        filled = [i for i in indices if df.at[i, "_cat_filled"]]
        if filled:
            keep_set.update(filled)
        else:
            keep_set.add(indices[0])

    result = df[df.index.isin(keep_set)].drop(columns=["_cat_filled"])
    return result.reset_index(drop=True)


def count_filled(row):
    cols = ["Город", "Широта", "Долгота", "Население",
            "Градация", "Категория", "Область"]
    return sum(1 for c in cols if c in row.index and not is_empty(row[c]))


def final_dedup(df):
    if len(df) == 0:
        return df

    df = df.copy()
    df["_filled"] = df.apply(count_filled, axis=1)
    df["_key"] = df.apply(
        lambda r: (squeeze(r["Город"]).lower(), squeeze(r["Область"]).lower()),
        axis=1
    )
    df["_orig_order"] = range(len(df))

    df_sorted = df.sort_values(
        ["_filled", "_orig_order"],
        ascending=[False, True],
        kind="stable"
    )

    idx_to_keep = df_sorted.groupby("_key", sort=False).head(1).index
    result = df.loc[sorted(idx_to_keep)].drop(
        columns=["_filled", "_key", "_orig_order"]
    )
    return result.reset_index(drop=True)


# ==================== ГОРОД NEW ====================
def compute_city_new(sheet_res):
    sheet_res = sheet_res.copy()
    sheet_res["Город new"] = None
    sheet_res["Область new"] = None
    sheet_res["Категория new"] = None
    sheet_res["Категория new с КФ доезда"] = None

    city_to_cat = {}
    for _, row in sheet_res.iterrows():
        city = squeeze(row["Город"])
        reg  = squeeze(row["Область"])
        cat  = row["Категория"]
        if city and reg and not is_empty(cat):
            key = (city.lower(), reg.lower())
            if key not in city_to_cat:
                city_to_cat[key] = cat

    for i, row in sheet_res.iterrows():
        K = row.get("Для НП 3-6 → НП 1, 2 Имя НП")
        L = row.get("Для НП 3-6 → НП 1, 2 расстояние")
        K_reg = row.get("Для НП 3-6 → НП 1, 2 Область")
        M = row.get("Для НП 5-6 → НП 3, 4 Имя НП")
        N = row.get("Для НП 5-6 → НП 3, 4 расстояние")
        M_reg = row.get("Для НП 5-6 → НП 3, 4 Область")
        O = row.get("Для НП 6 → НП 5 Имя НП")
        P = row.get("Для НП 6 → НП 5 расстояние")
        O_reg = row.get("Для НП 6 → НП 5 Область")

        B = squeeze(row["Город"])
        B_reg = squeeze(row["Область"])
        B_cat = row["Категория"]

        L_f = to_float(L); N_f = to_float(N); P_f = to_float(P)

        matched_name = None
        matched_reg = None
        if not pd.isna(L_f) and L_f < 31:
            matched_name = K; matched_reg = K_reg
        elif not pd.isna(N_f) and N_f < 21:
            matched_name = M; matched_reg = M_reg
        elif not pd.isna(P_f) and P_f < 11:
            matched_name = O; matched_reg = O_reg

        if matched_name is not None and not is_empty(matched_name):
            sheet_res.at[i, "Город new"] = matched_name
            sheet_res.at[i, "Область new"] = matched_reg
            if not is_empty(matched_reg):
                cat_new = city_to_cat.get(
                    (squeeze(matched_name).lower(), squeeze(matched_reg).lower())
                )
            else:
                cat_new = None
            cat_new = cat_new if not is_empty(cat_new) else B_cat
        else:
            sheet_res.at[i, "Город new"] = B
            sheet_res.at[i, "Область new"] = B_reg
            cat_new = B_cat

        sheet_res.at[i, "Категория new"] = cat_new

        cat_new_f = to_float(cat_new)
        cat_old_f = to_float(B_cat)
        if not pd.isna(cat_new_f) and not pd.isna(cat_old_f) and cat_new_f < cat_old_f:
            cat_with_kf = cat_new_f + 0.5
        else:
            cat_with_kf = cat_new_f
        sheet_res.at[i, "Категория new с КФ доезда"] = cat_with_kf

    return sheet_res


# ==================== AVP ====================
def build_avp(sx_df, result_df, coeffs, kf_df, avg_price_input):
    issues = {
        "no_region": pd.DataFrame(),
        "duplicates": pd.DataFrame(),
        "no_category": pd.DataFrame(),
    }

    # KF
    kf_slozhnosti = {}
    kf_lokal = {}
    if kf_df is not None and len(kf_df) > 0:
        col_reg = find_col(kf_df, "Сценарий (идентификатор)*") or find_col(kf_df, "Сценарий (идентификатор)") or kf_df.columns[0]
        col_sl = find_col(kf_df, "КФ сложности по типу регионов") or find_col(kf_df, "КФ сложности")
        col_lk = find_col(kf_df, "КФ локальной сложности") or find_col(kf_df, "КФ локальной")
        for _, r in kf_df.iterrows():
            reg = squeeze(r[col_reg]).upper()
            if not reg:
                continue
            if col_sl:
                v = to_float(r[col_sl])
                kf_slozhnosti[reg] = v if not pd.isna(v) else 0.0
            if col_lk:
                v = to_float(r[col_lk])
                kf_lokal[reg] = v if not pd.isna(v) else 0.0

    # Проверка «один проект»
    if "Project Code" in sx_df.columns:
        unique_projects = sx_df["Project Code"].dropna().astype(str).str.strip().unique()
        unique_projects = [p for p in unique_projects if p]
        if len(unique_projects) > 1:
            raise ValueError(
                f"В СХ найдено {len(unique_projects)} проектов. "
                f"Выбери 1 проект. Найдено: {unique_projects[:5]}"
            )

    sx_key_names = ["Client", "Project Name", "Wave Name", "Project Code", "Регион", "Город"]
    sx_cols = {}
    for name in sx_key_names + ["Region short"]:
        col = find_col(sx_df, name)
        if col is None:
            raise ValueError(f"В СХ нет колонки: {name}")
        sx_cols[name] = col

    sx_selected = sx_df[[sx_cols[k] for k in sx_key_names] + [sx_cols["Region short"]]].copy()
    sx_selected.columns = sx_key_names + ["Region short"]

    counts = sx_selected.groupby(sx_key_names + ["Region short"]).size().reset_index(name="Кол-во анкет")
    avp = counts.copy()

    # Мапа (Город, Region short) → (Категория new, Категория new с КФ доезда)
    res_city = find_col(result_df, "Город")
    res_region = find_col(result_df, "СВЗ регион")
    if res_city is None or res_region is None:
        raise ValueError("В result нет колонок 'Город' или 'СВЗ регион'")

    res_cat_new = find_col(result_df, "Категория new")
    res_cat_kf = find_col(result_df, "Категория new с КФ доезда")

    bad_region_mask = result_df[res_region].apply(is_empty)
    bad_region = result_df[bad_region_mask][[res_city, res_region]]
    issues["no_region"] = bad_region

    city_cat_map = {}
    dup_rows = []
    for _, r in result_df.iterrows():
        city = squeeze(r[res_city]).lower()
        reg = squeeze(r[res_region]).lower()
        if not city or not reg:
            continue
        cat_new = r.get(res_cat_new)
        cat_kf = r.get(res_cat_kf)
        key = (city, reg)
        if key in city_cat_map:
            dup_rows.append({
                "Город": r[res_city],
                "СВЗ регион": r[res_region],
                "Категория new": cat_new,
            })
            continue
        city_cat_map[key] = (cat_new, cat_kf)

    if dup_rows:
        issues["duplicates"] = pd.DataFrame(dup_rows)

    def find_cat(row):
        city = squeeze(row["Город"]).lower()
        reg = squeeze(row["Region short"]).lower()
        if not city or not reg:
            return None, None
        return city_cat_map.get((city, reg), (None, None))

    cats_and_kf = avp.apply(find_cat, axis=1)
    avp["Категория new"] = [t[0] for t in cats_and_kf]
    avp["_cat_with_kf"] = [t[1] for t in cats_and_kf]

    no_cat = avp[avp["Категория new"].apply(is_empty)]
    if len(no_cat):
        issues["no_category"] = no_cat[sx_key_names + ["Region short"]]

    kf_cat_settings = coeffs["kf_category"]

    avp["КФ категории"] = avp["Категория new"].apply(
        lambda c: kf_cat_settings.get(int(c), 0.0) if not is_empty(c) else 0.0
    )

    def calc_kf_doezd(row):
        c_new = to_float(row["Категория new"])
        c_kf = to_float(row["_cat_with_kf"])
        if pd.isna(c_new) or pd.isna(c_kf):
            return 0.0
        return round(c_kf - c_new, 2)

    avp["КФ доезда"] = avp.apply(calc_kf_doezd, axis=1)

    def find_kf_slozhnost(row):
        reg = squeeze(row["Region short"]).upper()
        return kf_slozhnosti.get(reg, 0.0)

    def find_kf_lokal(row):
        reg = squeeze(row["Region short"]).upper()
        return kf_lokal.get(reg, 0.0)

    avp["КФ сложности"] = avp.apply(find_kf_slozhnost, axis=1)
    avp["КФ локальной сложности"] = avp.apply(find_kf_lokal, axis=1)

    avp["КФ итог"] = (
        avp["КФ категории"] +
        avp["КФ доезда"] +
        avp["КФ сложности"] +
        avp["КФ локальной сложности"]
    ).round(4)

    # X
    total_count = avp["Кол-во анкет"].sum()
    weighted_sum = (avp["КФ итог"] * avp["Кол-во анкет"]).sum()

    if weighted_sum == 0 or total_count == 0:
        raise ValueError("Не удалось рассчитать: сумма КФ итог × кол-во = 0")

    X = avg_price_input * total_count / weighted_sum

    avp["Ср цена в поля"] = avg_price_input
    avp["Цена анкеты"] = (avp["КФ итог"] * X).round(2)

    avp["△ Цена анкеты - Ср цена в поля, руб"] = (
        avp["Цена анкеты"] - avp["Ср цена в поля"]
    ).round(2)

    avp["△ Цена анкеты - Ср цена в поля, %"] = (
        (avp["Цена анкеты"] - avp["Ср цена в поля"]) / avp["Ср цена в поля"] * 100
    ).round(2)

    rename_map = {
        "Client": "Клиент Имя",
        "Project Name": "Проект Имя",
        "Wave Name": "Волна Имя",
        "Project Code": "Проект Код",
    }
    avp = avp.rename(columns=rename_map)

    final_order = [
        "Клиент Имя", "Проект Имя", "Волна Имя", "Проект Код",
        "Регион", "Город", "Region short",
        "Категория new", "Ср цена в поля",
        "КФ категории", "КФ доезда", "КФ сложности", "КФ локальной сложности",
        "КФ итог", "Цена анкеты",
        "△ Цена анкеты - Ср цена в поля, руб",
        "△ Цена анкеты - Ср цена в поля, %",
    ]
    avp = avp[[c for c in final_order if c in avp.columns]]

    return avp, issues


# ==================== ГЕОМЕТРИЯ ====================
def haversine_matrix(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=float))[:, None]
    lon1 = np.radians(np.asarray(lon1, dtype=float))[:, None]
    lat2 = np.radians(np.asarray(lat2, dtype=float))[None, :]
    lon2 = np.radians(np.asarray(lon2, dtype=float))[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0, 1)
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def find_nearest(src_df, tgt_df):
    n = len(src_df)
    names = np.array([None] * n, dtype=object)
    regions = np.array([None] * n, dtype=object)
    dists = np.full(n, np.nan)
    if len(tgt_df) == 0 or n == 0:
        return names, regions, dists
    tgt = tgt_df[
        tgt_df["lat"].notna() & tgt_df["lon"].notna() &
        (tgt_df["_name_orig"].astype(str).str.strip() != "") &
        (tgt_df["_region_orig"].astype(str).str.strip() != "")
    ].copy()
    if len(tgt) == 0:
        return names, regions, dists
    D = haversine_matrix(src_df["lat"].values, src_df["lon"].values,
                         tgt["lat"].values, tgt["lon"].values)
    idx = D.argmin(axis=1)
    names = tgt["_name_orig"].values[idx]
    regions = tgt["_region_orig"].values[idx]
    dists = D[np.arange(n), idx]
    return names, regions, dists


# ==================== МАТЧИНГ ====================
def match_settlements(user_df, ds_df):
    ds = ds_df.copy()
    ds["_s"] = ds["settlement"].apply(squeeze).str.lower()
    ds["_r"] = ds["region"].apply(squeeze).str.lower()
    ds["_lat"] = [to_float(v) for v in ds["latitude_dd"].values]
    ds["_lon"] = [to_float(v) for v in ds["longitude_dd"].values]
    if "population" in ds.columns:
        ds["_pop"] = [to_float(v) for v in ds["population"].values]
    else:
        ds["_pop"] = [np.nan] * len(ds)

    sett_map = {}
    for s, r, lat, lon, pop in zip(
        ds["_s"].values, ds["_r"].values,
        ds["_lat"].values, ds["_lon"].values,
        ds["_pop"].values
    ):
        if not s or not r:
            continue
        if pd.isna(lat) or pd.isna(lon):
            continue
        key = (s, r)
        if key not in sett_map:
            sett_map[key] = (lat, lon, pop)

    out = user_df.copy()
    out["match_status"] = ""

    for i, row in out.iterrows():
        lat_in = to_float(row.get("Широта"))
        lon_in = to_float(row.get("Долгота"))
        if not pd.isna(lat_in) and not pd.isna(lon_in):
            out.at[i, "match_status"] = "из исходника"
            continue

        u_name = squeeze(row.get("Город", "")).lower()
        u_reg  = squeeze(row.get("Область", "")).lower()

        if not u_name or not u_reg:
            out.at[i, "match_status"] = "пусто в исходнике"
            continue

        if u_name in CITY_COORDS:
            lat, lon = CITY_COORDS[u_name]
            out.at[i, "Широта"] = float(lat)
            out.at[i, "Долгота"] = float(lon)
            out.at[i, "match_status"] = "из словаря"
            continue

        key = (u_name, u_reg)
        if key in sett_map:
            lat, lon, pop = sett_map[key]
            out.at[i, "Широта"] = float(lat)
            out.at[i, "Долгота"] = float(lon)
            out.at[i, "match_status"] = "из справочника"
            if is_empty(out.at[i, "Население"]) and not pd.isna(pop):
                out.at[i, "Население"] = float(pop)
            if is_empty(out.at[i, "Градация"]):
                grad = population_to_gradation(out.at[i, "Население"])
                if grad is not None:
                    out.at[i, "Градация"] = grad
            if is_empty(out.at[i, "Категория"]):
                cat = gradation_to_category(out.at[i, "Градация"])
                if cat is not None:
                    out.at[i, "Категория"] = cat
            continue

        out.at[i, "match_status"] = "не найдено"

    return out


def read_any(f):
    raw = f.read()
    name = f.name.lower()

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        except (UnicodeDecodeError, UnicodeError):
            df = pd.read_csv(io.BytesIO(raw), encoding="cp1251")
    else:
        try:
            df = pd.read_excel(io.BytesIO(raw), sheet_name=0)
        except Exception:
            # Fallback: пробуем CSV
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
            except Exception:
                df = pd.read_csv(io.BytesIO(raw), encoding="cp1251")

    df = df.loc[:, ~df.columns.duplicated()]
    return df


def df_to_csv_bytes(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()


# ==================== UI ====================
st.set_page_config(page_title="Ближайшие НП", layout="wide")
st.title("Расчёт ближайших населённых пунктов")

if "coeffs" not in st.session_state:
    st.session_state["coeffs"] = load_coefficients()

if "ds_df" not in st.session_state:
    ds_df_gh = github_read_csv(SPRAVOCHNIK_FILE)
    if ds_df_gh is not None:
        st.session_state["ds_df"] = ds_df_gh

if "kf_df" not in st.session_state:
    kf_df_gh = github_read_csv(KF_FILE)
    if kf_df_gh is not None:
        st.session_state["kf_df"] = kf_df_gh

tab_main, tab_settings = st.tabs(["Калькулятор", "⚙ Настройки калькулятора"])

# ==================== ВКЛАДКА НАСТРОЙКИ ====================
with tab_settings:
    st.markdown("## Настройки калькулятора")

    st.markdown("### Справочник НП")
    if "ds_df" in st.session_state:
        st.success(f"✅ Загружен: {len(st.session_state['ds_df'])} строк")
    else:
        st.warning("❌ Не загружен")

    user_ds_file = st.file_uploader(
        "Загрузите справочник НП (Excel/CSV)",
        type=["xlsx", "xls", "csv"],
        key="settings_ds_file"
    )
    if user_ds_file is not None:
        if st.button("💾 Сохранить справочник НП в GitHub"):
            ds_df_new = read_any(user_ds_file)
            csv_bytes = df_to_csv_bytes(ds_df_new)
            ok, msg = github_put_file(SPRAVOCHNIK_FILE, csv_bytes,
                                      "Update spravochnik.csv")
            if ok:
                st.session_state["ds_df"] = ds_df_new
                st.success(f"Сохранено: {len(ds_df_new)} строк")
                st.rerun()
            else:
                st.error(f"Ошибка: {msg}")

    st.markdown("---")

    st.markdown("### Справочник КФ (сложности + локальной)")
    if "kf_df" in st.session_state:
        st.success(f"✅ Загружен: {len(st.session_state['kf_df'])} строк")
    else:
        st.warning("❌ Не загружен")

    user_kf_file = st.file_uploader(
        "Загрузите КФ (Excel/CSV)",
        type=["xlsx", "xls", "csv"],
        key="settings_kf_file"
    )
    if user_kf_file is not None:
        if st.button("💾 Сохранить КФ в GitHub"):
            kf_df_new = read_any(user_kf_file)
            csv_bytes = df_to_csv_bytes(kf_df_new)
            ok, msg = github_put_file(KF_FILE, csv_bytes, "Update kf.csv")
            if ok:
                st.session_state["kf_df"] = kf_df_new
                st.success(f"Сохранено: {len(kf_df_new)} строк")
                st.rerun()
            else:
                st.error(f"Ошибка: {msg}")

    st.markdown("---")

    st.markdown("### Коэффициенты КФ категории")
    kf_cat = st.session_state["coeffs"]["kf_category"]
    new_kf_cat = {}
    for cat in range(1, 7):
        val = st.slider(
            f"Категория {cat}",
            min_value=0.0, max_value=2.0, step=0.05,
            value=float(kf_cat.get(cat, 0.0)),
            key=f"settings_kf_cat_{cat}"
        )
        new_kf_cat[cat] = round(val, 2)

    if st.button("💾 Сохранить коэффициенты в GitHub"):
        coeffs_to_save = {"kf_category": {str(k): v for k, v in new_kf_cat.items()}}
        content_bytes = json.dumps(coeffs_to_save, ensure_ascii=False, indent=2).encode("utf-8")
        ok, msg = github_put_file(COEFFICIENTS_FILE, content_bytes,
                                  "Update coefficients.json")
        if ok:
            st.session_state["coeffs"] = {"kf_category": new_kf_cat}
            st.success("Сохранено")
        else:
            st.error(f"Ошибка: {msg}")

    st.markdown("---")
    st.markdown("### Статус в GitHub")
    if st.button("🔄 Обновить статус"):
        for k in ["ds_df", "kf_df", "coeffs"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["coeffs"] = load_coefficients()
        st.rerun()

    for fname in [SPRAVOCHNIK_FILE, KF_FILE, COEFFICIENTS_FILE]:
        content, sha = github_get_file(fname)
        if content is not None:
            st.caption(f"✅ {fname} — {len(content)} байт")
        else:
            st.caption(f"❌ {fname} — не найден")

# ==================== ВКЛАДКА КАЛЬКУЛЯТОР ====================
with tab_main:
    st.markdown("### 1. Загрузите список НП")
    user_file = st.file_uploader(
        "Ваш Excel/CSV (колонки: Тип НП, Город, Широта, Долгота, Население, Градация, Категория, Область, ФО, СВЗ регион)",
        type=["xlsx", "xls", "csv"],
        key="user_file"
    )

    st.markdown("### 2. Введите среднюю цену в поля (обязательно)")
    avg_price_input = st.number_input(
        "Ср. цена в поля, руб",
        min_value=0.0,
        value=None,
        step=1.0,
        format="%.2f",
        key="avg_price_input"
    )

    st.markdown("### 3. (Необязательно) Загрузите СХ")
    sx_file = st.file_uploader(
        "СХ (колонки: Client, Project Name, Wave Name, Project Code, Регион, Город, Region short)",
        type=["xlsx", "xls", "csv"],
        key="sx_file"
    )

    ds_df = st.session_state.get("ds_df", None)

    if user_file and ds_df is not None:
        user_df_raw = clean_dataframe(read_any(user_file))
        user_df = remove_duplicates(user_df_raw)

        st.subheader("Ваш список")
        st.dataframe(user_df_raw.head(5), use_container_width=True)
        st.caption(f"Строк: {len(user_df_raw)}")

        required_user = ["Тип НП", "Город", "Широта", "Долгота", "Население",
                         "Градация", "Категория", "Область", "ФО", "СВЗ регион"]
        required_ds = ["settlement", "region", "latitude_dd", "longitude_dd"]

        missing_user = [c for c in required_user if c not in user_df_raw.columns]
        missing_ds = [c for c in required_ds if c not in ds_df.columns]

        if missing_user:
            st.error(f"В вашем файле нет колонок: {missing_user}")
        if missing_ds:
            st.error(f"В справочнике нет колонок: {missing_ds}")

        if not missing_user and not missing_ds:
            if st.button("1️⃣ Сопоставить"):
                for k in ["matched", "result_sheet", "avp", "avp_issues"]:
                    if k in st.session_state:
                        del st.session_state[k]

                with st.spinner("Сопоставляем..."):
                    matched = match_settlements(user_df, ds_df)

                st.session_state["matched"] = matched
                st.session_state["user_df"] = user_df

                stats = matched["match_status"].value_counts().reset_index()
                stats.columns = ["статус", "количество"]
                st.success("Готово")
                st.write("**Статусы:**")
                st.dataframe(stats, use_container_width=True)

                matched_cat_filled = ~matched["Категория"].apply(is_empty)
                nf_preview = matched[
                    ~(
                        matched["match_status"].isin(["из исходника", "из словаря", "из справочника"]) &
                        matched_cat_filled
                    )
                ]
                if len(nf_preview):
                    st.warning(f"Не в «Результат»: {len(nf_preview)} НП")
                    st.dataframe(
                        nf_preview[["Город", "Область", "match_status", "Категория"]].head(50),
                        use_container_width=True
                    )

                st.subheader("Предпросмотр")
                st.dataframe(matched.head(20), use_container_width=True)

            if "matched" in st.session_state:
                if st.button("2️⃣ Рассчитать и получить Excel"):
                    for k in ["result_sheet", "avp", "avp_issues"]:
                        if k in st.session_state:
                            del st.session_state[k]

                    work = st.session_state["matched"].copy()
                    user_df_saved = st.session_state.get("user_df", user_df)

                    work["_name_orig"] = work["Город"].astype(str).str.strip()
                    work["_region_orig"] = work["Область"].astype(str).str.strip()
                    work["_cat"] = pd.to_numeric(work["Категория"], errors="coerce")

                    work["lat"] = work["Широта"].apply(to_float)
                    work["lon"] = work["Долгота"].apply(to_float)
                    has_coords = work["lat"].notna() & work["lon"].notna()

                    def sub(cats):
                        return work[has_coords & work["_cat"].isin(cats)].copy()

                    g12 = sub([1, 2])
                    g34 = sub([3, 4])
                    g5  = sub([5])
                    g3456 = sub([3, 4, 5, 6])
                    g56   = sub([5, 6])
                    g6    = sub([6])

                    for c in NEW_COLS:
                        work[c] = None

                    if len(g3456) and len(g12):
                        names, regions, dists = find_nearest(g3456, g12)
                        work.loc[g3456.index, "Для НП 3-6 → НП 1, 2 Имя НП"] = names
                        work.loc[g3456.index, "Для НП 3-6 → НП 1, 2 Область"] = regions
                        work.loc[g3456.index, "Для НП 3-6 → НП 1, 2 расстояние"] = np.round(dists, 2)

                    if len(g56) and len(g34):
                        names, regions, dists = find_nearest(g56, g34)
                        work.loc[g56.index, "Для НП 5-6 → НП 3, 4 Имя НП"] = names
                        work.loc[g56.index, "Для НП 5-6 → НП 3, 4 Область"] = regions
                        work.loc[g56.index, "Для НП 5-6 → НП 3, 4 расстояние"] = np.round(dists, 2)

                    if len(g6) and len(g5):
                        names, regions, dists = find_nearest(g6, g5)
                        work.loc[g6.index, "Для НП 6 → НП 5 Имя НП"] = names
                        work.loc[g6.index, "Для НП 6 → НП 5 Область"] = regions
                        work.loc[g6.index, "Для НП 6 → НП 5 расстояние"] = np.round(dists, 2)

                    cat_filled = ~work["Категория"].apply(is_empty)
                    in_result = (
                        work["match_status"].isin(["из исходника", "из словаря", "из справочника"]) &
                        cat_filled
                    )

                    sheet_res = work[in_result].copy()
                    orig_cols = [c for c in user_df_saved.columns if c in sheet_res.columns]
                    sheet_res = sheet_res[orig_cols + ["match_status"] + NEW_COLS]

                    sheet_res = final_dedup(sheet_res)
                    sheet_res = compute_city_new(sheet_res)
                    sheet_res["Категория new"] = pd.to_numeric(
                        sheet_res["Категория new"], errors="coerce"
                    )

                    sheet_res = sheet_res[
                        orig_cols + ["match_status"] + NEW_COLS + FINAL_COLS
                    ]

                    sheet_nf = work[~in_result].copy()
                    nf_cols = [c for c in user_df_saved.columns if c in sheet_nf.columns]
                    sheet_nf = sheet_nf[nf_cols + ["match_status"]]

                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        sheet_res.to_excel(writer, sheet_name="Результат", index=False)
                        if len(sheet_nf):
                            sheet_nf.to_excel(writer, sheet_name="Не_найдено", index=False)

                    st.session_state["result_sheet"] = sheet_res

                    st.success("Готово!")
                    st.download_button(
                        "⬇ Скачать result.xlsx",
                        buf.getvalue(),
                        "result.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                # === Кнопка AvP ===
                can_build_avp = (
                    sx_file is not None and
                    "result_sheet" in st.session_state and
                    avg_price_input is not None and
                    avg_price_input > 0
                )

                st.markdown("---")
                if not can_build_avp:
                    reasons = []
                    if sx_file is None:
                        reasons.append("не загружен СХ")
                    if "result_sheet" not in st.session_state:
                        reasons.append("не выполнен расчёт категорий (кнопка 2️⃣)")
                    if avg_price_input is None or avg_price_input <= 0:
                        reasons.append("не введена Ср. цена в поля (блок 2)")
                    st.info(f"Кнопка «Рассчитать ср. цену анкеты» станет доступна, когда: {', '.join(reasons)}")
                else:
                    if st.button("3️⃣ Рассчитать ср. цену анкеты"):
                        with st.spinner("Считаем AvP..."):
                            sx_df = read_any(sx_file)
                            result_df = st.session_state["result_sheet"]
                            coeffs = st.session_state["coeffs"]
                            kf_df = st.session_state.get("kf_df", None)

                            try:
                                avp, issues = build_avp(
                                    sx_df, result_df,
                                    coeffs, kf_df, avg_price_input
                                )
                                st.session_state["avp"] = avp
                                st.session_state["avp_issues"] = issues
                            except ValueError as e:
                                st.error(f"Ошибка: {e}")
                                avp = None

                    if "avp" in st.session_state:
                        avp = st.session_state["avp"]
                        issues = st.session_state["avp_issues"]

                        st.success(f"Готово! Строк в AvP: {len(avp)}")

                        if len(issues["no_region"]):
                            st.warning(f"⚠ СВЗ регион пусто: {len(issues['no_region'])} строк в result")
                            st.dataframe(issues["no_region"].head(50), use_container_width=True)

                        if len(issues["duplicates"]):
                            st.warning(f"⚠ Дубликаты (Город, СВЗ регион) в result: {len(issues['duplicates'])}")
                            st.dataframe(issues["duplicates"].head(50), use_container_width=True)

                        if len(issues["no_category"]):
                            st.warning(f"⚠ Не нашли Категория new: {len(issues['no_category'])} строк")
                            st.dataframe(issues["no_category"].head(50), use_container_width=True)

                        st.subheader("Предпросмотр AvP")
                        st.dataframe(avp.head(20), use_container_width=True)

                        buf_avp = io.BytesIO()
                        with pd.ExcelWriter(buf_avp, engine="openpyxl") as writer:
                            avp.to_excel(writer, sheet_name="AvP", index=False)

                        st.download_button(
                            "⬇ Скачать AvP.xlsx",
                            buf_avp.getvalue(),
                            "AvP.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
    else:
        if not user_file:
            st.info("Загрузите список НП (блок 1).")
        if ds_df is None:
            st.info("Справочник НП не загружен. Зайдите в «⚙ Настройки калькулятора».")
