import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
import re
import io

# ---------- Константы ----------
CACHE_FILE = "coordinates.xlsx"
PHOTON_URL = "https://photon.komoot.io/api/"
HEADERS = {"User-Agent": "np-distance-calc/1.0"}
R_EARTH = 6371.0

NEW_COLS = [
    "Для НП 3,4 → НП 1, 2 Имя НП",
    "Для НП 3,4 → НП 1, 2 расстояние",
    "Для НП 5,6 → НП 3, 4 Имя НП",
    "Для НП 5,6 → НП 3, 4 расстояние",
    "Для НП 6 → НП 5 Имя НП",
    "Для НП 6 → НП 5 расстояние",
]

# ---------- Утилиты ----------
def clean_name(name):
    if pd.isna(name):
        return ""
    s = str(name).strip()
    s = re.sub(r"^(город|г\.?|пгт\.?|рп\.?|гт\.?|село|с\.?|посёлок|поселок|п\.|дп\.?|д\.?)\s+",
               "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(дп|рп|пгт)\.?$", "", s, flags=re.IGNORECASE)
    return s.strip()

def clean_region(region):
    if pd.isna(region):
        return ""
    s = str(region).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("Город федерального значения Москва", "Москва")
    s = s.replace("Город федерального значения Санкт-Петербург", "Санкт-Петербург")
    s = s.replace("Ханты-Мансийский автономный округ - Югра", "Ханты-Мансийский автономный округ")
    return s

def photon_geocode(name):
    """Photon: возвращает (lat, lon) или (None, None).
    Регион НЕ используется — Photon не умеет фильтровать по нему.
    """
    queries = [
        {"q": f"{name}, Россия", "limit": 5, "lang": "ru"},
        {"q": f"{name}", "limit": 5, "lang": "ru"},
    ]
    for params in queries:
        try:
            r = requests.get(PHOTON_URL, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                feats = r.json().get("features", [])
                for f in feats:
                    coords = f.get("geometry", {}).get("coordinates")
                    if not coords or len(coords) < 2:
                        continue
                    lon, lat = coords[0], coords[1]
                    props = f.get("properties", {})
                    cc = (props.get("countrycode") or "").upper()
                    country = (props.get("country") or "").lower()
                    if cc in ("RU", "") or "росси" in country:
                        return float(lat), float(lon)
        except Exception:
            pass
        time.sleep(0.3)
    return None, None

# ---------- Кэш (Excel) ----------
def save_cache(cache_map):
    if not cache_map:
        return
    rows = [{"name": k[0], "region": k[1], "lat": v[0], "lon": v[1]}
            for k, v in cache_map.items()]
    pd.DataFrame(rows).to_excel(CACHE_FILE, index=False, engine="openpyxl")

def load_cache_from_disk():
    if os.path.exists(CACHE_FILE):
        try:
            c = pd.read_excel(CACHE_FILE, engine="openpyxl")
            return {(str(row["name"]), str(row["region"])): (row["lat"], row["lon"])
                    for _, row in c.iterrows()}
        except Exception:
            return {}
    return {}

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
    dists = np.full(n, np.nan)
    if len(tgt_df) == 0 or n == 0:
        return names, dists
    D = haversine_matrix(src_df["lat"].values, src_df["lon"].values,
                         tgt_df["lat"].values, tgt_df["lon"].values)
    idx = D.argmin(axis=1)
    names = tgt_df["_name_clean"].values[idx]
    dists = D[np.arange(n), idx]
    return names, dists

# ---------- UI ----------
st.set_page_config(page_title="Ближайшие НП", layout="wide")
st.title("Расчёт ближайших населённых пунктов")

with st.sidebar:
    st.markdown("### Кэш координат")
    st.caption(f"Файл на диске: `{CACHE_FILE}`")
    if st.button("🗑 Очистить кэш"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        st.success("Кэш очищен")

# --- Загрузчик исходного файла ---
uploaded = st.file_uploader(
    "Загрузите Excel или CSV с населёнными пунктами",
    type=["xlsx", "xls", "csv"],
    key="data_upload"
)

if uploaded:
    # --- Чтение файла ---
    if uploaded.name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(uploaded, encoding="utf-8-sig")
        except (UnicodeDecodeError, UnicodeError):
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding="cp1251")
    else:
        df = pd.read_excel(uploaded, sheet_name=0)

    st.subheader("Предпросмотр")
    st.dataframe(df.head(15), use_container_width=True)
    st.caption(f"Всего строк: {len(df)}")

    cols = df.columns.tolist()

    def pick(name, default=0):
        for i, c in enumerate(cols):
            if str(c).strip().lower() == name.lower():
                return i
        return default

    name_col   = st.selectbox("Колонка с названием НП", cols, index=pick("город"))
    region_col = st.selectbox("Колонка с областью",   cols, index=pick("область"))
    cat_col    = st.selectbox("Колонка с категорией", cols, index=pick("категория"))

    # --- Шаг 1. Геокодинг ---
    if st.button("1️⃣ Геокодировать (Photon)"):
        work = df.copy()
        work["_name_clean"]   = work[name_col].apply(clean_name)
        work["_region_clean"] = work[region_col].apply(clean_region)

        skip_mask = work["_name_clean"].eq("") | work["_region_clean"].eq("")
        to_geo = work[~skip_mask].copy()

        cache = load_cache_from_disk()

        lats, lons = [], []
        progress = st.progress(0.0)
        status = st.empty()
        not_found = []

        for i, row in to_geo.iterrows():
            key = (row["_name_clean"], row["_region_clean"])
            if key in cache and cache[key][0] is not None and not pd.isna(cache[key][0]):
                lat, lon = cache[key]
            else:
                lat, lon = photon_geocode(row["_name_clean"])
                if lat is not None:
                    cache[key] = (lat, lon)
                    if len(cache) % 20 == 0:
                        save_cache(cache)

            if lat is None:
                not_found.append((row["_name_clean"], row["_region_clean"]))

            lats.append(lat if lat is not None else np.nan)
            lons.append(lon if lon is not None else np.nan)
            progress.progress(min(len(lats) / max(len(to_geo), 1), 1.0))
            status.text(f"Геокодинг {len(lats)}/{len(to_geo)} — не найдено: {len(not_found)}")

        save_cache(cache)

        work["lat"] = np.nan
        work["lon"] = np.nan
        work.loc[to_geo.index, "lat"] = lats
        work.loc[to_geo.index, "lon"] = lons

        st.session_state["df_geo"] = work
        st.session_state["not_found"] = not_found
        st.session_state["cols_used"] = (name_col, region_col, cat_col)
        st.session_state["orig_cols"] = list(df.columns)

        st.success(f"Готово. Не найдено: {len(not_found)}")
        if not_found:
            st.warning("Не удалось геокодировать:")
            st.dataframe(pd.DataFrame(not_found, columns=["Название", "Область"]),
                         use_container_width=True)

        buf = io.BytesIO()
        work.to_excel(buf, index=False, engine="openpyxl")
        st.download_button("⬇ Скачать geocoded.xlsx", buf.getvalue(),
                           "geocoded.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # --- Шаг 2. Расчёт ---
    if "df_geo" in st.session_state:
        if st.button("2️⃣ Рассчитать и получить Excel"):
            work = st.session_state["df_geo"].copy()
            name_col, region_col, cat_col = st.session_state["cols_used"]
            orig_cols = st.session_state["orig_cols"]

            work["_cat"] = pd.to_numeric(work[cat_col], errors="coerce")
            has_coords = work["lat"].notna() & work["lon"].notna()

            def sub(cats):
                return work[has_coords & work["_cat"].isin(cats)].copy()

            g12 = sub([1, 2])
            g34 = sub([3, 4])
            g5  = sub([5])
            g56 = sub([5, 6])
            g6  = sub([6])

            for c in NEW_COLS:
                work[c] = None

            if len(g34) and len(g12):
                names, dists = find_nearest(g34, g12)
                work.loc[g34.index, NEW_COLS[0]] = names
                work.loc[g34.index, NEW_COLS[1]] = np.round(dists, 2)

            if len(g56) and len(g34):
                names, dists = find_nearest(g56, g34)
                work.loc[g56.index, NEW_COLS[2]] = names
                work.loc[g56.index, NEW_COLS[3]] = np.round(dists, 2)

            if len(g6) and len(g5):
                names, dists = find_nearest(g6, g5)
                work.loc[g6.index, NEW_COLS[4]] = names
                work.loc[g6.index, NEW_COLS[5]] = np.round(dists, 2)

            result = work.drop(columns=["_name_clean", "_region_clean", "_cat"],
                               errors="ignore")

            sheet_all = result.copy()
            safe_orig = [c for c in orig_cols if c in result.columns]
            sheet_res = result[safe_orig + NEW_COLS].copy()

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                sheet_all.to_excel(writer, sheet_name="Все_НП", index=False)
                sheet_res.to_excel(writer, sheet_name="Результат", index=False)

            st.success("Готово!")
            st.subheader("Предпросмотр «Результат»")
            st.dataframe(sheet_res.head(30), use_container_width=True)

            st.download_button(
                "⬇ Скачать result.xlsx",
                buf.getvalue(),
                "result.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
