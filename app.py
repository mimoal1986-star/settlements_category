import streamlit as st
import pandas as pd
import numpy as np
import re
import io

R_EARTH = 6371.0

NEW_COLS = [
    "Для НП 3,4 → НП 1, 2 Имя НП",
    "Для НП 3,4 → НП 1, 2 расстояние",
    "Для НП 5,6 → НП 3, 4 Имя НП",
    "Для НП 5,6 → НП 3, 4 расстояние",
    "Для НП 6 → НП 5 Имя НП",
    "Для НП 6 → НП 5 расстояние",
]

# ==================== ШАГИ НОРМАЛИЗАЦИИ ====================
def _squeeze(s):
    return re.sub(r"\s+", " ", str(s).strip())

def _yo(s):
    return s.replace("ё", "е").replace("Ё", "Е")

def _dash(s):
    return s.replace("—", "-").replace("–", "-")

def _dash_spaces(s):
    return re.sub(r"\s*-\s*", "-", s)

def _region_tails(s):
    return re.sub(r"\s+без\s+[а-я\-\s]+$", "", s)

def _region_words(s):
    s = re.sub(
        r"\b(область|обл|край|республика|респ|автономный округ|ао|"
        r"город федерального значения|город|г)\b",
        "", s
    )
    return re.sub(r"\s+", " ", s).strip(" -,")

def _name_type(s):
    s = re.sub(
        r"^(город|г|пгт|рп|гт|село|с|посёлок|поселок|п|дп|д|кп|к\.п\.|"
        r"деревня|д\.)\s*\.?\s+",
        "", s
    )
    s = re.sub(r"\s+(дп|рп|пгт|г|кп|гт)\.?$", "", s)
    return re.sub(r"\s+", " ", s).strip()

def _name_im(s):
    s = re.sub(r"\bим\.\s*", "", s)
    s = re.sub(r"\bимени\s+", "", s)
    s = re.sub(r"\bим\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()

def _name_initials(s):
    s = re.sub(r"\b[а-я]\.\s*[а-я]\.\s*", "", s)
    s = re.sub(r"\b[а-я]\.\s*", "", s)
    return re.sub(r"\s+", " ", s).strip()


REGION_STEPS = [_squeeze, _yo, _dash, _dash_spaces, _region_tails, _region_words]
NAME_STEPS   = [_squeeze, _yo, _dash, _dash_spaces, _name_type, _name_im, _name_initials]


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
    dists = np.full(n, np.nan)
    if len(tgt_df) == 0 or n == 0:
        return names, dists
    D = haversine_matrix(src_df["lat"].values, src_df["lon"].values,
                         tgt_df["lat"].values, tgt_df["lon"].values)
    idx = D.argmin(axis=1)
    names = tgt_df["_name_orig"].values[idx]
    dists = D[np.arange(n), idx]
    return names, dists


# ==================== МАТЧИНГ ====================
def match_settlements(user_df, dataset_df,
                      user_name_col, user_region_col,
                      ds_settlement_col, ds_municipality_col,
                      ds_region_col, ds_lat_col, ds_lon_col):
    """
    Пошаговый матчинг:
    - 10 уровней нормализации (накопительно): ур.0 = squeeze, ур.1 = +ё→е, и т.д.
    - на каждом уровне проверяем settlement+region и municipality+region параллельно
    - приоритет: settlement
    - ключ всегда (name, region), регион обязателен
    """
    ds = dataset_df.reset_index(drop=True).copy()

    # --- Предвычисляем уровни нормализации датасета ---
    sett_cur = ds[ds_settlement_col].astype(str).apply(_squeeze).copy()
    reg_cur  = ds[ds_region_col].astype(str).apply(_squeeze).copy()
    muni_cur = ds[ds_municipality_col].astype(str).apply(_squeeze).copy() if ds_municipality_col else None

    ds_sett_levels = [sett_cur.copy()]
    ds_muni_levels = [muni_cur.copy() if muni_cur is not None else None]
    ds_reg_levels  = [reg_cur.copy()]

    max_steps = max(len(REGION_STEPS), len(NAME_STEPS))
    for step in range(max_steps):
        if step < len(REGION_STEPS):
            reg_cur = reg_cur.apply(REGION_STEPS[step])
        if step < len(NAME_STEPS):
            sett_cur = sett_cur.apply(NAME_STEPS[step])
            if muni_cur is not None:
                muni_cur = muni_cur.apply(NAME_STEPS[step])
        ds_sett_levels.append(sett_cur.copy())
        ds_muni_levels.append(muni_cur.copy() if muni_cur is not None else None)
        ds_reg_levels.append(reg_cur.copy())

    n_levels = len(ds_sett_levels)

    # --- Строим словари на каждом уровне (векторизованно) ---
    level_maps = []
    lats_arr = ds[ds_lat_col].values
    lons_arr = ds[ds_lon_col].values

    for lvl in range(n_levels):
        sett = ds_sett_levels[lvl].values
        reg  = ds_reg_levels[lvl].values
        muni = ds_muni_levels[lvl].values if ds_muni_levels[lvl] is not None else None

        sett_map = {}
        muni_map = {}

        for s_name, r_name, lat, lon in zip(sett, reg, lats_arr, lons_arr):
            if not s_name or not r_name:
                continue
            key = (s_name, r_name)
            if key not in sett_map:
                sett_map[key] = (lat, lon)

        if muni is not None:
            for m_name, r_name, lat, lon in zip(muni, reg, lats_arr, lons_arr):
                if not m_name or not r_name:
                    continue
                key = (m_name, r_name)
                if key not in muni_map:
                    muni_map[key] = (lat, lon)

        level_maps.append({"sett": sett_map, "muni": muni_map})

    # --- Матчинг ---
    out = user_df.copy()
    out["lat"] = np.nan
    out["lon"] = np.nan
    out["match_status"] = ""

    for i, row in out.iterrows():
        u_name_raw = row[user_name_col]
        u_reg_raw  = row[user_region_col]

        u_name = _squeeze(u_name_raw) if not pd.isna(u_name_raw) else ""
        u_reg  = _squeeze(u_reg_raw) if not pd.isna(u_reg_raw) else ""

        if not u_name or not u_reg:
            out.at[i, "match_status"] = "пусто в исходнике"
            continue

        u_name = u_name.lower()
        u_reg = u_reg.lower()

        found = False
        for lvl in range(n_levels):
            reg_work = u_reg
            for step in range(lvl):
                if step < len(REGION_STEPS):
                    reg_work = REGION_STEPS[step](reg_work)

            name_work = u_name
            for step in range(lvl):
                if step < len(NAME_STEPS):
                    name_work = NAME_STEPS[step](name_work)

            key = (name_work, reg_work)

            # 1. settlement
            m = level_maps[lvl]["sett"]
            if key in m:
                lat, lon = m[key]
                out.at[i, "lat"] = lat
                out.at[i, "lon"] = lon
                out.at[i, "match_status"] = f"settlement, ур.{lvl}"
                found = True
                break

            # 2. municipality
            if ds_municipality_col:
                m2 = level_maps[lvl]["muni"]
                if key in m2:
                    lat, lon = m2[key]
                    out.at[i, "lat"] = lat
                    out.at[i, "lon"] = lon
                    out.at[i, "match_status"] = f"municipality, ур.{lvl}"
                    found = True
                    break

        if not found:
            out.at[i, "match_status"] = "не найдено"

    return out


# ==================== UI ====================
st.set_page_config(page_title="Ближайшие НП", layout="wide")
st.title("Расчёт ближайших населённых пунктов")

st.markdown("### 1. Загрузите список НП")
user_file = st.file_uploader(
    "Ваш Excel/CSV со списком НП",
    type=["xlsx", "xls", "csv"],
    key="user_file"
)

st.markdown("### 2. Загрузите датасет с координатами")
ds_file = st.file_uploader(
    "Датасет (region, municipality, settlement, latitude_dd, longitude_dd)",
    type=["xlsx", "xls", "csv"],
    key="ds_file"
)


def read_any(f):
    raw = f.read()
    if f.name.lower().endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        except (UnicodeDecodeError, UnicodeError):
            return pd.read_csv(io.BytesIO(raw), encoding="cp1251")
    return pd.read_excel(io.BytesIO(raw), sheet_name=0)


if user_file and ds_file:
    user_df = read_any(user_file)
    ds_df = read_any(ds_file)

    st.subheader("Ваш список")
    st.dataframe(user_df.head(5), use_container_width=True)
    st.caption(f"Строк: {len(user_df)}")

    st.subheader("Датасет с координатами")
    st.dataframe(ds_df.head(5), use_container_width=True)
    st.caption(f"Строк: {len(ds_df)}")

    st.markdown("### 3. Укажите колонки")

    u_cols = user_df.columns.tolist()
    d_cols = ds_df.columns.tolist()

    def pick(cols, name, default=0):
        for i, c in enumerate(cols):
            if str(c).strip().lower() == name.lower():
                return i
        return default

    c1, c2, c3 = st.columns(3)
    with c1:
        u_name_col   = st.selectbox("Ваш: название", u_cols, index=pick(u_cols, "город"))
        u_region_col = st.selectbox("Ваш: регион",   u_cols, index=pick(u_cols, "область"))
        u_cat_col    = st.selectbox("Ваш: категория", u_cols, index=pick(u_cols, "категория"))
    with c2:
        d_name_col   = st.selectbox("Датасет: settlement", d_cols, index=pick(d_cols, "settlement"))
        d_muni_col   = st.selectbox("Датасет: municipality",
                                    ["— нет —"] + d_cols,
                                    index=(["— нет —"] + d_cols).index("municipality")
                                          if "municipality" in d_cols else 0)
        d_muni_col = None if d_muni_col == "— нет —" else d_muni_col
        d_region_col = st.selectbox("Датасет: region", d_cols, index=pick(d_cols, "region"))
    with c3:
        d_lat_col = st.selectbox("Датасет: широта",  d_cols, index=pick(d_cols, "latitude_dd"))
        d_lon_col = st.selectbox("Датасет: долгота", d_cols, index=pick(d_cols, "longitude_dd"))

    # --- Матчинг ---
    if st.button("1️⃣ Сопоставить и получить координаты"):
        with st.spinner("Сопоставляем..."):
            matched = match_settlements(
                user_df, ds_df,
                u_name_col, u_region_col,
                d_name_col, d_muni_col,
                d_region_col, d_lat_col, d_lon_col,
            )

        st.session_state["matched"]      = matched
        st.session_state["user_df"]      = user_df
        st.session_state["u_cat_col"]    = u_cat_col
        st.session_state["u_name_col"]   = u_name_col
        st.session_state["u_region_col"] = u_region_col

        stats = matched["match_status"].value_counts().reset_index()
        stats.columns = ["статус", "количество"]
        st.success("Готово")
        st.write("**Статусы сопоставления:**")
        st.dataframe(stats, use_container_width=True)

        not_found = matched[matched["match_status"] == "не найдено"]
        if len(not_found):
            st.warning(f"Не найдено: {len(not_found)} НП")
            st.dataframe(not_found[[u_name_col, u_region_col]].head(50),
                         use_container_width=True)

        st.subheader("Предпросмотр результата")
        st.dataframe(matched.head(20), use_container_width=True)

    # --- Расчёт ---
    if "matched" in st.session_state:
        if st.button("2️⃣ Рассчитать и получить Excel"):
            work = st.session_state["matched"].copy()
            user_df = st.session_state["user_df"]
            u_cat_col = st.session_state["u_cat_col"]
            u_name_col = st.session_state["u_name_col"]

            work["_name_orig"]  = work[u_name_col].astype(str)
            work["_cat"] = pd.to_numeric(work[u_cat_col], errors="coerce")
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

            orig_cols = [c for c in user_df.columns if c in work.columns]
            sheet_res = work[orig_cols + ["lat", "lon", "match_status"] + NEW_COLS].copy()

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                sheet_res.to_excel(writer, sheet_name="Результат", index=False)
                not_found = work[work["match_status"] == "не найдено"]
                if len(not_found):
                    cols = [st.session_state["u_name_col"],
                            st.session_state["u_region_col"]]
                    not_found[cols].to_excel(
                        writer, sheet_name="Не_найдено", index=False)

            st.success("Готово!")
            st.download_button(
                "⬇ Скачать result.xlsx",
                buf.getvalue(),
                "result.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
