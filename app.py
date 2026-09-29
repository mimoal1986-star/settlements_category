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

# ---------- Нормализация ----------
def norm_region(s):
    """Нормализует название региона для сопоставления."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("ё", "е")
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s*-\s*", "-", s)              # FIX: "Осетия - Алания" → "Осетия-Алания"
    s = re.sub(r"\s+", " ", s)
    # Убираем общие слова (без мёртвого "г\.")
    s = re.sub(
        r"\b(область|обл|край|республика|респ|автономный округ|ао|"
        r"город федерального значения|город|г)\b",
        "", s
    )
    s = re.sub(r"\s+", " ", s).strip(" -,")
    return s


def norm_name(s):
    """Нормализует название НП для сопоставления."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("ё", "е")
    s = s.replace("—", "-").replace("–", "-")

    # FIX: "им." / "имени" — приводим к единому виду, потом убираем
    s = re.sub(r"\bим\.\s*", "", s)
    s = re.sub(r"\bимени\s+", "", s)
    # FIX: убираем инициалы "в.и." → ""
    s = re.sub(r"\b[а-я]\.\s*[а-я]\.\s*", "", s)
    s = re.sub(r"\b[а-я]\.\s*", "", s)

    # Префиксы типа "город", "пгт", "рп", "дп" и т.д. — в начале
    s = re.sub(
        r"^(город|г|пгт|рп|гт|село|с|посёлок|поселок|п|дп|д|кп|к\.п\.|"
        r"деревня|д\.)\s*\.?\s+",
        "", s
    )
    # FIX: добавили "кп", "гт" в финальный regex
    s = re.sub(r"\s+(дп|рп|пгт|г|кп|гт)\.?$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


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
    """Возвращает (names, dists). Имена — из колонки _name_orig (оригинальные)."""
    n = len(src_df)
    names = np.array([None] * n, dtype=object)
    dists = np.full(n, np.nan)
    if len(tgt_df) == 0 or n == 0:
        return names, dists
    D = haversine_matrix(src_df["lat"].values, src_df["lon"].values,
                         tgt_df["lat"].values, tgt_df["lon"].values)
    idx = D.argmin(axis=1)
    # FIX: возвращаем оригинальные имена, а не нормализованные
    names = tgt_df["_name_orig"].values[idx]
    dists = D[np.arange(n), idx]
    return names, dists


# ---------- Матчинг ----------
def match_settlements(user_df, dataset_df,
                      user_name_col, user_region_col,
                      ds_name_col, ds_region_col,
                      ds_lat_col, ds_lon_col,
                      ds_pop_col=None):
    """
    Возвращает user_df с колонками lat, lon, match_status.
    Проходы:
      1. Точное (name_norm + region_norm)
      2. name_norm + region_norm, если в регионе один вариант
      3. name_norm уникально во всей базе
      4. Дубли в регионе → помечаем, координаты не заполняем
    """
    ds = dataset_df.copy()
    ds["_name_norm"] = ds[ds_name_col].apply(norm_name)
    ds["_region_norm"] = ds[ds_region_col].apply(norm_region)
    ds = ds[ds["_name_norm"] != ""].copy()

    # Словарь 1: (name, region) → список (lat, lon, population)
    exact_map = {}
    for _, r in ds.iterrows():
        key = (r["_name_norm"], r["_region_norm"])
        exact_map.setdefault(key, []).append(
            (r[ds_lat_col], r[ds_lon_col],
             r[ds_pop_col] if ds_pop_col else 0)
        )

    # Словарь 2: name → список (region, lat, lon, population)
    name_map = {}
    for _, r in ds.iterrows():
        name_map.setdefault(r["_name_norm"], []).append(
            (r["_region_norm"], r[ds_lat_col], r[ds_lon_col],
             r[ds_pop_col] if ds_pop_col else 0)
        )

    out = user_df.copy()
    out["lat"] = np.nan
    out["lon"] = np.nan
    out["match_status"] = ""

    for i, row in out.iterrows():
        u_name = norm_name(row[user_name_col])
        u_region = norm_region(row[user_region_col])

        if not u_name or not u_region:
            out.at[i, "match_status"] = "пусто в исходнике"
            continue

        # Проход 1 + 2: точное по (name, region)
        key = (u_name, u_region)
        if key in exact_map:
            variants = exact_map[key]
            if len(variants) == 1:
                lat, lon, _ = variants[0]
                out.at[i, "lat"] = lat
                out.at[i, "lon"] = lon
                out.at[i, "match_status"] = "точно"
            else:
                # FIX: дубли в датасете — берём с наибольшим населением,
                # но помечаем
                best = max(variants, key=lambda v: v[2] if pd.notna(v[2]) else 0)
                out.at[i, "lat"] = best[0]
                out.at[i, "lon"] = best[1]
                out.at[i, "match_status"] = f"дубль в датасете ({len(variants)})"
            continue

        # Проход 3: по имени, если в регионе один кандидат
        candidates = name_map.get(u_name, [])
        in_region = [c for c in candidates if c[0] == u_region]
        if len(in_region) == 1:
            out.at[i, "lat"] = in_region[0][1]
            out.at[i, "lon"] = in_region[0][2]
            out.at[i, "match_status"] = "по названию+региону"
            continue
        if len(in_region) > 1:
            # FIX: дубли в регионе — берём крупнейший, помечаем
            best = max(in_region, key=lambda v: v[3] if pd.notna(v[3]) else 0)
            out.at[i, "lat"] = best[1]
            out.at[i, "lon"] = best[2]
            out.at[i, "match_status"] = f"дубль в регионе ({len(in_region)})"
            continue

        # Проход 4: по имени без региона, если во всей базе один вариант
        if len(candidates) == 1:
            out.at[i, "lat"] = candidates[0][1]
            out.at[i, "lon"] = candidates[0][2]
            out.at[i, "match_status"] = "по названию (без региона)"
            continue

        out.at[i, "match_status"] = "не найдено"

    return out


# ---------- UI ----------
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
    "Датасет (region, settlement, latitude_dd, longitude_dd)",
    type=["xlsx", "xls", "csv"],
    key="ds_file"
)


# FIX: чтение через BytesIO — надёжнее, чем seek
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
        d_name_col   = st.selectbox("Датасет: название", d_cols, index=pick(d_cols, "settlement"))
        d_region_col = st.selectbox("Датасет: регион",   d_cols, index=pick(d_cols, "region"))
    with c3:
        d_lat_col = st.selectbox("Датасет: широта",  d_cols, index=pick(d_cols, "latitude_dd"))
        d_lon_col = st.selectbox("Датасет: долгота", d_cols, index=pick(d_cols, "longitude_dd"))
        d_pop_col = st.selectbox(
            "Датасет: население (опц.)",
            ["— нет —"] + d_cols,
            index=(["— нет —"] + d_cols).index("population")
                  if "population" in d_cols else 0
        )
        d_pop_col = None if d_pop_col == "— нет —" else d_pop_col

    # --- Матчинг ---
    if st.button("1️⃣ Сопоставить и получить координаты"):
        with st.spinner("Сопоставляем..."):
            matched = match_settlements(
                user_df, ds_df,
                u_name_col, u_region_col,
                d_name_col, d_region_col,
                d_lat_col, d_lon_col,
                d_pop_col,
            )
        # FIX: сохраняем user_df и все нужные колонки в session_state
        st.session_state["matched"]     = matched
        st.session_state["user_df"]     = user_df
        st.session_state["u_cat_col"]   = u_cat_col
        st.session_state["u_name_col"]  = u_name_col
        st.session_state["u_region_col"] = u_region_col

        # FIX: правильная статистика
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
            user_df = st.session_state["user_df"]   # FIX: из session_state
            u_cat_col = st.session_state["u_cat_col"]
            u_name_col = st.session_state["u_name_col"]

            # FIX: оригинальное имя для вывода + нормализованное для матчинга
            work["_name_orig"]  = work[u_name_col].astype(str)
            work["_name_clean"] = work[u_name_col].apply(norm_name)
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

            # Собираем финальные колонки
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
