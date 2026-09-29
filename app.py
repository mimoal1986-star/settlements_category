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


def squeeze(s):
    """Только сжатие пробелов и strip."""
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).strip())


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


def match_settlements(user_df, ds_df):
    """
    Простой матчинг: ключ (город, область) == (settlement, region).
    Только сжатие пробелов. Регион обязателен.
    """
    # Нормализуем датасет
    ds = ds_df.copy()
    ds["_s"] = ds["settlement"].apply(squeeze).str.lower()
    ds["_r"] = ds["region"].apply(squeeze).str.lower()

    # Словарь: (settlement, region) → (lat, lon)
    sett_map = {}
    for s, r, lat, lon in zip(ds["_s"].values, ds["_r"].values,
                              ds["latitude_dd"].values, ds["longitude_dd"].values):
        if not s or not r:
            continue
        key = (s, r)
        if key not in sett_map:
            sett_map[key] = (lat, lon)

    # Матчим
    out = user_df.copy()
    out["lat"] = np.nan
    out["lon"] = np.nan
    out["match_status"] = ""

    for i, row in out.iterrows():
        u_name = squeeze(row["город"]).lower()
        u_reg  = squeeze(row["область"]).lower()

        if not u_name or not u_reg:
            out.at[i, "match_status"] = "пусто в исходнике"
            continue

        key = (u_name, u_reg)
        if key in sett_map:
            lat, lon = sett_map[key]
            out.at[i, "lat"] = lat
            out.at[i, "lon"] = lon
            out.at[i, "match_status"] = "найдено"
        else:
            out.at[i, "match_status"] = "не найдено"

    return out


def read_any(f):
    raw = f.read()
    if f.name.lower().endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        except (UnicodeDecodeError, UnicodeError):
            return pd.read_csv(io.BytesIO(raw), encoding="cp1251")
    return pd.read_excel(io.BytesIO(raw), sheet_name=0)


# ==================== UI ====================
st.set_page_config(page_title="Ближайшие НП", layout="wide")
st.title("Расчёт ближайших населённых пунктов")

st.markdown("### 1. Загрузите список НП")
user_file = st.file_uploader(
    "Ваш Excel/CSV со списком НП (колонки: город, область, категория)",
    type=["xlsx", "xls", "csv"],
    key="user_file"
)

st.markdown("### 2. Загрузите справочник")
ds_file = st.file_uploader(
    "Справочник (колонки: settlement, region, latitude_dd, longitude_dd)",
    type=["xlsx", "xls", "csv"],
    key="ds_file"
)

if user_file and ds_file:
    user_df = read_any(user_file)
    ds_df = read_any(ds_file)

    st.subheader("Ваш список")
    st.dataframe(user_df.head(5), use_container_width=True)
    st.caption(f"Строк: {len(user_df)}")

    st.subheader("Справочник")
    st.dataframe(ds_df.head(5), use_container_width=True)
    st.caption(f"Строк: {len(ds_df)}")

    # Проверка, что нужные колонки есть
    required_user = ["город", "область", "категория"]
    required_ds = ["settlement", "region", "latitude_dd", "longitude_dd"]

    missing_user = [c for c in required_user if c not in user_df.columns]
    missing_ds = [c for c in required_ds if c not in ds_df.columns]

    if missing_user:
        st.error(f"В вашем файле нет колонок: {missing_user}")
    if missing_ds:
        st.error(f"В справочнике нет колонок: {missing_ds}")

    if not missing_user and not missing_ds:
        # --- Матчинг ---
        if st.button("1️⃣ Сопоставить"):
            with st.spinner("Сопоставляем..."):
                matched = match_settlements(user_df, ds_df)

            st.session_state["matched"] = matched
            st.session_state["user_df"] = user_df

            stats = matched["match_status"].value_counts().reset_index()
            stats.columns = ["статус", "количество"]
            st.success("Готово")
            st.write("**Статусы:**")
            st.dataframe(stats, use_container_width=True)

            nf = matched[matched["match_status"] == "не найдено"]
            if len(nf):
                st.warning(f"Не найдено: {len(nf)} НП")
                st.dataframe(nf[["город", "область"]].head(50),
                             use_container_width=True)

            st.subheader("Предпросмотр")
            st.dataframe(matched.head(20), use_container_width=True)

        # --- Расчёт ---
        if "matched" in st.session_state:
            if st.button("2️⃣ Рассчитать и получить Excel"):
                work = st.session_state["matched"].copy()
                user_df = st.session_state["user_df"]

                work["_name_orig"] = work["город"].astype(str)
                work["_cat"] = pd.to_numeric(work["категория"], errors="coerce")
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
                    nf = work[work["match_status"] == "не найдено"]
                    if len(nf):
                        nf[["город", "область"]].to_excel(
                            writer, sheet_name="Не_найдено", index=False)

                st.success("Готово!")
                st.download_button(
                    "⬇ Скачать result.xlsx",
                    buf.getvalue(),
                    "result.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
