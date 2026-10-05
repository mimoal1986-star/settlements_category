import streamlit as st
import pandas as pd
import numpy as np
import re
import io

R_EARTH = 6371.0

NEW_COLS = [
    "Для НП 3-6 → НП 1, 2 Имя НП",
    "Для НП 3-6 → НП 1, 2 расстояние",
    "Для НП 5-6 → НП 3, 4 Имя НП",
    "Для НП 5-6 → НП 3, 4 расстояние",
    "Для НП 6 → НП 5 Имя НП",
    "Для НП 6 → НП 5 расстояние",
]

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


def squeeze(s):
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).strip())


def is_empty(v):
    """True, если значение пустое (None, NaN или пустая строка)."""
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if pd.isna(v):
        return True
    return str(v).strip() == ""


def population_to_gradation(pop):
    if pd.isna(pop):
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
    if pd.isna(grad):
        return None
    return GRAD_TO_CAT.get(str(grad).strip(), None)


def remove_duplicates(user_df):
    """
    Удаляет дубликаты по (Город, Область), сохраняя исходный порядок:
      - все дубликаты с пустой Категорией → 1 строка
      - часть пустых, часть заполненных → удаляем только пустые
      - несколько заполненных → оставляем все
    """
    df = user_df.copy()
    df["_cat_filled"] = df["Категория"].apply(lambda v: not is_empty(v))

    # Собираем keep в порядке появления
    keep_set = set()
    # Проходим в исходном порядке: для каждой группы запоминаем первую "хорошую"
    # или первую вообще
    groups_seen = {}
    for idx, row in df.iterrows():
        key = (squeeze(row["Город"]).lower(), squeeze(row["Область"]).lower())
        groups_seen.setdefault(key, []).append(idx)

    for key, indices in groups_seen.items():
        filled = [i for i in indices if df.at[i, "_cat_filled"]]
        if filled:
            keep_set.update(filled)
        else:
            keep_set.add(indices[0])

    result = df[df.index.isin(keep_set)].drop(columns=["_cat_filled"])
    return result.reset_index(drop=True)


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
    Матчинг:
      1. Словарь CITY_COORDS (приоритет)
      2. Справочник по (settlement, region)
    Обогащает Население/Градацию/Категорию из справочника, если пусто.
    """
    def to_float(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return np.nan
        try:
            if pd.isna(v):
                return np.nan
        except (TypeError, ValueError):
            pass
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        try:
            return float(str(v).replace(",", ".").strip())
        except (ValueError, TypeError):
            return np.nan

    ds = ds_df.copy()
    ds["_s"] = ds["settlement"].apply(squeeze).str.lower()
    ds["_r"] = ds["region"].apply(squeeze).str.lower()
    ds["_lat"] = [to_float(v) for v in ds["latitude_dd"].values]
    ds["_lon"] = [to_float(v) for v in ds["longitude_dd"].values]

    # population может отсутствовать
    if "population" in ds.columns:
        ds["_pop"] = [to_float(v) for v in ds["population"].values]
    else:
        ds["_pop"] = [np.nan] * len(ds)

    # Словарь (settlement, region) → (lat, lon, pop)
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
    out["lat"] = np.nan
    out["lon"] = np.nan
    out["match_status"] = ""

    for i, row in out.iterrows():
        u_name = squeeze(row["Город"]).lower()
        u_reg  = squeeze(row["Область"]).lower()

        if not u_name or not u_reg:
            out.at[i, "match_status"] = "пусто в исходнике"
            continue

        # 1. Словарь
        if u_name in CITY_COORDS:
            lat, lon = CITY_COORDS[u_name]
            out.at[i, "lat"] = lat
            out.at[i, "lon"] = lon
            out.at[i, "match_status"] = "из словаря"
            continue

        # 2. Справочник
        key = (u_name, u_reg)
        if key in sett_map:
            lat, lon, pop = sett_map[key]
            out.at[i, "lat"] = lat
            out.at[i, "lon"] = lon
            out.at[i, "match_status"] = "из справочника"

            # --- Обогащение: Население ---
            cur_pop = out.at[i, "Население"]
            if is_empty(cur_pop) and not pd.isna(pop):
                out.at[i, "Население"] = pop

            # --- Обогащение: Градация ---
            cur_grad = out.at[i, "Градация"]
            if is_empty(cur_grad):
                pop_use = out.at[i, "Население"]
                grad = population_to_gradation(pop_use)
                if grad is not None:
                    out.at[i, "Градация"] = grad

            # --- Обогащение: Категория ---
            cur_cat = out.at[i, "Категория"]
            if is_empty(cur_cat):
                grad_use = out.at[i, "Градация"]
                cat = gradation_to_category(grad_use)
                if cat is not None:
                    out.at[i, "Категория"] = cat

            continue

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
    "Ваш Excel/CSV (колонки: Тип НП, Город, Население, Градация, Категория, Область, ФО, СВЗ регион)",
    type=["xlsx", "xls", "csv"],
    key="user_file"
)

st.markdown("### 2. Загрузите справочник")
ds_file = st.file_uploader(
    "Справочник (колонки: settlement, region, latitude_dd, longitude_dd, population)",
    type=["xlsx", "xls", "csv"],
    key="ds_file"
)

if user_file and ds_file:
    user_df_raw = read_any(user_file)
    ds_df = read_any(ds_file)

    st.subheader("Ваш список")
    st.dataframe(user_df_raw.head(5), use_container_width=True)
    st.caption(f"Строк: {len(user_df_raw)}")

    st.subheader("Справочник")
    st.dataframe(ds_df.head(5), use_container_width=True)
    st.caption(f"Строк: {len(ds_df)}")

    required_user = ["Тип НП", "Город", "Население", "Градация", "Категория", "Область", "ФО", "СВЗ регион"]
    required_ds = ["settlement", "region", "latitude_dd", "longitude_dd"]

    missing_user = [c for c in required_user if c not in user_df_raw.columns]
    missing_ds = [c for c in required_ds if c not in ds_df.columns]

    if missing_user:
        st.error(f"В вашем файле нет колонок: {missing_user}")
    if missing_ds:
        st.error(f"В справочнике нет колонок: {missing_ds}")

    if not missing_user and not missing_ds:
        # Удаление дубликатов (тихо)
        user_df = remove_duplicates(user_df_raw)

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

            nf = matched[matched["match_status"].isin(["не найдено", "пусто в исходнике"])]
            if len(nf):
                st.warning(f"Не найдено / пусто: {len(nf)} НП")
                st.dataframe(nf[["Город", "Область"]].head(50),
                             use_container_width=True)

            st.subheader("Предпросмотр")
            st.dataframe(matched.head(20), use_container_width=True)

        if "matched" in st.session_state:
            if st.button("2️⃣ Рассчитать и получить Excel"):
                work = st.session_state["matched"].copy()
                user_df = st.session_state["user_df"]

                work["_name_orig"] = work["Город"].astype(str)
                work["_cat"] = pd.to_numeric(work["Категория"], errors="coerce")
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
                    names, dists = find_nearest(g3456, g12)
                    work.loc[g3456.index, NEW_COLS[0]] = names
                    work.loc[g3456.index, NEW_COLS[1]] = np.round(dists, 2)

                if len(g56) and len(g34):
                    names, dists = find_nearest(g56, g34)
                    work.loc[g56.index, NEW_COLS[2]] = names
                    work.loc[g56.index, NEW_COLS[3]] = np.round(dists, 2)

                if len(g6) and len(g5):
                    names, dists = find_nearest(g6, g5)
                    work.loc[g6.index, NEW_COLS[4]] = names
                    work.loc[g6.index, NEW_COLS[5]] = np.round(dists, 2)

                # Лист «Результат» — только найденные
                sheet_res = work[
                    work["match_status"].isin(["из словаря", "из справочника"])
                ].copy()

                orig_cols = [c for c in user_df.columns if c in sheet_res.columns]
                sheet_res = sheet_res[orig_cols + ["lat", "lon", "match_status"] + NEW_COLS]

                # Лист «Не_найдено»
                sheet_nf = work[
                    work["match_status"].isin(["не найдено", "пусто в исходнике"])
                ].copy()
                nf_cols = [c for c in user_df.columns if c in sheet_nf.columns]
                sheet_nf = sheet_nf[nf_cols]

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    sheet_res.to_excel(writer, sheet_name="Результат", index=False)
                    if len(sheet_nf):
                        sheet_nf.to_excel(writer, sheet_name="Не_найдено", index=False)

                st.success("Готово!")
                st.download_button(
                    "⬇ Скачать result.xlsx",
                    buf.getvalue(),
                    "result.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
