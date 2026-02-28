import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Viz Dataset", layout="wide")
sns.set_theme(style="whitegrid")

st.title("Visualisation de dataset (CSV)")

# --- Helpers ---
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def require_df():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("Charge d'abord un dataset (page 1).")
        st.stop()
    return st.session_state["df"]

def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

# --- Sidebar navigation ---
st.sidebar.title("Sommaire")
pages = ["Charger les données", "Analyse du contenu", "Visualisation"]
page = st.sidebar.radio("Aller :", pages)

if page == pages[0]:
    st.subheader("1) Charger les données")

    file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    colA, colB = st.columns([1, 1])
    with colA:
        sep = st.text_input("Séparateur (optionnel)", value=",")
    with colB:
        encoding = st.text_input("Encodage (optionnel)", value="utf-8")

    if file is not None:
        if st.button("Charger ce dataset", type="primary"):
            try:
                # lecture simple + options
                if sep.strip() == "":
                    df = pd.read_csv(file, encoding=encoding)
                else:
                    df = pd.read_csv(file, sep=sep, encoding=encoding)
                st.session_state["df"] = df
                st.success(f"Dataset chargé ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
            except Exception as e:
                st.error(f"Erreur de lecture CSV : {e}")
    
    if st.button("Afficher un échantillon"):
        if "df" in st.session_state and st.session_state["df"] is not None:
            st.dataframe(st.session_state["df"].head(10), use_container_width=True)
        else:
            st.warning("Charge d'abord un dataset.")

    if st.button("Réinitialiser"):
        st.session_state["df"] = None
        st.info("Dataset supprimé de la session.")

elif page == pages[1]:
    st.subheader("2) Analyse du contenu")
    df = require_df()

    st.write("Aperçu :")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lignes", df.shape[0])
    with col2:
        st.metric("Colonnes", df.shape[1])
    with col3:
        st.metric("Valeurs manquantes", int(df.isna().sum().sum()))

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Types de colonnes")
        st.dataframe(df.dtypes.astype(str).to_frame("Type"), use_container_width=True)

    with c2:
        st.subheader("Valeurs manquantes par colonne")
        miss = df.isna().sum().sort_values(ascending=False)
        st.dataframe(miss.to_frame("NA count"), use_container_width=True)

    st.subheader("Description statistique (numérique)")
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] == 0:
        st.info("Aucune colonne numérique détectée.")
    else:
        st.dataframe(num_df.describe().T, use_container_width=True)

else:
    st.subheader("3) Visualisation")
    df = require_df()
    df_num = df.select_dtypes(include="number")
    df_cat = df.select_dtypes(exclude="number")

    viz = st.sidebar.radio("Type de données à visualiser", ["Numérique", "Catégorielle"])

    if viz == "Catégorielle":
        if df_cat.shape[1] == 0:
            st.info("Aucune colonne catégorielle détectée.")
        else:
            st.subheader("Colonnes catégorielles")
            st.dataframe(df_cat, use_container_width=True)

        st.markdown("---")

        graph_type = st.selectbox(
            "Type de graphe",
            ["Histogramme", "Count plot", "Box plot"]
        )

        all_cat_cols = list(df_cat.columns)
        x = st.selectbox("X (abscisse)", all_cat_cols)
        y = None
        if graph_type in ["Nuage de points", "Box plot"]:
            y_candidates = ["Aucun"] + all_cat_cols
            y = st.selectbox("Y (ordonnée)", y_candidates)
            if y == "Aucun":
                y = None
        
        hue_candidates = ["Aucun"] + all_cat_cols
        hue = st.selectbox("Hue (séparer par)", hue_candidates)
        hue = None if hue == "Aucun" else hue

        col_opt_1, col_opt_2, col_opt_3 = st.columns(3)
        with col_opt_1:
            kde = st.checkbox("KDE (Histogramme)", value=True, disabled=(graph_type != "Histogramme"))
        with col_opt_2:
            bins = st.slider("Bins (Histogramme)", 5, 100, 30, disabled=(graph_type != "Histogramme"))
        with col_opt_3:
            rotate = st.slider("Rotation labels X", 45, 90, 45)

        if st.button("Tracer", type="primary"):
            fig, ax = plt.subplots(figsize=(10, 5))

            try:
                if graph_type == "Histogramme":
                    sns.histplot(data=df, x=x, hue=hue, kde=kde, bins=bins, ax=ax)

                elif graph_type == "Count plot":
                    sns.countplot(data=df, x=x, hue=hue, ax=ax)

                elif graph_type == "Box plot":
                    if y is None:
                        st.error("Box plot : tu dois choisir une colonne Y.")
                        st.stop()
                    if not is_numeric(df[y]):
                        st.warning("Box plot : Y devrait être numérique.")
                    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)

                ax.set_title(f"{graph_type} — X: {x}" + (f" | Y: {y}" if y else "") + (f" | Hue: {hue}" if hue else ""))
                if rotate:
                    ax.tick_params(axis="x", rotation=rotate)

                st.pyplot(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur lors du tracé : {e}")

    else:
        if df_num.shape[1] == 0:
            st.info("Aucune colonne numérique détectée.")
        else:
            st.subheader("Colonnes numériques")
            st.dataframe(df_num, use_container_width=True)

        st.markdown('---')

        graph_type = st.selectbox(
            "Type de graphe",
            ["Histogramme", "Nuage de points", "Box plot"]
        )

        all_num_cols = list(df_num.columns)
        x = st.selectbox("X (abscisse)", all_num_cols)

        y = None
        if graph_type in ["Nuage de points", "Box plot"]:
            y_candidates = ["Aucun"] + all_num_cols
            y = st.selectbox("Y (ordonnée)", y_candidates)
            if y == "Aucun":
                y = None

        hue_candidates = ["Aucun"] + all_num_cols
        hue = st.selectbox("Hue (séparer par)", hue_candidates)
        hue = None if hue == "Aucun" else hue

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            kde = st.checkbox("KDE (Histogramme)", value=True, disabled=(graph_type != "Histogramme"))
        with col_opt2:
            bins = st.slider("Bins (Histogramme)", 5, 100, 30, disabled=(graph_type != "Histogramme"))
        with col_opt3:
            rotate = st.checkbox("Rotation labels X", value=True)

        if st.button("Tracer", type="primary"):
            fig, ax = plt.subplots(figsize=(10, 5))

            try:
                if graph_type == "Histogramme":
                    if not is_numeric(df[x]):
                        st.warning("Histogramme : X devrait être numérique (ou convertible).")
                    sns.histplot(data=df, x=x, hue=hue, kde=kde, bins=bins, ax=ax)

                elif graph_type == "Nuage de points":
                    if y is None:
                        st.error("Nuage de points : tu dois choisir une colonne Y.")
                        st.stop()
                    if not is_numeric(df[x]) or not is_numeric(df[y]):
                        st.warning("Nuage de points : X et Y devraient être numériques.")
                    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)

                elif graph_type == "Count plot":
                    sns.countplot(data=df, x=x, hue=hue, ax=ax)

                elif graph_type == "Box plot":
                    if y is None:
                        st.error("Box plot : tu dois choisir une colonne Y.")
                        st.stop()
                    if not is_numeric(df[y]):
                        st.warning("Box plot : Y devrait être numérique.")
                    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)

                ax.set_title(f"{graph_type} — X: {x}" + (f" | Y: {y}" if y else "") + (f" | Hue: {hue}" if hue else ""))
                if rotate:
                    ax.tick_params(axis="x", rotation=45)

                st.pyplot(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur lors du tracé : {e}")