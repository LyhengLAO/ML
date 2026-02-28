# Streamlit Cheatsheet 🇫🇷

Cheatsheet pratique pour créer rapidement des applications avec **Streamlit**.

---

## 📦 Installation & lancement
```bash
pip install streamlit
streamlit hello
streamlit run app.py
```

**Structure minimale**
```python
import streamlit as st

st.title("Mon app Streamlit")
st.write("Hello 👋")
```

---

## ✍️ Texte & mise en page
```python
st.title("Titre")
st.header("Header")
st.subheader("Subheader")
st.text("Texte brut")
st.markdown("**Markdown** _ici_")
st.caption("Légende")
st.code("print('hi')", language="python")
st.latex(r"\\sum_{i=1}^n i")
```

### Colonnes & containers
```python
col1, col2 = st.columns(2)
with col1:
    st.write("Gauche")
with col2:
    st.write("Droite")

with st.container():
    st.write("Bloc")
```

### Tabs & expander
```python
tab1, tab2 = st.tabs(["A", "B"])
with tab1:
    st.write("Contenu A")

with st.expander("Voir plus"):
    st.write("Détails")
```

### Sidebar
```python
st.sidebar.title("Menu")
st.sidebar.write("Options")
```

---

## 🎛️ Widgets (inputs)
```python
st.button("Clique")
st.checkbox("Activer", value=True)
st.radio("Choix", ["A","B"])
st.selectbox("Select", ["A","B"])
st.multiselect("Multi", ["A","B","C"])
st.slider("Slider", 0, 100, 50)
st.text_input("Texte")
st.text_area("Zone texte")
st.number_input("Nombre", 0, 10, 2)
st.date_input("Date")
st.file_uploader("Upload", type=["csv","png","jpg"])
st.color_picker("Couleur")
```

### Formulaire
```python
with st.form("mon_form"):
    nom = st.text_input("Nom")
    ok = st.form_submit_button("Envoyer")

if ok:
    st.success(f"Salut {nom}")
```

---

## 📊 Données
```python
import pandas as pd

df = pd.DataFrame({"a": [1,2], "b": [3,4]})

st.dataframe(df)
st.table(df)
st.json({"a": 1})
```

### Metrics
```python
st.metric("CA", "120k€", "+5%")
```

---

## 📈 Graphiques
### Streamlit natif
```python
st.line_chart(df)
st.bar_chart(df)
st.area_chart(df)
```

### Matplotlib
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1,2,3],[1,4,2])
st.pyplot(fig)
```

### Plotly
```python
import plotly.express as px
fig = px.scatter(df, x="a", y="b")
st.plotly_chart(fig, use_container_width=True)
```

---

## 🖼️ Médias
```python
st.image("img.png", caption="Image", use_container_width=True)
st.audio("sound.mp3")
st.video("video.mp4")
```

---

## 🚦 Messages & UX
```python
st.success("OK")
st.info("Info")
st.warning("Attention")
st.error("Erreur")

st.progress(30)
with st.spinner("Chargement..."):
    ...
```

---

## 🧭 Navigation (multi-pages)
**Via dossier `pages/`**
```
app.py
pages/
 ├─ 1_Accueil.py
 └─ 2_Analyse.py
```

**Navigation custom**
```python
page = st.sidebar.selectbox("Page", ["Accueil", "Analyse"])
if page == "Accueil":
    st.write("Home")
else:
    st.write("Analyse")
```

---

## 🧠 Session State
```python
if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("++"):
    st.session_state.count += 1

st.write(st.session_state.count)
```

---

## ⚡ Cache & performance
```python
@st.cache_data
def load_data():
    return [1,2,3]

@st.cache_resource
def load_model():
    ...
```

---

## 🔑 Secrets & config
**`.streamlit/secrets.toml`**
```toml
API_KEY = "xxx"
```
```python
key = st.secrets["API_KEY"]
```

---

## ⛔ Stop & rerun
```python
st.stop()
st.rerun()
```

---

## 📤 Téléchargement
```python
st.download_button(
    "Télécharger",
    data=df.to_csv(index=False),
    file_name="data.csv",
    mime="text/csv"
)
```

---

## 🚀 Déploiement (rapide)
- Streamlit Community Cloud (GitHub)
- Docker
- Hugging Face Spaces
- Render

---

💡 *Tip* : Streamlit relance le script **de haut en bas** à chaque interaction.

