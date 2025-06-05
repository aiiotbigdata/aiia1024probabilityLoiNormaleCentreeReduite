import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import json
import io

st.set_page_config(page_title="Analyse de la distribution normale", layout="centered")

st.title("Analyse de la distribution normale d'une variable")

st.markdown(
    "<h4><b>Importez votre fichier au format (.csv)</b></h4>",
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    colonnes_numeriques = df.select_dtypes(include=np.number).columns.tolist()

    if colonnes_numeriques:
        colonne = st.selectbox("Choisissez la variable numérique :", colonnes_numeriques)
        valeurs = df[colonne].dropna()

        # Paramètres d'analyse
        min_val, max_val = float(valeurs.min()), float(valeurs.max())
        moyenne, ecart_type = float(valeurs.mean()), float(valeurs.std())

        st.write(f"Moyenne : **{moyenne:.2f}**, Écart-type : **{ecart_type:.2f}**")
        st.write(f"Min : {min_val:.2f}, Max : {max_val:.2f}")

        with st.expander("Ajouter des intervalles (a, b)", expanded=True):
            bornes = []
            for i in range(1, 4):
                col1, col2 = st.columns(2)
                a = col1.number_input(f"Début de l'intervalle {i}", value=moyenne - i * ecart_type)
                b = col2.number_input(f"Fin de l'intervalle {i}", value=moyenne + i * ecart_type)
                if a < b:
                    bornes.append((a, b))

        show_hist = st.checkbox("Afficher l'histogramme expérimental", value=True)
        annotate = st.checkbox("Annoter les probabilités sur le graphe", value=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.linspace(moyenne - 4*ecart_type, moyenne + 4*ecart_type, 1000)
        y = norm.pdf(x, moyenne, ecart_type)

        couleurs = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#d28aff']
        probas = {}

        if show_hist:
            sns.histplot(valeurs, bins=30, stat='density', color='lightgray', edgecolor='black', alpha=0.5, label="Histogramme", ax=ax)

        sns.lineplot(x=x, y=y, color='black', label=f"Densité normale", ax=ax)

        for i, (a, b) in enumerate(bornes):
            x_fill = np.linspace(a, b, 300)
            y_fill = norm.pdf(x_fill, moyenne, ecart_type)
            ax.fill_between(x_fill, y_fill, color=couleurs[i % len(couleurs)], alpha=0.5, label=f"P({a:.1f} < X < {b:.1f})")
            ax.axvline(a, color=couleurs[i % len(couleurs)], linestyle='--')
            ax.axvline(b, color=couleurs[i % len(couleurs)], linestyle='--')
            p = norm.cdf(b, moyenne, ecart_type) - norm.cdf(a, moyenne, ecart_type)
            probas[f"{a:.1f} < X < {b:.1f}"] = round(p, 4)
            if annotate:
                ax.text((a + b) / 2, norm.pdf((a + b) / 2, moyenne, ecart_type) + 0.002,
                        f"{p * 100:.2f}%", ha='center', fontsize=9, color=couleurs[i % len(couleurs)])

        ax.set_title(f"Distribution de {colonne}  |  Moyenne = {moyenne:.1f}, Écart-type = {ecart_type:.1f}")
        ax.set_xlabel(colonne)
        ax.set_ylabel("Densité")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # Affichage tableau des probabilités
        st.subheader("Probabilités calculées :")
        st.table(pd.DataFrame.from_dict(probas, orient='index', columns=["Probabilité"]))

        # Export image
        with st.expander("Exporter le graphique ou les données"):
            export_format = st.selectbox("Format du graphique :", ["png", "pdf", "svg"])
            export_btn = st.button("Télécharger le graphique")

            if export_btn:
                buf = io.BytesIO()
                fig.savefig(buf, format=export_format, dpi=300, bbox_inches='tight')
                st.download_button("Télécharger le fichier", buf.getvalue(), file_name=f"graphique.{export_format}")

            format_proba = st.radio("Exporter les probabilités :", ["CSV", "JSON"])
            if format_proba == "CSV":
                csv = pd.DataFrame.from_dict(probas, orient='index', columns=["Probabilité"]).to_csv().encode('utf-8')
                st.download_button("Télécharger les probabilités (CSV)", csv, file_name="probabilites.csv")
            else:
                json_data = json.dumps(probas, indent=4)
                st.download_button("Télécharger les probabilités (JSON)", json_data, file_name="probabilites.json")

    else:
        st.warning("Le fichier ne contient pas de colonnes numériques.")
else:
    st.info("Veuillez importer un fichier CSV contenant des données numériques.")
