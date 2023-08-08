import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from functions2 import get_table_download_link, get_image_download_link, preprocessing, best_performances, plots, shap_plots
import shap
shap.initjs()

st.set_page_config(page_title="Sekkat Ahmed App", layout="wide")
block_container = {"max-width": f"{80}%"}
text_align = {"text-align": "center"}

st.markdown(
    f"""
    <style>
    .block-container {{
        max-width: {block_container['max-width']};
    }}
    .text-center {{
        text-align: {text_align['text-align']};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.set_option('deprecation.showPyplotGlobalUse', False)
np.float = float

placeholder = st.empty()

with placeholder.container():
    st.title("Analyse du statut de prêt")
    st.write("")
    st.markdown("""Notre application vous permet d'analyser et de prédire le statut de remboursement d'un prêt. En utilisant la fonctionnalité de téléchargement de fichiers, vous pouvez importer les données de votre banque.""")
    st.divider() 
    st.markdown("""Une fois le fichier sélectionné, l'application affiche un tableau de données correspondant dans le premier onglet vous trouverez ainsi les métriques et performances du meilleur modèle de prédiction. Des graphiques tels que la courbe ROC AUC, la matrice de confusion et la courbe de précision-rappel vous donnent un aperçu des performances du modèle.

Dans le deuxième onglet, vous pouvez explorer les variables qui influencent les remboursements de prêts. Des graphiques interactifs tels que shap.summary_plot et shap.dependence_plot vous aident à visualiser les relations et l'impact des variables sur les prédictions.

De plus, l'application propose des shap.force_plot pour chaque estimation choisie par l'utilisateur, fournissant des explications détaillées des prédictions spécifiques.

L'application vous permet d'analyser et de prédire le statut de remboursement des prêts. Elle offre des visualisations détaillées des performances du modèle et des variables influentes, vous permettant de prendre des décisions éclairées dans le domaine des prêts.""")

if 'data' not in st.session_state:
    st.session_state.data = None

if 'prep_data' not in st.session_state:
    st.session_state.prep_data = None

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = None

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Your Data")
    placeholder1 = st.empty()
    with placeholder1.container():
        if st.session_state.button_clicked is None:
            st.divider()
            st.markdown("*Vous pouvez télécharger le fichier CSV pour l'uploader:*")
            data_to_upload = pd.read_csv("data_test.csv")
            csv_content = data_to_upload.to_csv(index=False)
            st.session_state.button_clicked = st.download_button(
                    label="Click here to download CSV File",
                    data=csv_content,
                    file_name="data_to_upload.csv",
                    mime="text/csv",
                    )
    if not uploaded_file:
        st.stop()
    if st.session_state.data is None:
        with st.spinner("Importation in Progress"):
            st.session_state.data = pd.read_csv(uploaded_file)
    if st.session_state.prep_data is None:
        st.session_state.prep_data = st.session_state.data.copy()
        with st.spinner("Loading..."):
            st.session_state.prep_data = preprocessing(st.session_state.prep_data)
            st.session_state.prep_data.reset_index(drop=True, inplace=True)
    placeholder.empty()

tab1, tab2, tab3 = st.tabs(["Model Performances", "Explainability", "Search By ID"])

with tab1:
        st.header("Display Data")
        st.dataframe(st.session_state.data.iloc[:, 1:])
        st.divider()
        st.session_state.prep_data = st.session_state.prep_data[['int_rate', 'total_rec_int', 'last_week_pay', 'total_rev_hi_lim', 'tot_cur_bal', 'collection_recovery_fee', 'recoveries', 'batch_enrolled', 'term', 'sub_grade', 'loan_status']]
        X = st.session_state.prep_data.drop("loan_status", axis=1).reset_index(drop=True)
        y = st.session_state.prep_data.loan_status.reset_index(drop=True)
        st.header("Performances du Modèle")
        with st.spinner("Performances in Progress"):
            df_report, conf_matrix, target_names, final_pipeline, len_s, score = best_performances(X, y)
            st.dataframe(df_report)
            st.divider()
            #X_transformed = final_pipeline.named_steps['prep_pipeline'].transform(X)
            X_transformed = final_pipeline.named_steps['preprocessor'].transform(X)
            X = pd.DataFrame(X_transformed, columns = X.columns).reset_index(drop=True)
            fig1, fig2, fig3, fig4, optimal_porba_cutoffs = plots(conf_matrix, final_pipeline, target_names, X, y, len_s, score)
        
            buf1=BytesIO()
            fig1.savefig(buf1, format='png')

            buf2=BytesIO()
            fig2.savefig(buf2, format='png')

            buf3=BytesIO()
            fig3.savefig(buf3, format='png')

            buf4=BytesIO()
            fig4.savefig(buf4, format='png')

            buf5=BytesIO()

            buf6=BytesIO()

            buf7=BytesIO()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Matrice de Confusion")
                st.image(buf1)
                st.markdown(get_image_download_link(buf1, "Confusion_Matrix.png", 'Click here to Download Confusion_Matrix'), unsafe_allow_html=True)

            with col2:
                st.subheader("Class Prediction Error")
                st.image(buf2)
                st.markdown(get_image_download_link(buf2, "Class_Prediction.png", 'Click here to Download Class_Prediction_Error'), unsafe_allow_html=True)

            st.divider()

            col3, col4 = st.columns(2)

            with col3:
                st.subheader("ROC AUC Curve")
                st.image(buf3)
                st.markdown(get_image_download_link(buf3, "ROC_AUC.png", 'Click here to Download ROC_AUC Plot'), unsafe_allow_html=True)
        
            with col4:
                st.subheader("Precision Recall Curve")
                st.image(buf4)
                st.markdown(get_image_download_link(buf4, "Precision_Recall.png", 'Click here to Download Precision_Recall Curve'), unsafe_allow_html=True)

with tab2:
    st.header("Summary Plot")
    with st.spinner("Plots in Progress"):
        estimator, explainer, shap_values, X_sample, y_sample, data_sample, fig = shap_plots(X, y, st.session_state.prep_data)
        fig5, ax5 = plt.subplots(1, 1, figsize=(25, 17))
        shap.summary_plot(shap_values, X_sample, show=False)
        ax5 = plt.gca()
        fig5.savefig(buf5, format='png')
        st.image(buf5)
        st.markdown(get_image_download_link(buf5, "Summary_Plot.png", 'Click here to Download Summary_Plot'), unsafe_allow_html=True)
        
        st.header("Dependence Plots")
        fig.savefig(buf6, format='png')
        st.image(buf6)
        st.markdown(get_image_download_link(buf6, "Dependence_Plots.png", 'Click here to Download Dependence_Plots'), unsafe_allow_html=True)   

with tab3:
    st.header("Enter ID")
    data_Xy = pd.concat((X_sample, y_sample), axis=1)
    probabilities = estimator.predict_proba(X_sample)
    df_proba = pd.DataFrame(probabilities, columns = [0, 1])
    selected_index = st.selectbox("Sélectionnez l'indice correspondant:", range(len(X_sample)))
    if selected_index is not None:
        selected_probability = probabilities[selected_index]
        df_submit = pd.DataFrame(data_sample.iloc[selected_index]).transpose()
        st.dataframe(df_submit)
        st.subheader("Résultat de la recherche")
        if data_sample.loc[selected_index, 'loan_status'] == 0:
            st.success('Le client est non défaillant réellement!', icon="✅")
            shap.force_plot(explainer.expected_value[1], shap_values[1][selected_index,:], X_sample.iloc[selected_index], show=False, matplotlib=True, figsize=(30, 8)).savefig(buf7, format='png')
            st.image(buf7)
            st.markdown(get_image_download_link(buf7, "Force_Plot.png", 'Click here to Download Force Plot'), unsafe_allow_html=True)
            if selected_probability[0] < 0.5:
                st.error('Le modèle n\'a pas bien prédit le statut de remboursement! Probabilité Prédite : {}%'.format(round(selected_probability[0]*100, 2)), icon="👎🏻")
            if selected_probability[0] > 0.5:
                st.success('Le modèle a bien prédit le statut de remboursement! Probabilité Prédite : {}%'.format(round(selected_probability[0]*100, 2)), icon="👍🏻")
        if data_sample.loc[selected_index, 'loan_status'] == 1:
            st.error('Le client est défaillant réellement!', icon="🚨")
            shap.force_plot(explainer.expected_value[1], shap_values[1][selected_index,:], X_sample.iloc[selected_index], show=False, matplotlib=True, figsize=(30, 8)).savefig(buf7, format='png')
            st.image(buf7)
            st.markdown(get_image_download_link(buf7, "Force_Plot.png", 'Click here to Download Force Plot'), unsafe_allow_html=True)           
            if selected_probability[1] < 0.5:
                st.error('Le modèle n\'a pas bien prédit le statut de remboursement! Probabilité Prédite : {}%'.format(round(selected_probability[1]*100, 2)), icon="👎🏻")
            if selected_probability[1] > 0.5:
                st.success('Le modèle a bien prédit le statut de remboursement! Probabilité Prédite : {}%'.format(round(selected_probability[1]*100, 2)) , icon="👍🏻")     
