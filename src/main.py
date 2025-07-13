import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
from src.preprocessing import DataPreprocessor
from src.ui import introduction, dataset_upload, split_data, scale_data, model_selector, evaluate_model, footer
from src.functions import get_model_url, load_css, plot_metrics, generate_snippet, save_model_history, get_model_tips

from src.logger.custom_logger import logging
from src.exception.exception import AutoMLException

def main():
    try:
        st.set_page_config(page_title = "AutoML Studio", layout= "wide")
        load_css("css/styles.css")
        introduction()
        result = dataset_upload()
        if result:
            view = st.sidebar.checkbox("View the Dataset")
            if view:
                st.write(result[2])
            X_train, X_test, y_train, y_test, test_size, random_state = split_data(result)
            X_train, X_test = scale_data(result, X_train, X_test)
            model_type, model, duration, problem_type = model_selector(result[0], X_train, y_train)
            if model:
                metrics = evaluate_model(model, X_train, y_train, X_test, y_test, duration, problem_type)
                model, train_accuracy, train_f1, test_accuracy, test_f1 = metrics
                snippet = generate_snippet(model, model_type, result[5], test_size, random_state, result[1], problem_type)
                footer()
                col1, col2 = st.columns([2, 1])
                with col1:
                    duration_placeholder = st.empty()
                    model_url_placeholder = st.empty()
                    code_header_placeholder = st.empty()
                    snippet_placeholder = st.empty()
                    tips_header_placeholder = st.empty()
                    tips_placeholder = st.empty()
                with col2:
                    relative_metrics = st.empty()
                    add_placeholder = st.empty()
                    show_placeholder = st.empty()
                    plot_placeholder = st.empty()
                    models_placeholder = st.empty()
                metrics_dict = {
                    "Classification": {"train_accuracy": train_accuracy, "train_f1": train_f1, "test_accuracy": test_accuracy, "test_f1": test_f1},
                    "Regression": {"train_mse": train_accuracy, "train_rmse": train_f1, "test_mse": test_accuracy, "test_rmse": test_f1}
                }.get(problem_type, {})
                fig = plot_metrics(metrics_dict, problem_type)
                relative_metrics.warning("Increase or Decrease is with respect to Training Dataset")
                plot_placeholder.plotly_chart(fig, use_container_width=True)
                if add_placeholder.button("Click to record these Hyperparameters"):
                    save_model_history(model, model_type, metrics_dict, result[5])
                    st.success("Model history saved!")
                if show_placeholder.button("Click to view all models"):
                    try:
                        with open("data/model_history.txt", "r") as f:
                            models_placeholder.code(f.read())
                    except:
                        models_placeholder.info("No model history available.")
                duration_placeholder.warning(f"Training took {duration:.3f} seconds")
                model_url_placeholder.markdown(get_model_url(model_type), unsafe_allow_html=True)
                code_header_placeholder.header("**Retrain the same model in Python**")
                snippet_placeholder.code(snippet)
                tips_header_placeholder.header(f"**Tips on the {model_type} 💡**")
                tips_placeholder.info(get_model_tips(model_type))
    except AutoMLException as e:
        logging.error(f"Main app error: {str(e)}")
        st.error(str(e))

if __name__ == "__main__":
    main()