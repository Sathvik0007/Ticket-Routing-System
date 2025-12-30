import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Load model and vectorizer
svm = joblib.load("svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
kmeans = joblib.load("kmeans_model.pkl")
cluster_label_dist = pd.read_csv("cluster_label_distribution.csv")
cluster_keywords = joblib.load("cluster_keywords.pkl")


THRESHOLD = 0.5  # confidence threshold

st.title("Customer Support Ticket Router")
st.write("Paste a customer support ticket to predict the routing queue.")

ticket_text = st.text_area("Ticket Text", height=200)

if st.button("Route Ticket"):
    if ticket_text.strip() == "":
        st.warning("Please enter some ticket text.")
    else:
        vec = tfidf.transform([ticket_text])
        scores = svm.decision_function(vec)

        pred_label = svm.classes_[np.argmax(scores)]
        confidence = np.max(scores)

        st.subheader("Prediction Result")
        st.write(f"**Primary Prediction:** {pred_label}")
        st.write(f"**Decision Score (SVM margin):** {confidence:.3f}")


        if confidence < THRESHOLD:
            st.warning("Low confidence prediction. Ticket will not be auto-routed.")

            # find cluster
            cluster_id = kmeans.predict(vec)[0]
        
            # label distribution for this cluster
            dist = (
                cluster_label_dist[cluster_label_dist["cluster"] == cluster_id]
                .sort_values("percentage", ascending=False)
                .head(2)
            )
        
            st.markdown("### Suggested Queues (based on similar past tickets)")
        
            for _, row in dist.iterrows():
                label = row["queue"]
                pct = row["percentage"]
                st.write(f"**{label}** — {int(pct * 100)}%")
                st.progress(float(pct))
        
            st.markdown("**Common keywords in similar tickets:**")
            st.write(", ".join(cluster_keywords[cluster_id]))
        
            st.caption("Final routing should be confirmed by a support agent.")
        
        else:
            st.success("Ticket can be auto-routed confidently.")