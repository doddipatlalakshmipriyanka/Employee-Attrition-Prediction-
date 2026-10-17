""", unsafe_allow_html=True)

personal_col, work_col = st.columns(2, gap="medium")

with personal_col:
    with st.container(border=True):
        st.markdown("### 👤  Personal Details")
        age = st.number_input("Age (years)", min_value=18, max_value=80, value=30, step=1)
        income = st.number_input("Monthly Income (₹)", min_value=0, max_value=1000000, value=40000, step=1000)
        years = st.number_input("Years at Company", min_value=0, max_value=60, value=3, step=1)
        predict_button = st.button("✨  Analyze Employee Attrition  →", use_container_width=True)

with work_col:
    with st.container(border=True):
        st.markdown("### 💼  Work Details")
        satisfaction = st.slider(
            "Job Satisfaction", min_value=1, max_value=4, value=3,
            key="job_satisfaction"
        )
        overtime = st.radio(
            "Works Overtime?", ["No", "Yes"],
            horizontal=True, key="overtime"
        )
        st.markdown("""
        <p style="color:#a9c8ff;font-size:.69rem;margin-top:20px">
        Satisfaction scale: 1 (Low) to 4 (High).
        </p>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Prediction result
# ---------------------------------------------------------
if predict_button:
    employee = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": income,
        "YearsAtCompany": years,
        "JobSatisfaction": satisfaction,
        "OverTime": overtime
    }])

    prediction = str(model.predict(employee)[0]).strip().title()
    probabilities = model.predict_proba(employee)[0]
    yes_index = next(
        (i for i, label in enumerate(classes)
         if str(label).strip().lower() == "yes"),
        None
    )
    if yes_index is None:
        st.error("The Attrition column must include a 'Yes' class.")
        st.stop()

    probability = float(probabilities[yes_index]) * 100
    probability = max(0.0, min(100.0, probability))

    st.markdown("""
    <div class="section-title" style="margin-top:13px">
      <span class="section-num">02</span> Prediction Result
    </div>
    <div class="section-sub">See the model's prediction and the estimated probability of attrition.</div>
    """, unsafe_allow_html=True)

    result_col, prob_col = st.columns(2, gap="medium")

    if prediction == "Yes":
        result_icon = "⚠️"
        result_color = "#ff829a"
        result_title = "Attrition: YES"
        result_copy = "The model predicts that this employee may leave the company."
    else:
        result_icon = "🛡️"
        result_color = "#00e0b7"
        result_title = "Attrition: NO"
        result_copy = "The model predicts that this employee may stay with the company."

    with result_col:
        st.markdown(f"""
        <div class="result-card">
          <div class="result-icon">{result_icon}</div>
          <div class="result-label">MODEL PREDICTION</div>
          <div class="result-title" style="color:{result_color}">{escape(result_title)}</div>
          <div class="result-copy">{escape(result_copy)}</div>
        </div>
        """, unsafe_allow_html=True)

    with prob_col:
        st.markdown(f"""
        <div class="prob-card">
          <div class="prob-layout">
            <div class="prob-circle" style="--prob:{probability * 3.6}deg">
              {probability:.1f}%
            </div>
            <div>
              <div class="prob-heading">Attrition Probability</div>
              <div class="prob-copy">Estimated probability assigned to the Yes class.</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(round(probability)), text=f"Estimated probability: {probability:.1f}%")

    st.markdown("""
    <div class="notice">
      ⓘ This is a machine-learning estimate, not a guarantee. Results depend on the quality and representativeness of the training dataset.
    </div>
    """, unsafe_allow_html=True)
    st.balloons()

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div class="footer">
  <div style="font-weight:700;color:#bcd9ff">AttritionAI　·　Employee Analytics Project</div>
  <div style="margin-top:4px">Built with Python　·　 Streamlit　·　 Scikit-learn</div>
</div>
""", unsafe_allow_html=True)
