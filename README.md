
<div align="center">

💼 Employee Attrition Prediction
An AI-powered Streamlit application to explore employee attrition risk




Predict • Explore • Learn

A beginner-friendly machine learning project built with Python and Streamlit.
</div>

🌟 About the Project
Employee Attrition Prediction is a machine learning application that estimates whether an employee may leave an organization based on selected work-related details. Enter employee information, run the model, and view its predicted result in a simple interactive interface.
Note: This is an educational project. Its output is an estimate—not a fact about an employee—and should not be used alone to make hiring, firing, promotion, or other employment decisions.

✨ Features
- 🎯 Attrition prediction — estimates Yes or No using the trained model.
- 🧑‍💼 Employee inputs — age, monthly income, years at the company, job satisfaction, and overtime status.
- 📊 Interactive interface — use the Streamlit app in your browser.
- 🧹 Data preprocessing — prepare numeric and categorical values for the model.
- 🌙 Custom UI — dark navy and purple styling.
- ☁️ Deployment-ready — can be hosted with Streamlit Community Cloud.
- 🧪 Learning-focused — practice Python, pandas, and scikit-learn workflows.
🧾 Dataset
The app is designed for a CSV file named attrition.csv with these six columns:
Column	Description	Example
Age	Employee age	30
MonthlyIncome	Monthly income as a number	40000
YearsAtCompany	Years employed at the company	3
JobSatisfaction	Job satisfaction rating (1–4)	3
OverTime	Whether the employee works overtime	No
Attrition	Target label: whether the employee left	No


Important: Keep Attrition as the target column, not as an input feature. Store MonthlyIncome as a numeric value (for example, 18210, not 18,210.00) to avoid conversion errors.

If you use generated or expanded rows, identify them as synthetic data. More rows alone do not guarantee better prediction accuracy.
🧰 Built With
Tool	Purpose
Python	Main programming language
Streamlit	Interactive web interface
pandas	Data loading and preparation
scikit-learn	Preprocessing, model training, and prediction
NumPy	Numerical operations, when used by the app


📁 Project Structure
employee-attrition-prediction/
├── app.py                 # Streamlit application
├── attrition.csv          # Employee dataset
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── image/
│   └── banner.png         # Optional app banner
└── .streamlit/
    └── config.toml        # Optional theme configuration
Your repository may not include the optional image/ or .streamlit/ folders.
🚀 Run the Project Locally
1. Clone the repository
Replace the URL below with your actual GitHub repository URL.
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
2. Install dependencies
pip install -r requirements.txt
If you do not have a requirements.txt yet, a typical starting point is:
streamlit
pandas
scikit-learn
numpy
Save these package names in requirements.txt, adjusting the list to match the imports used by your app.
3. Start Streamlit
streamlit run app.py
Streamlit will provide a local URL to open in your browser.
☁️ Deploy on Streamlit Community Cloud
1. Push app.py, attrition.csv, requirements.txt, and any required images to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app and select your repository and branch.
4. Set the main file path to app.py (or the correct path if it is inside a folder).
5. Deploy the app and check the logs if deployment fails.
Deployment checklist
- [ ] app.py is in the selected path.
- [ ] requirements.txt lists every package the app imports.
- [ ] attrition.csv is committed and its filename matches the code.
- [ ] Income values are numeric and CSV columns match the expected names.
- [ ] Any model files and images referenced by the code are included in the repository.
🧠 How It Works
1. Input: The user enters employee details.
2. Preprocessing: Numeric and categorical columns are prepared for the model.
3. Prediction: The trained classifier estimates the Attrition label.
4. Result: The app displays the predicted class.
Performance depends on the dataset, label quality, model setup, and evaluation method. Measure performance on a held-out test set before describing the model as accurate.
🔧 Common Issues
<details>
<summary><strong>MonthlyIncome conversion error</strong></summary>

If you see an error such as could not convert string to float: '18,210.00', remove commas and currency symbols before converting the column to numeric values. Make sure the cleaned dataset is also used when training the model.
</details>

<details>
<summary><strong>Module not found</strong></summary>


Add the missing package to requirements.txt, commit the file, and redeploy the app.
</details>

<details>
<summary><strong>CSV file not found</strong></summary>


Check the filename, capitalization, and repository path. The name in pd.read_csv(...) must match the actual file.
</details>

<details>
<summary><strong>App works locally but not after deployment</strong></summary>


Review the Streamlit Cloud logs. Confirm that required files are committed to GitHub and file paths are relative to the project directory where possible.
</details>

🔐 Responsible Use
Employee attrition predictions can be inaccurate and may reflect bias in the data. Use this app for learning and demonstration only. Do not use predictions as the sole basis for decisions that affect employees. Protect personal and confidential employee information.
👩‍💻 Author
Doddipatla Lakshmi Priyanka
B.Tech ECE Student | Machine Learning Project
- GitHub: https://github.com/doddipatlalakshmipriyanka
<div align="center">

Made with 💜 using Python and Streamlit
</div>
