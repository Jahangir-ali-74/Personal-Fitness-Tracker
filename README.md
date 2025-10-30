# 🏃 Personal Fitness Tracker Using Machine Learning

This project is a **Streamlit-based web application** that predicts the number of **calories burned** during exercise based on user-specific inputs such as age, BMI, duration, heart rate, and body temperature. It uses a trained **Random Forest Regressor** model to make predictions based on real-world fitness data.

---

## 📌 Features

- 🔢 Predicts calories burned using user inputs
- 📊 Compares your data with others in the dataset
- 🔎 Displays personalized health stats (age, heart rate, etc.)
- 🖼️ Visually styled UI with background image and live feedback
- ✅ No wearables or devices needed — just enter your info!

---

## 🧠 Tech Stack

- **Frontend/UI:** Streamlit
- **Backend/ML:** Python, scikit-learn (Random Forest)
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly

---

## 📁 Project Structure

```
├── app.py                 # Main Streamlit app
├── calories.csv           # Calorie data
├── exercise.csv           # Exercise data
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🚀 How to Run the Project Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Jahangir-ali-74/Implementation-of-Personal-Fitness-Tracker-using-Python
   cd personal-fitness-tracker
   ```

2. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

4. **Open in Browser**
   Streamlit will open automatically at: [http://localhost:8501](http://localhost:8501)

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`, including:
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly

---

## 🖼️ Background Image Credit

Background image used from:  
🔗 [https://meridianfitness.in](https://meridianfitness.in/wp-content/uploads/2019/07/collage-strength.jpg)

---

## 📚 References

- [scikit-learn.org](https://scikit-learn.org/)
- [streamlit.io](https://streamlit.io/)
- [pandas.pydata.org](https://pandas.pydata.org/)
- [plotly.com](https://plotly.com/)
- [Original Datasets](https://www.kaggle.com/) (if applicable)

---

## 👤 Author

Created by [**Sayed Jahangir Ali**](https://github.com/Jahangir-ali-74)  
© 2025

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).