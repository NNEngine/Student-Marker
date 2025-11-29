import streamlit as st # UI
import numpy as np
import joblib # for model loading

st.set_page_config(
  page_title = "Marker",
  page_icon = "😎"
)

st.title("😎Students Marks Prediction")
st.write("It is a simple students marks prediction Machine learning Model which is trained in LinearRegression")
st.markdown("---")


# Sidebar
st.sidebar.title("Student Performance Tracker")
st.sidebar.image("robot-human-hands-interacting.jpg")
st.sidebar.write("It is a machine learning based app used for tracking students marks on the basis of the following factor: ")
st.sidebar.markdown("- Gender")
st.sidebar.markdown("- Hours studied")
st.sidebar.markdown("- Attendance percent")
st.sidebar.markdown("- Assignments completed")
st.sidebar.markdown("---")
st.sidebar.markdown("`Made by Shivam Sharam`")

# load model
model = joblib.load("model.pkl")

input_names = ["gender", "hours_studied", "attendance_percent", "assignments_completed"]

gender_value = st.selectbox("Gender", [0, 1])
hours_studied_values = st.number_input("Hours Studied", min_value = 5, max_value = 17)
attendance_percent_values = st.number_input("Attendance percenter", min_value = 60, max_value = 99)
assignments_completed_values = st.number_input("Assignments Completed", min_value = 5, max_value = 14)

inputs = [gender_value, hours_studied_values, attendance_percent_values, assignments_completed_values]

inputs = np.array(inputs).reshape(1, -1)

if st.button("predict"):
  output = model.predict(inputs)
  st.success(f"test score {output}")

  if output < 33:
    st.warning("Student Failed!")
  elif output > 33 and output < 50:
    st.warning("You were on the brink of fail")
  else:
    st.success("You passed with good marks 🎇🎆🎉✨")   

