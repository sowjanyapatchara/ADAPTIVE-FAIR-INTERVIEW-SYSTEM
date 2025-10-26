import streamlit as st
from groq import Groq
import os
import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------#
# 1️⃣ Setup and Configuration
# -----------------------------#
st.set_page_config(page_title="Fair AI Interview System", layout="wide")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("⚠ Please set your GROQ_API_KEY before running.")
else:
    client = Groq(api_key=groq_api_key)

st.title("🤖 Fair & Adaptive AI Interview System")
st.write("An ethical, skill-aware, and fair AI-powered interview assistant.")

# -----------------------------#
# 2️⃣ Candidate Information
# -----------------------------#
st.sidebar.header("📋 Candidate Information")
candidate_name = st.sidebar.text_input("Candidate Name:")
candidate_gender = st.sidebar.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
candidate_experience = st.sidebar.selectbox("Experience Level", ["Fresher", "Junior", "Mid-level", "Senior"])
job_role = st.sidebar.text_input("Job Role (e.g., Software Engineer, Data Analyst):")
resume_text = st.sidebar.text_area("Paste Candidate Resume or Skills Summary:")
num_questions = st.sidebar.slider("Number of Questions", 3, 10, 5)

# Initialize state
if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []
if "score" not in st.session_state:
    st.session_state.score = None

# -----------------------------#
# 3️⃣ Tabs for System Sections
# -----------------------------#
tab1, tab2, tab3 = st.tabs(["🧠 Interview", "📊 Fairness Dashboard", "🕸 Skills Graph"])

# -----------------------------#
# 🧠 TAB 1: Interview System
# -----------------------------#
with tab1:
    st.subheader("AI Interview System")

    if st.button("Generate Interview Questions"):
        if not client:
            st.error("Groq API key missing.")
        elif resume_text and job_role:
            with st.spinner("Generating adaptive questions..."):
                prompt = f"""
                You are an expert interviewer for {job_role}.
                Candidate's resume: {resume_text}.
                Generate {num_questions} clear, skill-focused, adaptive questions 
                based on their skills and job role.
                Only list the questions numbered 1 to {num_questions}.
                """
                try:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": "You are a professional interviewer."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=700
                    )
                    text = response.choices[0].message.content.strip()
                    st.session_state.questions = re.findall(r'\d+\.\s+(.*)', text)
                    st.session_state.answers = [""] * len(st.session_state.questions)
                    st.success("✅ Questions generated successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please fill out job role and resume.")

    if st.session_state.questions:
        st.write("### 📝 Candidate Responses")
        for i, q in enumerate(st.session_state.questions):
            st.session_state.answers[i] = st.text_area(f"{i+1}. {q}", value=st.session_state.answers[i])

        if st.button("Analyze Answers"):
            with st.spinner("Analyzing candidate answers for scoring and feedback..."):
                answer_prompt = "Analyze the following answers and score them (0–100). Give overall feedback:\n"
                for i, q in enumerate(st.session_state.questions):
                    answer_prompt += f"Q{i+1}: {q}\nAnswer: {st.session_state.answers[i]}\n\n"

                try:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": "You are an expert HR and technical interviewer."},
                            {"role": "user", "content": answer_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    feedback = response.choices[0].message.content.strip()
                    st.session_state.score = int(re.search(r'(\d+)', feedback.split("\n")[0]).group(1)) if re.search(r'(\d+)', feedback.split("\n")[0]) else 75
                    st.subheader("📊 Feedback & Score")
                    st.write(feedback)
                except Exception as e:
                    st.error(f"Error: {e}")

# -----------------------------#
# 📊 TAB 2: Fairness Dashboard
# -----------------------------#
with tab2:
    st.subheader("Fairness Dashboard (Bias Detection)")

    if st.session_state.score is not None:
        # Save candidate data
        new_data = pd.DataFrame([{
            "Name": candidate_name,
            "Gender": candidate_gender,
            "Experience": candidate_experience,
            "Job Role": job_role,
            "Score": st.session_state.score,
            "Selected": 1 if st.session_state.score >= 70 else 0
        }])

        if os.path.exists("results.csv"):
            old_data = pd.read_csv("results.csv")
            df = pd.concat([old_data, new_data], ignore_index=True)
        else:
            df = new_data

        df.to_csv("results.csv", index=False)
        st.success("✅ Candidate data saved for fairness analysis!")

        st.write("### 📈 Overall Candidate Data")
        st.dataframe(df)

        # Compute fairness metrics
        st.write("### ⚖ Group Fairness Analysis")
        group_sr = df.groupby("Gender")["Selected"].mean()
        st.bar_chart(group_sr)

        max_sr = group_sr.max()
        air = (group_sr / max_sr).round(2)
        st.write("Adverse Impact Ratio (AIR):")
        st.write(air)

        for group, ratio in air.items():
            if ratio < 0.8:
                st.warning(f"⚠ Potential bias detected against {group} (AIR = {ratio})")
            else:
                st.success(f"✅ Fair for {group} (AIR = {ratio})")

# -----------------------------#
# 🕸 TAB 3: Skills Graph
# -----------------------------#
with tab3:
    st.subheader("Dynamic Skills Graph")

    if resume_text:
        # Extract simple skill keywords (mockup example)
        skills = re.findall(r'\b(Python|Java|SQL|HTML|CSS|React|C\+\+|Machine Learning|Data Analysis|AI|Spring Boot)\b', resume_text, re.I)
        if skills:
            G = nx.Graph()
            for skill in skills:
                G.add_node(skill)
            for i in range(len(skills)):
                for j in range(i+1, len(skills)):
                    G.add_edge(skills[i], skills[j])

            fig, ax = plt.subplots()
            nx.draw(G, with_labels=True, node_color="skyblue", node_size=2000, font_size=10, ax=ax)
            st.pyplot(fig)
            st.info("🧠 This graph shows related skills detected from the resume, forming the basis for adaptive question generation.")
        else:
            st.warning("No recognizable skills found in the resume.")
    else:
        st.info("Paste resume text in sidebar to generate a skills graph.")
