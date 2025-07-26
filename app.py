# app.py
import streamlit as st
import os
import re
import requests
import tempfile
import zipfile
from dotenv import load_dotenv
import streamlit.components.v1 as components
from utils import extract_text_from_pdf
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from contextlib import suppress
import base64
load_dotenv()

st.set_page_config(page_title="Career Advisor Agent", page_icon="💼", layout="centered")

# Set background image using base64 encoding
with open("1.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()
st.markdown(f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
    }}
    </style>
''', unsafe_allow_html=True)

# --- Ensure LLM session state is always initialized ---
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory()
if "llm" not in st.session_state:
    st.session_state["llm"] = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="LLaMA3-70b-8192",
        temperature=0.3
    )
if "conversation" not in st.session_state:
    st.session_state["conversation"] = ConversationChain(
        llm=st.session_state["llm"],
        memory=st.session_state["memory"]
    )

# Helper function for safe LLM calls
def safe_llm_run(prompt, spinner_msg="Thinking..."):
    try:
        with st.spinner(spinner_msg):
            return st.session_state["conversation"].run(prompt)
    except Exception as e:
        st.error(f"Could not connect to the LLM API. Please check your internet connection and API key.\nError: {e}")
        return None

# --- Custom Sidebar with Branding and Menu ---
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 1.5em;'>
        <img src='https://img.icons8.com/ios-filled/100/22223b/briefcase.png' width='60'/>
        <h2 style='color: #22223b; margin: 0.2em 0 0.5em 0;'>Career Advisor</h2>
    </div>
    """,
    unsafe_allow_html=True
)

menu = st.sidebar.radio(
    "Select a feature",
    [
        "Resume Upload",
        "Personality Quiz",
        "Career Chat",
        "Job Search",
        "Goal Tracker",
        "Interview Practice",
        "Portfolio Generator"
    ]
)

# Custom CSS for modern look
st.markdown('''
<style>
/* body, .stApp { background: #f7f7fa; } */
.big-title { font-size: 2.6rem; font-weight: 700; color: #22223b; margin-bottom: 0.5em; }
.subtext { color: #4a4e69; font-size: 1.1rem; margin-bottom: 2em; }
.upload-box { background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(34,34,59,0.07); padding: 2em 2em 1em 2em; }
.resume-feedback { background: #e9ecef; border-radius: 8px; padding: 1.2em; margin-top: 1.5em; }
</style>
''', unsafe_allow_html=True)

# --- Resume Upload & Feedback ---
if menu == "Resume Upload":
    st.markdown('<div class="big-title">💼 Career Advisor Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtext">Upload your resume to get instant feedback, corrections, and personalized learning suggestions to boost your career prospects.</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.header("Upload Your Resume (PDF)")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    job_desc = st.text_area("Paste a Job Description (optional, for keyword match)", value="")
    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.session_state["resume_text"] = resume_text
        st.success("Resume uploaded and parsed!")
        st.subheader("AI Suggestions for Your Resume")
        suggestion_prompt = (
            "You are a career advisor. Analyze the following resume and give specific, actionable suggestions for improvement, missing skills, or learning opportunities. "
            "Then, provide a concise summary of the resume in 2-3 sentences, and rate the resume out of 10 (as 'Score: X/10') based on overall quality, clarity, and relevance. "
            + ("If a job description is provided, compare the resume to the job description and list important keywords or skills from the job description that are missing in the resume as a bullet list under 'Missing Keywords'. " if job_desc else "")
            + "\n\nResume:\n" + resume_text
            + ("\n\nJob Description:\n" + job_desc if job_desc else "")
        )
        suggestions = safe_llm_run(suggestion_prompt, spinner_msg="Analyzing your resume...")
        if suggestions:
            st.markdown(suggestions)
    else:
        st.session_state["resume_text"] = None
        st.info("Please upload your resume to get started.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Personality Quiz ---
if menu == "Personality Quiz":
    def get_personality_profile(answers):
        traits = {
            "Extraversion": answers[0],
            "Conscientiousness": answers[1],
            "Openness": answers[2],
            "Neuroticism": answers[3],
            "Agreeableness": answers[4],
        }
        return traits
    def map_traits_to_careers(traits):
        suggestions = []
        summary = []
        if traits["Extraversion"] >= 3:
            suggestions += ["Sales Manager", "Event Planner", "Public Relations Specialist"]
            summary.append("You are outgoing and energized by social interaction.")
        else:
            suggestions += ["Data Analyst", "Researcher", "Technical Writer"]
            summary.append("You are introspective and may prefer focused, independent work.")
        if traits["Conscientiousness"] >= 3:
            suggestions += ["Project Manager", "Accountant", "Quality Assurance Engineer"]
            summary.append("You are organized and detail-oriented.")
        if traits["Openness"] >= 3:
            suggestions += ["Designer", "Entrepreneur", "Software Developer"]
            summary.append("You are creative and open to new experiences.")
        if traits["Neuroticism"] >= 3:
            suggestions += ["Counselor", "Writer", "Artist"]
            summary.append("You are sensitive and empathetic.")
        if traits["Agreeableness"] >= 3:
            suggestions += ["Teacher", "Nurse", "Human Resources Specialist"]
            summary.append("You are cooperative and value helping others.")
        return list(set(suggestions)), " ".join(summary)
    st.title("🧑‍💼 Personality Quiz (Big Five)")
    st.write("Take this short quiz to discover your personality profile and get career suggestions tailored to your traits.")
    personality_questions = [
        ("I enjoy social gatherings.", ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]),
        ("I like to plan and organize everything in advance.", ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]),
        ("I am open to trying new experiences.", ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]),
        ("I often feel anxious or stressed.", ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]),
        ("I am considerate and cooperative with others.", ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]),
    ]
    if "personality_answers" not in st.session_state:
        st.session_state["personality_answers"] = [2] * len(personality_questions)
    with st.form("personality_quiz_form"):
        answers = []
        for i, (q, opts) in enumerate(personality_questions):
            ans = st.radio(q, opts, index=st.session_state["personality_answers"][i], key=f"pq_{i}")
            answers.append(opts.index(ans))
        submitted = st.form_submit_button("Get Career Matches")
        if submitted:
            st.session_state["personality_answers"] = answers
            st.session_state["personality_quiz_done"] = True
    if st.session_state.get("personality_quiz_done"):
        traits = get_personality_profile(st.session_state["personality_answers"])
        careers, personality_summary = map_traits_to_careers(traits)
        st.markdown("---")
        st.subheader("🧑‍💼 Personality-Based Career Matches")
        st.write(f"**Personality Summary:** {personality_summary}")
        st.write("**Suggested Careers:**")
        for c in careers:
            st.markdown(f"- {c}")
    else:
        st.info("Take the quiz and submit to see your career matches.")

# --- Career Chat ---
if menu == "Career Chat":
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if st.session_state["messages"]:
        chat_text = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state["messages"]])
        st.download_button("Download Chat History", chat_text, file_name="career_chat.txt")

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state["messages"] = []

    # Chat logic
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input("Ask me anything about your career, roadmap, or courses...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # Add context if available
        context = ""
        if st.session_state.get("resume_text"):
            context += f"Here is the user's resume:\n{st.session_state['resume_text']}\n"
        persona_prompts = {
            "Career Coach": "You are a supportive and insightful career coach. Give actionable, motivational, and personalized career advice.",
            "Resume Reviewer": "You are an expert resume reviewer. Analyze resumes, suggest improvements, and optimize for ATS.",
            "Skill Trainer": "You are a practical skill trainer. Recommend learning resources, exercises, and step-by-step skill-building plans.",
            "Job Interviewer": "You are a professional job interviewer. Ask interview questions, evaluate answers, and provide constructive feedback."
        }
        persona_system = persona_prompts.get(st.session_state["chat_persona"], "")
        if persona_system:
            context = persona_system + "\n\n" + context
        if context:
            full_prompt = context + "\n" + prompt
        else:
            full_prompt = prompt
        response = safe_llm_run(full_prompt, spinner_msg="Thinking about your question...")
        if response:
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

    # --- Career Path Recommendation Feature ---
    st.markdown("---")
    st.subheader("🚀 Career Path Recommendation")
    st.write("Get a personalized learning path, course suggestions, and a roadmap based on your resume and your career goal.")
    with st.form("career_path_form"):
        user_goal = st.text_input("What is your career goal or target role? (e.g., I want to become a Data Analyst)")
        submit_path = st.form_submit_button("Get My Career Path")
    if submit_path and user_goal:
        resume_context = st.session_state.get("resume_text", "")
        path_prompt = (
            "You are a career advisor. Based on the following resume and the user's goal, recommend an ideal learning path (step-by-step), suggest relevant online courses (with links), and provide a clear roadmap to achieve the goal. "
            "Format the learning path as a numbered or bulleted list, list courses with platform and links, and provide a concise roadmap.\n\n"
            f"Resume:\n{resume_context}\n\nGoal: {user_goal}"
        )
        path_response = safe_llm_run(path_prompt, spinner_msg="Generating your personalized career path...")
        if path_response:
            st.markdown(path_response)

# --- Live Job Search ---
if menu == "Job Search":
    st.title("🔎 Live Job Search")
    st.write("Find live jobs matching your skills and interests. Enter your desired job title, location, and skills to get real-time job listings.")
    with st.form("job_search_form"):
        job_title = st.text_input("Job Title (e.g., Data Analyst)")
        location = st.text_input("Location (e.g., Bangalore)")
        skills_input = st.text_input("Skills (comma separated, optional)")
        submit_job_search = st.form_submit_button("Search Jobs")
    if submit_job_search and job_title and location:
        serp_api_key = "d513873ea6dd48900c6b68aff03428245161381645c604f118aebfe42713cb5d"
        st.info("Fetching live jobs from SERP API...")
        params = {
            "engine": "google_jobs",
            "q": f"{job_title} {skills_input}",
            "location": location,
            "api_key": serp_api_key
        }
        try:
            response = requests.get("https://serpapi.com/search", params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                jobs = data.get("jobs_results", [])
                if jobs:
                    st.markdown(f"#### Top {min(5, len(jobs))} Jobs for '{job_title}' in {location}")
                    job_rows = []
                    for job in jobs[:5]:
                        job_rows.append({
                            "Title": job.get("title", "-"),
                            "Company": job.get("company_name", "-"),
                            "Location": job.get("location", "-"),
                            "Apply": job.get("via", "-"),
                            "Link": job.get("job_highlights", [{}])[0].get("link", job.get("apply_options_link", "-"))
                        })
                    st.dataframe([{k: v for k, v in row.items() if k != "Link"} for row in job_rows])
                    for job in job_rows:
                        if job["Link"] and job["Link"] != "-":
                            st.markdown(f"- [{job['Title']} at {job['Company']} in {job['Location']}]({job['Link']})")
                        else:
                            st.markdown(f"- {job['Title']} at {job['Company']} in {job['Location']}")
                else:
                    st.warning("No jobs found for your query.")
            else:
                st.error(f"Job search API error: {response.status_code}")
        except Exception as e:
            st.error(f"Error fetching jobs: {e}")
    elif submit_job_search:
        st.warning("Please enter both job title and location.")

# --- Career Goal Tracker ---
if menu == "Goal Tracker":
    st.title("🎯 Career Goal Tracker")
    if "career_goals" not in st.session_state:
        st.session_state["career_goals"] = []
    if "goal_checklists" not in st.session_state:
        st.session_state["goal_checklists"] = {}

    with st.form("goal_form"):
        new_goal = st.text_input("Career Goal (e.g., Learn Power BI)")
        target_date = st.date_input("Target Date")
        add_goal = st.form_submit_button("Add Goal")
    if add_goal and new_goal:
        st.session_state["career_goals"].append({
            "goal": new_goal,
            "date": str(target_date)
        })
        st.session_state["goal_checklists"][new_goal] = []
        st.success(f"Added goal: {new_goal} (by {target_date})")

    if st.session_state["career_goals"]:
        for goal_item in st.session_state["career_goals"]:
            goal = goal_item["goal"]
            date = goal_item["date"]
            st.markdown(f"**Goal:** {goal}  ")
            st.markdown(f"**Target Date:** {date}")
            checklist = st.session_state["goal_checklists"].get(goal, [])
            if not checklist:
                suggest_prompt = (
                    f"Suggest a 3-step learning path or checklist to achieve this goal: {goal}. "
                    f"Be concise and actionable."
                )
                with st.spinner("Suggesting learning path..."):
                    suggestion = safe_llm_run(suggest_prompt, spinner_msg="Suggesting learning path...")
                steps = re.findall(r"(?:\d+\. |[-*] )(.+)", suggestion)
                if not steps:
                    steps = [suggestion]
                st.session_state["goal_checklists"][goal] = [
                    {"step": s, "done": False} for s in steps
                ]
                checklist = st.session_state["goal_checklists"][goal]
            for i, item in enumerate(checklist):
                checked = st.checkbox(item["step"], value=item["done"], key=f"{goal}_{i}")
                st.session_state["goal_checklists"][goal][i]["done"] = checked
            done_count = sum(1 for item in checklist if item["done"])
            st.progress(done_count / len(checklist))
            if done_count == len(checklist):
                st.success(f"Goal '{goal}' completed! 🎉")
    else:
        st.info("Add a career goal to start tracking your progress.")

# --- Interview Practice Agent ---
if menu == "Interview Practice":
    st.title("🗣️ Interview Practice Agent")
    # Session state for interview
    if "interview_active" not in st.session_state:
        st.session_state["interview_active"] = False
    if "interview_role" not in st.session_state:
        st.session_state["interview_role"] = ""
    if "interview_topic" not in st.session_state:
        st.session_state["interview_topic"] = "Technical"
    if "interview_q" not in st.session_state:
        st.session_state["interview_q"] = ""
    if "interview_a" not in st.session_state:
        st.session_state["interview_a"] = ""
    if "interview_feedback" not in st.session_state:
        st.session_state["interview_feedback"] = ""
    if "interview_history" not in st.session_state:
        st.session_state["interview_history"] = []

    # --- Mock Interview Generator ---
    with st.expander("📝 Mock Interview Generator", expanded=False):
        st.write("Generate 5 job-specific interview questions (technical + behavioral) for any role.")
        mock_role = st.text_input("Target Role for Mock Interview Questions", value="Backend Developer")
        if st.button("Generate Mock Interview Questions"):
            mock_prompt = f"Generate 5 interview questions for the role of {mock_role}, both technical and behavioral."
            with st.spinner("Generating questions..."):
                questions = safe_llm_run(mock_prompt, spinner_msg="Generating questions...")
            if questions:
                q_list = re.findall(r"\d+\.\s*(.+)", questions)
                if not q_list:
                    q_list = [q.strip() for q in questions.split("\n") if q.strip()]
                st.markdown("**Mock Interview Questions:**")
                for i, q in enumerate(q_list, 1):
                    st.markdown(f"{i}. {q}")

    with st.expander("Simulate a Mock Interview Session", expanded=True):
        if not st.session_state["interview_active"]:
            with st.form("start_interview_form"):
                role_options = ["Backend Developer", "Frontend Developer", "Data Scientist", "Product Manager", "Business Analyst", "Custom..."]
                role = st.selectbox("Select Interview Role", role_options)
                custom_role = st.text_input("Custom Role (if selected above)")
                topic = st.selectbox("Select Topic", ["Technical", "Behavioral"])
                start = st.form_submit_button("Start Interview")
            if start:
                st.session_state["interview_active"] = True
                st.session_state["interview_role"] = custom_role if role == "Custom..." and custom_role else role
                st.session_state["interview_topic"] = topic
                st.session_state["interview_history"] = []
                prompt = (
                    f"You are an expert interviewer for the role of {st.session_state['interview_role']}. "
                    f"Ask a {st.session_state['interview_topic']} interview question. "
                    f"If technical, cover topics like DSA, DBMS, etc. If behavioral, ask about experiences, challenges, or teamwork."
                )
                with st.spinner("Generating first question..."):
                    q = safe_llm_run(prompt, spinner_msg="Generating first question...")
                if q:
                    st.session_state["interview_q"] = q
                    st.session_state["interview_a"] = ""
                    st.session_state["interview_feedback"] = ""
                    st.success(f"Interview started for: {st.session_state['interview_role']} ({st.session_state['interview_topic']})")
        else:
            st.markdown(f"**Role:** {st.session_state['interview_role']}")
            st.markdown(f"**Topic:** {st.session_state['interview_topic']}")
            st.markdown(f"**Question:** {st.session_state['interview_q']}")
            with st.form("answer_form"):
                answer = st.text_area("Your Answer", value=st.session_state["interview_a"])
                submit_a = st.form_submit_button("Submit Answer")
            if submit_a and answer:
                st.session_state["interview_a"] = answer
                feedback_prompt = (
                    f"You are an expert interviewer for the role of {st.session_state['interview_role']}. "
                    f"Here is the candidate's answer to the {st.session_state['interview_topic']} question '{st.session_state['interview_q']}':\n{answer}\n"
                    f"Give constructive feedback and suggestions for improvement."
                )
                with st.spinner("Evaluating answer and generating feedback..."):
                    feedback = safe_llm_run(feedback_prompt, spinner_msg="Evaluating answer and generating feedback...")
                if feedback:
                    st.session_state["interview_feedback"] = feedback
                    st.session_state["interview_history"].append({
                        "question": st.session_state["interview_q"],
                        "answer": answer,
                        "feedback": feedback,
                        "topic": st.session_state["interview_topic"]
                    })
                    st.session_state["interview_a"] = ""
            if st.session_state["interview_feedback"]:
                st.markdown(f"**Feedback:** {st.session_state['interview_feedback']}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Next Question"):
                        next_q_prompt = (
                            f"You are an expert interviewer for the role of {st.session_state['interview_role']}. "
                            f"Ask another {st.session_state['interview_topic']} interview question. "
                            f"If technical, cover topics like DSA, DBMS, etc. If behavioral, ask about experiences, challenges, or teamwork."
                        )
                        with st.spinner("Generating next question..."):
                            next_q = safe_llm_run(next_q_prompt, spinner_msg="Generating next question...")
                        if next_q:
                            st.session_state["interview_q"] = next_q
                            st.session_state["interview_feedback"] = ""
                with col2:
                    if st.button("Switch Topic"):
                        st.session_state["interview_topic"] = "Behavioral" if st.session_state["interview_topic"] == "Technical" else "Technical"
                        switch_q_prompt = (
                            f"You are an expert interviewer for the role of {st.session_state['interview_role']}. "
                            f"Ask a {st.session_state['interview_topic']} interview question. "
                            f"If technical, cover topics like DSA, DBMS, etc. If behavioral, ask about experiences, challenges, or teamwork."
                        )
                        with st.spinner("Generating question for new topic..."):
                            switch_q = safe_llm_run(switch_q_prompt, spinner_msg="Generating question for new topic...")
                        if switch_q:
                            st.session_state["interview_q"] = switch_q
                            st.session_state["interview_feedback"] = ""
            if st.button("End Interview"):
                st.session_state["interview_active"] = False
                st.session_state["interview_q"] = ""
                st.session_state["interview_a"] = ""
                st.session_state["interview_feedback"] = ""
                st.success("Interview session ended.")
            if st.session_state["interview_history"]:
                st.markdown("---")
                st.markdown("#### Interview Q&A History")
                for idx, item in enumerate(st.session_state["interview_history"], 1):
                    st.markdown(f"**Q{idx} ({item['topic']}):** {item['question']}")
                    st.markdown(f"**Your Answer:** {item['answer']}")
                    st.markdown(f"**Feedback:** {item['feedback']}")
                    st.markdown("---")

# --- Portfolio Generator ---
if menu == "Portfolio Generator":
    st.title("🌐 Portfolio Generator")
    with st.form("portfolio_form"):
        name = st.text_input("Your Name", value="")
        interests = st.text_input("Interests (comma separated)", value="")
        summary = st.text_area("Professional Summary", value="")
        resume_text = st.text_area("Paste Resume Text (optional)", value="")
        photo_file = st.file_uploader("Upload a Profile Photo (optional)", type=["jpg", "jpeg", "png"])
        submit_portfolio = st.form_submit_button("Generate Portfolio")
    if submit_portfolio:
        st.markdown("#### Portfolio Summary")
        st.write(f"**Name:** {name}")
        st.write(f"**Interests:** {interests}")
        st.write(f"**Professional Summary:** {summary}")
        if resume_text:
            st.write(f"**Resume (truncated):** {resume_text[:500]}{'...' if len(resume_text) > 500 else ''}")
        if photo_file:
            st.image(photo_file, caption="Profile Photo", width=180)
