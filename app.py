import streamlit as st
import pickle
import pandas as pd

st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #00cc89;
        color: white;
        height: 3rem;
        margin-top: 2rem;
    }
    .stButton>button:hover {
        background-color: #00b377;
    }
    .title {
        color: #00cc89;
        text-align: center;
        padding-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .stats-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-label {
        color: #ffffff;
        opacity: 0.8;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

pipe = load_model()

st.markdown("<h1 class='title'>🏏 IPL Win Predictor</h1>", unsafe_allow_html=True)

st.markdown("""
    <p style='text-align: center; color: #ffffff; opacity: 0.8; margin-bottom: 2rem;'>
    Predict the winning probability of IPL matches based on current match situation
    </p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='color: #00cc89;'>🏃‍♂️ Batting Team</h3>", unsafe_allow_html=True)
    batting_team = st.selectbox('Select the batting team', sorted(teams), key='batting')

with col2:
    st.markdown("<h3 style='color: #00cc89;'>⚾ Bowling Team</h3>", unsafe_allow_html=True)
    bowling_team = st.selectbox('Select the bowling team', sorted(teams), key='bowling')

st.markdown("<h3 style='color: #00cc89;'>🏟️ Venue</h3>", unsafe_allow_html=True)
selected_city = st.selectbox('Select host city', sorted(cities))

st.markdown("<h3 style='color: #00cc89;'>📊 Match Situation</h3>", unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    target = st.number_input('Target Score', min_value=0, help="Total runs to chase")
    
with col4:
    score = st.number_input('Current Score', min_value=0, help="Current runs scored")

col5, col6, col7 = st.columns(3)

with col5:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1, 
                           help="Overs completed (e.g., 10.4)")
with col6:
    wickets = st.number_input('Wickets Lost', min_value=0, max_value=10,
                             help="Number of wickets fallen")
with col7:
    st.markdown("<br>", unsafe_allow_html=True)

if st.button('Predict Win Probability', help="Click to calculate win probabilities"):
    if batting_team == bowling_team:
        st.error("⚠️ Batting and bowling teams cannot be the same!")
    elif score > target:
        st.error("⚠️ Current score cannot be greater than target!")
    elif overs >= 20:
        st.error("⚠️ Overs cannot exceed 20!")
    else:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs if overs != 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        st.markdown("<h3 style='color: #00cc89;'>📈 Match Statistics</h3>", unsafe_allow_html=True)
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.markdown("""
                <div class='stats-card'>
                    <div class='metric-label'>Required Run Rate</div>
                    <div class='metric-value'>{:.2f}</div>
                </div>
            """.format(rrr), unsafe_allow_html=True)
            
        with stats_col2:
            st.markdown("""
                <div class='stats-card'>
                    <div class='metric-label'>Current Run Rate</div>
                    <div class='metric-value'>{:.2f}</div>
                </div>
            """.format(crr), unsafe_allow_html=True)
            
        with stats_col3:
            st.markdown("""
                <div class='stats-card'>
                    <div class='metric-label'>Runs Left</div>
                    <div class='metric-value'>{}</div>
                </div>
            """.format(runs_left), unsafe_allow_html=True)
            
        with stats_col4:
            st.markdown("""
                <div class='stats-card'>
                    <div class='metric-label'>Balls Left</div>
                    <div class='metric-value'>{}</div>
                </div>
            """.format(balls_left), unsafe_allow_html=True)

        st.markdown("<h3 style='color: #00cc89;'>🎯 Win Probability</h3>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style='background-color: rgba(0, 204, 137, 0.2); padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                <h4 style='color: #ffffff;'>{batting_team}</h4>
                <div style="background-color: rgba(255, 255, 255, 0.1); height: 20px; border-radius: 10px; overflow: hidden;">
                    <div style="width: {win*100}%; background-color: #00cc89; height: 100%;"></div>
                </div>
                <h3 style='color: #ffffff;'>{win*100:.1f}%</h3>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style='background-color: rgba(255, 59, 59, 0.2); padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                <h4 style='color: #ffffff;'>{bowling_team}</h4>
                <div style="background-color: rgba(255, 255, 255, 0.1); height: 20px; border-radius: 10px; overflow: hidden;">
                    <div style="width: {loss*100}%; background-color: #ff3b3b; height: 100%;"></div>
                </div>
                <h3 style='color: #ffffff;'>{loss*100:.1f}%</h3>
            </div>
        """, unsafe_allow_html=True)
