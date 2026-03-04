import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Ben 10 Battle Predictor", layout="wide")

@st.cache_data
def load_and_prep():
    aliens = pd.read_csv('ben10_aliens_dataset.csv')
    battles = pd.read_csv('ben10_battle_dataset.csv')
    
    le_t = LabelEncoder()
    le_b = LabelEncoder()
    battles['terrain_n'] = le_t.fit_transform(battles['terrain'])
    battles['battle_type_n'] = le_b.fit_transform(battles['battle_type'])
    
    return aliens, battles, le_t, le_b

aliens_df, battles_df, le_t, le_b = load_and_prep()

features = ['hero_combat', 'hero_speed', 'hero_durability', 'hero_intelligence', 'terrain_n', 'battle_type_n']
X = battles_df[features]
y = battles_df['battle_outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
st.title("🛡️ Omnitrix Battle Simulator")
st.markdown("---")
col_a, col_b, col_c = st.columns([2, 1, 1])

with col_a:
    st.subheader("1. Select your Alien")
    selected_hero = st.selectbox("Choose from the Omnitrix:", aliens_df['name'].unique())
    hero_stats = aliens_df[aliens_df['name'] == selected_hero].iloc[0]

with col_b:
    st.subheader("2. Terrain")
    selected_terrain = st.selectbox("Battle Location:", battles_df['terrain'].unique())

with col_c:
    st.subheader("3. Mode")
    selected_battle = st.selectbox("Engagement Type:", battles_df['battle_type'].unique())

st.write("### Hero Attributes")
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
stat_col1.metric("Combat", hero_stats['combat'])
stat_col2.metric("Speed", hero_stats['speed'])
stat_col3.metric("Durability", hero_stats['durability'])
stat_col4.metric("Intelligence", hero_stats['intelligence'])

st.markdown("---")
if st.button("INITIATE BATTLE PREDICTION", use_container_width=True):
    terrain_val = le_t.transform([selected_terrain])[0]
    type_val = le_b.transform([selected_battle])[0]
    
    input_data = [[
        hero_stats['combat'], 
        hero_stats['speed'], 
        hero_stats['durability'], 
        hero_stats['intelligence'],
        terrain_val,
        type_val
    ]]
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 'Win':
        st.success(f"### VICTORY: {selected_hero} dominates the battlefield!")
        st.balloons()
    elif prediction == 'Loss':
        st.error(f"### DEFEAT: {selected_hero} is overwhelmed by the villain.")
    else:
        st.warning(f"### STALEMATE: The battle is a dead draw.")

st.write("")
st.markdown("---")
bottom_col1, bottom_col2 = st.columns([3, 1])
with bottom_col2:
    st.info(f"**System Accuracy:** {acc*100:.2f}%")
    st.caption(f"Trained on {len(X_train)} historical battles")