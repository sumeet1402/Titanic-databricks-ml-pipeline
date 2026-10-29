import streamlit as st
import requests

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Titanic AI | Survival Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------
def prepare_features(
    pclass, sex, age, fare, sibsp, parch,
    embarked, title, cabin_deck
):
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    # Male = 0, Female = 1
    sex_encoded = 0 if sex == "Male" else 1

    # Match the notebook's pd.cut bins: [0, 12, 17, 59, 100].
    # Values at or below 0 are outside those bins, so all age-group
    # dummy columns remain 0 for age 0, just as in the notebook.
    age_group = (
        "Child" if 0 < age <= 12 else
        "Teen" if 12 < age <= 17 else
        "Adult" if 17 < age <= 59 else
        "Senior" if 59 < age <= 100 else
        None
    )

    family_group = (
        "Alone" if family_size == 1 else
        "Small" if family_size <= 4 else
        "Medium" if family_size <= 6 else
        "Large"
    )

    title = title if title in [
        "Mr", "Mrs", "Miss", "Master"
    ] else "Other"

    features = {
        "Pclass": pclass,
        "Sex": sex_encoded,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "FamilySize": family_size,
        "IsAlone": is_alone,
    }

    for value in ["C", "Q", "S"]:
        features[f"Embarked_{value}"] = int(
            embarked == value
        )

    for value in ["Master", "Miss", "Mr", "Mrs", "Other"]:
        features[f"Title_{value}"] = int(title == value)

    for value in ["Child", "Teen", "Adult", "Senior"]:
        features[f"AgeGroup_{value}"] = int(
            age_group == value
        )

    for value in ["Alone", "Small", "Medium", "Large"]:
        features[f"FamilySizeGroup_{value}"] = int(
            family_group == value
        )

    for value in ["A", "B", "C", "D", "E", "F", "G", "T", "Unknown"]:
        features[f"Cabin_Deck_{value}"] = int(
            cabin_deck == value
        )

    return features

# --------------------------------------------------
# 4. DATABRICKS MODEL SERVING
# --------------------------------------------------

# Use the narrow inference scope assigned to this service principal.
# Credentials stay in .streamlit/secrets.toml and are never shown in the UI.
def get_access_token():
    cfg = st.secrets["databricks"]
    host = cfg["host"].rstrip("/")

    response = requests.post(
        f"{host}/oidc/v1/token",
        auth=(cfg["client_id"], cfg["client_secret"]),
        data={
            "grant_type": "client_credentials",
            "scope": "model-serving-inference",
        },
        timeout=30,
    )

    if not response.ok:
        # The OAuth error body is safe to show; do not print request headers.
        raise RuntimeError(
            f"OAuth token request failed (HTTP {response.status_code}): "
            f"{response.text[:600]}"
        )

    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("OAuth response did not include an access token.")
    return token


def predict_survival(features):
    # Keep the order aligned with the registered model's 33-feature signature.
    columns = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
        "FamilySize", "IsAlone",
        "Embarked_C", "Embarked_Q", "Embarked_S",
        "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Other",
        "AgeGroup_Child", "AgeGroup_Teen", "AgeGroup_Adult", "AgeGroup_Senior",
        "FamilySizeGroup_Alone", "FamilySizeGroup_Small",
        "FamilySizeGroup_Medium", "FamilySizeGroup_Large",
        "Cabin_Deck_A", "Cabin_Deck_B", "Cabin_Deck_C", "Cabin_Deck_D",
        "Cabin_Deck_E", "Cabin_Deck_F", "Cabin_Deck_G", "Cabin_Deck_T",
        "Cabin_Deck_Unknown",
    ]

    missing = [column for column in columns if column not in features]
    if missing:
        raise ValueError(f"Missing model input features: {missing}")

    host = st.secrets["databricks"]["host"].rstrip("/")
    token = get_access_token()

    payload = {
        "dataframe_split": {
            "columns": columns,
            "index": [0],
            "data": [[features[column] for column in columns]],
        }
    }

    response = requests.post(
        f"{host}/serving-endpoints/titanic-survival-endpoint/invocations",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"Model endpoint request failed (HTTP {response.status_code}): "
            f"{response.text[:800]}"
        )

    result = response.json()
    predictions = result.get("predictions")
    if isinstance(predictions, list) and predictions:
        return predictions[0]

    # Some serving response formats return outputs rather than predictions.
    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict) and "0" in first:
            return first["0"]
        return first

    raise RuntimeError(
        "Unexpected endpoint response structure. "
        f"Returned keys: {list(result.keys())}"
    )


# --------------------------------------------------
# 5. LANDING PAGE HEADER AND STYLING
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background: radial-gradient(ellipse at 15% 0%, #14284a 0%, transparent 45%),
                linear-gradient(135deg, #080d19 0%, #0d1729 55%, #101d33 100%);
    color: #e6edf8;
    font-family: 'Inter', sans-serif;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, .section-title, .brand, .stat-value, .info-value {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.6px;
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 1120px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.brand {
    color: #93c5fd;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

.hero {
    padding: 55px 38px;
    margin: 22px 0 32px;
    border-radius: 20px;
    border: 1px solid #29466e;
    background: linear-gradient(120deg, #142747, #0e192c 75%);
    box-shadow: 0 12px 45px rgba(0, 0, 0, 0.18);
}

.eyebrow {
    color: #60a5fa;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

.hero h1 {
    color: #f3f7ff;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: clamp(36px, 5vw, 54px);
    line-height: 1.12;
    letter-spacing: -1.8px;
    margin: 16px 0;
}

.hero p {
    color: #b6c5dc;
    font-size: 16px;
    line-height: 1.8;
    max-width: 650px;
}

.hero-tag {
    display: inline-block;
    color: #bfdbfe;
    background: #1e3a5f;
    border: 1px solid #315580;
    border-radius: 20px;
    padding: 7px 12px;
    font-size: 12px;
    margin-top: 12px;
}

.section-title {
    color: #eff6ff;
    font-size: 27px;
    font-weight: 600;
    margin-bottom: 5px;
}

.section-subtitle, .section-description {
    color: #9eb1cc;
    font-size: 14px;
    line-height: 1.8;
    margin-bottom: 22px;
}

.info-card, .stat-card {
    background: linear-gradient(145deg, #111f34, #0d1728);
    border: 1px solid #263d5c;
    border-radius: 14px;
    padding: 19px;
    min-height: 100px;
}

.info-label, .stat-label {
    color: #93a8c5;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.3px;
}

.info-value, .stat-value {
    color: #f0f6ff;
    font-size: 19px;
    font-weight: 650;
    margin-top: 9px;
}

.stat-detail {
    color: #7db4ff;
    font-size: 12px;
    margin-top: 7px;
}

div[data-testid="stForm"] {
    background: rgba(15, 27, 46, 0.92);
    border: 1px solid #263d5c;
    border-radius: 17px;
    padding: 24px;
}

div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: #ffffff;
    border: 1px solid #5a9bff;
    border-radius: 10px;
    min-height: 48px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    letter-spacing: 0.2px;
    transition: all 0.2s ease;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(90deg, #3478ff, #60a5fa);
    box-shadow: 0 0 22px rgba(59, 130, 246, 0.25);
    color: #ffffff;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #111f34, #0d1728);
    border: 1px solid #263d5c;
    border-radius: 14px;
    padding: 17px;
}

div[data-testid="stMetricLabel"] { color: #9eb1cc; }

div[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #eaf2ff;
}

div[data-testid="stAlert"] { border-radius: 10px; }

.prediction-card {
    border-radius: 18px;
    padding: 30px;
    margin-top: 20px;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
.prediction-card.survived {
    background: radial-gradient(ellipse at 100% 0%, rgba(34, 197, 94, 0.18), transparent 58%),
                linear-gradient(145deg, #102f29, #0b1c20);
    border: 1px solid #22c55e;
    box-shadow: 0 0 30px rgba(34, 197, 94, 0.18);
}
.prediction-card.not-survived {
    background: radial-gradient(ellipse at 100% 0%, rgba(250, 204, 21, 0.16), transparent 58%),
                linear-gradient(145deg, #302713, #17150f);
    border: 1px solid #eab308;
    box-shadow: 0 0 30px rgba(234, 179, 8, 0.16);
}
.prediction-card.survived .prediction-label { color: #86efac; }
.prediction-card.survived .prediction-title {
    color: #bbf7d0;
    text-shadow: 0 0 18px rgba(74, 222, 128, 0.28);
}
.prediction-card.survived .prediction-description { color: #c4f1d4; }
.prediction-card.not-survived .prediction-label { color: #fde047; }
.prediction-card.not-survived .prediction-title {
    color: #fef08a;
    text-shadow: 0 0 18px rgba(250, 204, 21, 0.25);
}
.prediction-card.not-survived .prediction-description { color: #f3e6b2; }

.prediction-label {
    color: #7db4ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

.prediction-title {
    color: #f0f6ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(25px, 3vw, 35px);
    font-weight: 600;
    letter-spacing: -1px;
    line-height: 1.2;
    margin: 15px 0 12px;
}

.prediction-description {
    color: #aabbd2;
    font-size: 14px;
    line-height: 1.8;
    margin-bottom: 0;
}

hr { border-color: #253a57; }

.footer {
    color: #7589a7;
    font-size: 11px;
    letter-spacing: 1.4px;
    text-align: center;
    padding-top: 30px;
}

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 6. NAVIGATION HEADER
# --------------------------------------------------

nav_left, nav_right = st.columns([3, 1])

with nav_left:
    st.markdown(
        '<div class="brand">TITANIC / MACHINE LEARNING</div>',
        unsafe_allow_html=True
    )

with nav_right:
    st.markdown(
        "<div style='text-align:right;color:#93a8c5;"
        "font-size:13px;padding-top:3px;'>"
        "MODEL INFERENCE</div>",
        unsafe_allow_html=True
    )


# --------------------------------------------------
# 7. HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">Machine Learning Prediction Platform</div>
    <h1>Predict Titanic<br>Passenger Survival</h1>
    <p>
        Explore how passenger characteristics can influence a
        survival prediction. Enter passenger details below and
        receive a prediction from a trained Logistic Regression
        model served through Databricks.
    </p>
    <div class="hero-tag">
        Logistic Regression &nbsp; | &nbsp; Databricks Model Serving
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 8. PROJECT INFORMATION CARDS
# --------------------------------------------------

info1, info2, info3 = st.columns(3)

with info1:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Model</div>
        <div class="info-value">Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)

with info2:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Prediction target</div>
        <div class="info-value">Passenger survival</div>
    </div>
    """, unsafe_allow_html=True)

with info3:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Serving platform</div>
        <div class="info-value">Databricks</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")
st.write("")


# --------------------------------------------------
# 9. PASSENGER INPUT FORM
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Passenger details</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Complete the fields below to generate a survival prediction.'
    '</div>',
    unsafe_allow_html=True
)

with st.form("passenger_form"):

    st.markdown("#### Personal information")

    col1, col2, col3 = st.columns(3)

    with col1:
        pclass = st.selectbox(
            "Passenger class",
            options=[1, 2, 3]
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            options=["Female", "Male"]
        )

    with col3:
        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0
        )

    st.divider()

    st.markdown("#### Ticket and journey")

    col4, col5, col6 = st.columns(3)

    with col4:
        fare = st.number_input(
            "Ticket fare",
            min_value=0.0,
            value=15.0,
            step=1.0
        )

    with col5:
        embarked = st.selectbox(
            "Port of embarkation",
            options=[
                "Southampton (S)",
                "Cherbourg (C)",
                "Queenstown (Q)"
            ]
        )

    with col6:
        cabin_deck = st.selectbox(
            "Cabin deck",
            options=[
                "Unknown", "A", "B", "C", "D",
                "E", "F", "G", "T"
            ]
        )

    st.divider()

    st.markdown("#### Family information")

    col7, col8, col9 = st.columns(3)

    with col7:
        sibsp = st.number_input(
            "Siblings / spouses aboard",
            min_value=0,
            max_value=10,
            value=0
        )

    with col8:
        parch = st.number_input(
            "Parents / children aboard",
            min_value=0,
            max_value=10,
            value=0
        )

    with col9:
        title = st.selectbox(
            "Passenger title",
            options=["Mr", "Mrs", "Miss", "Master", "Other"]
        )

    submitted = st.form_submit_button(
        "Predict passenger survival",
        use_container_width=True
    )




# --------------------------------------------------
# 10. PROFESSIONAL PREDICTION RESULT
# --------------------------------------------------

if submitted:
    embarked_code = embarked[-2]

    features = prepare_features(
        pclass=pclass,
        sex=sex,
        age=age,
        fare=fare,
        sibsp=sibsp,
        parch=parch,
        embarked=embarked_code,
        title=title,
        cabin_deck=cabin_deck
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Prediction result</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Your passenger assessment from the deployed ML model.'
        '</div>',
        unsafe_allow_html=True
    )

    # Passenger summary
    st.markdown("#### Passenger summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Passenger class", f"Class {pclass}")

    with col2:
        st.metric("Age", f"{age:g} years")

    with col3:
        st.metric("Family size", features["FamilySize"])

    try:
        with st.spinner("Requesting prediction from Databricks..."):
            prediction = predict_survival(features)

        # Accept only the binary class labels returned by the model.
        if prediction in (1, "1"):
            st.markdown("""
            <div class="prediction-card survived">
                <div class="prediction-label">MODEL OUTPUT / CLASS 01</div>
                <div class="prediction-title">Predicted to survive</div>
                <p class="prediction-description">
                    The model predicts that this passenger survived.
                </p>
            </div>
            """, unsafe_allow_html=True)

        elif prediction in (0, "0"):
            st.markdown("""
            <div class="prediction-card not-survived">
                <div class="prediction-label">MODEL OUTPUT / CLASS 00</div>
                <div class="prediction-title">Predicted not to survive</div>
                <p class="prediction-description">
                    The model predicts that this passenger did not survive.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error(f"Unexpected prediction returned: {prediction}")

        # Additional passenger information
        st.write("")
        with st.expander("View submitted passenger details"):
            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.write(f"**Sex:** {sex}")
                st.write(f"**Ticket fare:** {fare:.2f}")
                st.write(f"**Passenger title:** {title}")
                st.write(f"**Port:** {embarked}")

            with detail_col2:
                st.write(f"**Cabin deck:** {cabin_deck}")
                st.write(f"**Siblings / spouses:** {sibsp}")
                st.write(f"**Parents / children:** {parch}")
                st.write(
                    f"**Travelling alone:** "
                    f"{'Yes' if features['IsAlone'] else 'No'}"
                )

        st.caption(
            "This is a machine learning prediction, not a guarantee "
            "of the passenger's actual outcome."
        )

    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        st.error(
            f"Databricks returned an HTTP error (status {status_code}). "
            "Check that the serving endpoint is ready and your token has permission."
        )

    except requests.exceptions.Timeout:
        st.error(
            "The request to Databricks timed out. Check that the serving "
            "endpoint is ready, then try again."
        )

    except requests.exceptions.RequestException:
        st.error(
            "Unable to connect to Databricks. Check your internet connection "
            "and the serving endpoint status, then try again."
        )

    except (KeyError, ValueError, IndexError, TypeError):
        st.error(
            "The prediction response could not be interpreted. "
            "Check the endpoint response and feature schema."
        )

    except Exception as exc:
        # Show useful diagnostics without displaying client secrets or access tokens.
        st.error(f"Prediction failed ({type(exc).__name__}): {exc}")
