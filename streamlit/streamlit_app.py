import streamlit as st
from streamlit_option_menu import option_menu


st.markdown(
    """
    <style>
    .navbar-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 20px;
    }
    .content-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        width: 100vw;
    }
    </style>
    """,
    unsafe_allow_html=True
)

pages = {
    "Home": "home",
    "Build CNN": "build_cnn",
    "Train CNN": "train_cnn",
    "View Results": "view_results"
}

# Create a function for each page
def home():
    st.title("Welcome to Omdena Quantumn Self Driving Project")
    # Add content for the home page

def build_cnn():
    st.title("Build Your CNN Network")
    with st.form(key="cnn"):
        st.subheader("Input")
        layer1 = st.selectbox("Layer1",["CNN", "Activation", "MaxPool","Dense", "Flatten"])
        layer2 = st.selectbox("Layer2",["CNN", "Activation", "MaxPool","Dense", "Flatten"])
        layer3 = st.selectbox("Layer3",["CNN", "Activation", "MaxPool","Dense", "Flatten"])
        layer4 = st.selectbox("Layer4",["CNN", "Activation", "MaxPool","Dense", "Flatten"])
        layer5 = st.selectbox("Layer5",["CNN", "Activation", "MaxPool","Dense", "Flatten"])
        create_cnn = st.form_submit_button(label = "Create CNN")
    # Add content for the build CNN page

def train_cnn():
    st.title("Train Your CNN Network")
    # Add content for the train CNN page

def view_results():
    st.title("Results")

with st.sidebar:
    selection = option_menu("Go to", list(pages.keys()),
                        menu_icon="cast",
                        default_index=0,
                        )
# Create the navigation menu
# st.markdown('<div class="navbar-container">', unsafe_allow_html=True)
# # Create the navigation menu using option_menu
# selection = option_menu("Go to", list(pages.keys()),
#                         menu_icon="cast",
#                         default_index=0,
#                         orientation='horizontal')
# st.markdown('</div>', unsafe_allow_html=True)

# Center the content
st.markdown('<div class="content-container">', unsafe_allow_html=True)
# Map the selection to the corresponding function
page = pages[selection]

# Call the selected page function
if page == "home":
    home()
elif page == "build_cnn":
    build_cnn()
elif page == "train_cnn":
    train_cnn()
elif page == "view_results":
    view_results()
st.markdown('</div>', unsafe_allow_html=True)