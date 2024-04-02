import streamlit as st
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import joblib
# from tensorflow.keras.models import load_model
import os
from keras.models import load_model
from keras import backend as K
from keras import backend as K
from PIL import Image
import numpy as np
import io
import base64
import json
import requests

# Define global variables
Usernm = ""

# Load encoder model
encoder_model_path = "model/encoder_model.h5"
encoder_model = load_model(encoder_model_path)

# Function for feature extraction
def feature_extraction(model, image_path, layer=4):
    """
    Extract features from the specified layer of the autoencoder model.
    Arguments:
    model: Trained autoencoder model.
    image_path: Path to the image to extract features from.
    layer: Layer index from which to extract features (default: 4).
    Returns:
    extracted_features: 1D array of extracted features.
    """
    # Load and preprocess the image
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension
    
    # Create a Keras function to extract features from the specified layer
    encoded = K.function([model.layers[0].input], [model.layers[layer].output])
    
    # Extract features from the image using the specified layer
    encoded_array = encoded([image])[0]
    
    # Pool the features
    pooled_array = encoded_array.max(axis=(-2, -1))  # Pool over both height and width dimensions
    
    return pooled_array.flatten()  # Flatten the pooled array to make it 1D

# Function to find similar images based on features
def find_similar_images(query_features, database_features):
    # Ensure query_features and database_features are 2D arrays
    if query_features.ndim != 2 or database_features.ndim != 2:
        query_features = np.atleast_2d(query_features)
        database_features = np.atleast_2d(database_features)

    # Check the number of features in query and database
    num_query_features = query_features.shape[1]
    num_db_features = database_features.shape[1]

    print("Query Features Shape:", query_features.shape)
    print("Database Features Shape:", database_features.shape)

    if num_query_features != num_db_features:
        raise ValueError("Number of features in query and database do not match.")

    # Compute cosine similarity
    similarities = cosine_similarity(query_features, database_features)

    # Get indices of top similar images
    similar_image_indices = np.argsort(similarities[0])[::-1]
    return similar_image_indices

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("image.jpg")

page_bg_image = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://media.istockphoto.com/id/1186232338/vector/marble-textured-light-colored-beige-vintage-paper-vector-illustration.jpg?s=612x612&w=0&k=20&c=MF4dl0cbwHVYoo8mz_LoaJXmjFambYK3fzdjuPTNCjM=");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stImage"]  > img {{
height: 400px;
width: 224px!important;
align-items: center;
justify-content: center;
justify-items: center;
}}

[data-testid="stImage"] {{
justify-content: center;
align-items: center;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

</style>
"""
def main():
    st.title("Find Related Photos")
    st.markdown(page_bg_image, unsafe_allow_html=True)
    if 'username' not in st.session_state or not st.session_state.username:
        login_signup_page()
    else:
        image_related_functionality_page()

def login_signup_page():
    st.title("Login/Sign up")
    st.write("Please log in or sign up to continue.")

    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'useremail' not in st.session_state:
        st.session_state.useremail = ''

    choice = st.selectbox('Login/Signup', ['Login', 'Sign up'])
    email = st.text_input('Email Address', key="email_input")
    password = st.text_input('Password', type='password', key="password_input")

    if choice == 'Sign up':
        username = st.text_input("Enter your unique username")

        if st.button('Create my account'):
            user = sign_up_with_email_and_password(email=email, password=password, username=username)

            if user:
                st.success('Account created successfully!')
                st.markdown('Please login using your email and password')
                st.balloons()
            else:
                st.warning('Signup failed. Please try again.')

    else:
        if st.button('Login', on_click=login):
            pass

    # JavaScript to blur the input fields after email and password are entered
    st.markdown(
        """
        <script>
        document.getElementById("email_input").addEventListener("input", function(){
            this.blur();
        });
        document.getElementById("password_input").addEventListener("input", function(){
            this.blur();
        });
        </script>
        """,
        unsafe_allow_html=True
    )


def login():
    try:
        userinfo = sign_in_with_email_and_password(st.session_state.email_input, st.session_state.password_input)
        if userinfo:
            st.session_state.username = userinfo['username']
            st.session_state.useremail = userinfo['email']
            global Usernm
            Usernm = userinfo['username']
            st.session_state.signedout = True
            st.session_state.signout = True
        else:
            st.warning('Login Failed')
    except Exception as e:
        st.warning(f'Login failed: {e}')

def image_related_functionality_page():
    st.title("Welcome, " + st.session_state.username + "!")
    # Create a dictionary to map filenames to feature vectors
    database_image_mapping = {}


    # Streamlit UI
    st.title("Find Related Photos")

    # Text input for the folder path containing database images
    database_image_folder_path = st.text_input("Enter the path to the folder containing database images:")

    # File uploader for the reference image
    uploaded_image = st.file_uploader("Upload a reference image", type=["jpg", "png", "jpeg"])

    # Display the uploaded image if it exists
    if uploaded_image is not None:
        try:
            uploaded_image_display = Image.open(uploaded_image)
            st.image(uploaded_image_display, caption='Uploaded Image', use_column_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
            st.error("Unable to open the uploaded image. Please make sure it's a valid image file.")

    # Button to find related photos
    if st.button("Find Related Photos") and uploaded_image is not None and database_image_folder_path:
        # Check if the specified folder exists
        if not os.path.exists(database_image_folder_path):
            st.error("Specified folder does not exist.")
        else:
            # Check if the specified folder contains any image files
            image_files = [f for f in os.listdir(database_image_folder_path) if os.path.isfile(os.path.join(database_image_folder_path, f)) and f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not image_files:
                st.error("Specified folder does not contain any valid image files.")
            else:
                # Extract features from the uploaded reference image
                uploaded_image_features = feature_extraction(encoder_model, uploaded_image)
                
                
                # Extract features from images in the folder and populate database_image_mapping
                database_image_features = []
                for filename in image_files:
                    image_path = os.path.join(database_image_folder_path, filename)
                    features = feature_extraction(encoder_model, image_path)
                    database_image_features.append(features)
                    # Populate database_image_mapping
                    database_image_mapping[filename] = features


                # Find similar images based on features
                similar_image_indices = find_similar_images(uploaded_image_features, database_image_features)

                # Display similar images in a grid view with three columns
                st.subheader("Related Photos:")
                num_columns = 3  # Number of columns in the grid
                num_images = len(similar_image_indices)
                num_rows = (num_images + num_columns - 1) // num_columns  # Calculate number of rows needed

                # Create a grid layout
                for i in range(num_rows):
                    cols = st.columns(num_columns)  # Create columns for each row
                    for j in range(num_columns):
                        idx = i * num_columns + j
                        if idx < num_images:
                            filename = list(database_image_mapping.keys())[similar_image_indices[idx]]
                            image_path = os.path.join(database_image_folder_path, filename)  # Corrected variable name
                            if os.path.exists(image_path):
                                image = Image.open(image_path)
                                cols[j].image(image, caption=f"Related Image {image_path}", use_column_width=True)
                            else:
                                st.write(f"Related Image {filename} not found")

def sign_up_with_email_and_password(email, password, username=None, return_secure_token=True):
    try:
        rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": return_secure_token
        }
        if username:
            payload["displayName"] = username 
        payload = json.dumps(payload)
        r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
        try:
            return r.json()['email']
        except:
            st.warning(r.json())
    except Exception as e:
        st.warning(f'Signup failed: {e}')

def sign_in_with_email_and_password(email=None, password=None, return_secure_token=True):
    rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

    try:
        payload = {
            "returnSecureToken": return_secure_token
        }
        if email:
            payload["email"] = email
        if password:
            payload["password"] = password
        payload = json.dumps(payload)
        print('payload sigin',payload)
        r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
        try:
            data = r.json()
            user_info = {
                'email': data['email'],
                'username': data.get('displayName')  # Retrieve username if available
            }
            return user_info
        except:
            st.warning(data)
    except Exception as e:
        st.warning(f'Signin failed: {e}')

if __name__ == "__main__":
    main()
