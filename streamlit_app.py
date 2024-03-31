# import streamlit as st
# import numpy as np
# from PIL import Image
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# from tensorflow.keras.models import load_model
# import os
# from keras import backend as K
# from tkinter import Tk, filedialog

# from PIL import Image
# import numpy as np
# import io

# # Function for feature extraction
# def feature_extraction(model, image_path, layer=4):
#     """
#     Extract features from the specified layer of the autoencoder model.
#     Arguments:
#     model: Trained autoencoder model.
#     image_path: Path to the image to extract features from.
#     layer: Layer index from which to extract features (default: 4).
#     Returns:
#     extracted_features: 1D array of extracted features.
#     """
#     # Load and preprocess the image
#     image = Image.open(image_path)
#     image = image.convert('RGB')
#     image = image.resize((224, 224))
#     image = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension
    
#     # Create a Keras function to extract features from the specified layer
#     encoded = K.function([model.layers[0].input], [model.layers[layer].output])
    
#     # Extract features from the image using the specified layer
#     encoded_array = encoded([image])[0]
    
#     # Pool the features
#     pooled_array = encoded_array.max(axis=(-2, -1))  # Pool over both height and width dimensions
    
#     return pooled_array.flatten()  # Flatten the pooled array to make it 1D

# # Function to extract features from images in the subfolders
# def extract_features_from_subfolders(folder_path, encoder_model):
#     # Initialize lists to store image features and mapping
#     image_features = []
#     image_mapping = {}

#     # Recursively traverse the main folder
#     for root, dirs, files in os.walk(folder_path):
#         # Iterate over the subfolders
#         for subfolder in dirs:
#             subfolder_path = os.path.join(root, subfolder)
#             # Iterate over the files in each subfolder
#             for filename in os.listdir(subfolder_path):
#                 # Check if the file is an image
#                 if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     # Get the full path to the image file
#                     image_path = os.path.join(subfolder_path, filename)
#                     # Extract features from the image
#                     features = feature_extraction(encoder_model, image_path)
#                     # Store the features in the list
#                     image_features.append(features)
#                     # Store the mapping of filename to features
#                     image_mapping[image_path] = features
    
#     # Convert the features list to numpy array
#     image_features = np.array(image_features)
    
#     return image_features, image_mapping

# # Function to find similar images based on features
# def find_similar_images(query_features, database_features):
#     # Ensure query_features and database_features are 2D arrays
#     if query_features.ndim != 2 or database_features.ndim != 2:
#         query_features = np.atleast_2d(query_features)
#         database_features = np.atleast_2d(database_features)

#     # Check the number of features in query and database
#     num_query_features = query_features.shape[1]
#     num_db_features = database_features.shape[1]

#     print("Query Features Shape:", query_features.shape)
#     print("Database Features Shape:", database_features.shape)

#     if num_query_features != num_db_features:
#         raise ValueError("Number of features in query and database do not match.")

#     # Compute cosine similarity
#     similarities = cosine_similarity(query_features, database_features)

#     # Get indices of top similar images
#     similar_image_indices = np.argsort(similarities[0])[::-1]
#     return similar_image_indices

# # Load encoder model
# encoder_model_path = "model/encoder_model.h5"
# encoder_model = load_model(encoder_model_path)


# # Streamlit UI
# st.title("Find Related Photos")

# # Function to get the folder path using Tkinter
# def get_folder_path():
#     root = Tk()
#     root.withdraw()
#     folder_path = filedialog.askdirectory()
#     root.destroy()
#     return folder_path

# # Function to get or set the selected folder path using session state
# def get_set_selected_folder_path():
#     if 'selected_folder_path' not in st.session_state:
#         st.session_state.selected_folder_path = None
#     return st.session_state.selected_folder_path

# # Button to select the folder containing subfolders with images
# selected_folder_path = get_set_selected_folder_path()
# if st.button("Select Folder"):
#     selected_folder_path = get_folder_path()
#     st.session_state.selected_folder_path = selected_folder_path
# st.write(f"Selected Folder: {selected_folder_path}")  # Display the selected folder path below the button

# # File uploader for the reference image
# uploaded_image = st.file_uploader("Upload a reference image", type=["jpg", "png", "jpeg"])

# # Display the uploaded image if it exists
# if uploaded_image is not None:
#     try:
#         uploaded_image_display = Image.open(uploaded_image)
#         st.image(uploaded_image_display, caption='Uploaded Image', use_column_width=True)
#     except Exception as e:
#         st.error(f"Error: {e}")
#         st.error("Unable to open the uploaded image. Please make sure it's a valid image file.")

# # Button to find related photos
# if st.button("Find Related Photos") and uploaded_image is not None and selected_folder_path:
#     # Check if the specified folder exists
#     if not os.path.exists(selected_folder_path):
#         st.error("Specified folder does not exist.")
#     else:
#         # Extract features from the uploaded reference image
#         uploaded_image_features = feature_extraction(encoder_model, uploaded_image)
        
#         # Extract features from images in the subfolders
#         database_image_features, database_image_mapping = extract_features_from_subfolders(selected_folder_path, encoder_model)

#         # Find similar images based on features
#         similar_image_indices = find_similar_images(uploaded_image_features, database_image_features)

#         # Display similar images in a grid view with three columns
#         st.subheader("Related Photos:")
#         num_columns = 3  # Number of columns in the grid
#         num_images = len(similar_image_indices)
#         num_rows = (num_images + num_columns - 1) // num_columns  # Calculate number of rows needed

#         # Create a grid layout
#         for i in range(num_rows):
#             cols = st.columns(num_columns)  # Create columns for each row
#             for j in range(num_columns):
#                 idx = i * num_columns + j
#                 if idx < num_images:
#                     image_path = list(database_image_mapping.keys())[similar_image_indices[idx]]
#                     image = Image.open(image_path)
#                     cols[j].image(image, caption=f"Related Image {image_path}", use_column_width=True)


# -----------------------------------------08-03-2024------------------------------------------------------------

# import streamlit as st
# import numpy as np
# from PIL import Image
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# from tensorflow.keras.models import load_model
# import os
# from keras import backend as K
# from tkinter import Tk, filedialog

# from PIL import Image
# import numpy as np
# import io



# # Function for feature extraction
# def feature_extraction(model, image_path, layer=4):
#     """
#     Extract features from the specified layer of the autoencoder model.
#     Arguments:
#     model: Trained autoencoder model.
#     image_path: Path to the image to extract features from.
#     layer: Layer index from which to extract features (default: 4).
#     Returns:
#     extracted_features: 1D array of extracted features.
#     """
#     # Load and preprocess the image
#     image = Image.open(image_path)
#     image = image.convert('RGB')
#     image = image.resize((224, 224))
#     image = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension
    
#     # Create a Keras function to extract features from the specified layer
#     encoded = K.function([model.layers[0].input], [model.layers[layer].output])
    
#     # Extract features from the image using the specified layer
#     encoded_array = encoded([image])[0]
    
#     # Pool the features
#     pooled_array = encoded_array.max(axis=(-2, -1))  # Pool over both height and width dimensions
    
#     return pooled_array.flatten()  # Flatten the pooled array to make it 1D

# # Function to extract features from images in the subfolders
# def extract_features_from_subfolders(folder_path, encoder_model):
#     # Initialize lists to store image features and mapping
#     image_features = []
#     image_mapping = {}

#     # Recursively traverse the main folder
#     for root, dirs, files in os.walk(folder_path):
#         # Iterate over the subfolders
#         for subfolder in dirs:
#             subfolder_path = os.path.join(root, subfolder)
#             # Iterate over the files in each subfolder
#             for filename in os.listdir(subfolder_path):
#                 # Check if the file is an image
#                 if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     # Get the full path to the image file
#                     image_path = os.path.join(subfolder_path, filename)
#                     # Extract features from the image
#                     features = feature_extraction(encoder_model, image_path)
#                     # Store the features in the list
#                     image_features.append(features)
#                     # Store the mapping of filename to features
#                     image_mapping[image_path] = features
    
#     # Convert the features list to numpy array
#     image_features = np.array(image_features)
    
#     return image_features, image_mapping

# # Function to find similar images based on features
# def find_similar_images(query_features, database_features):
#     # Ensure query_features and database_features are 2D arrays
#     if query_features.ndim != 2 or database_features.ndim != 2:
#         query_features = np.atleast_2d(query_features)
#         database_features = np.atleast_2d(database_features)

#     # Check the number of features in query and database
#     num_query_features = query_features.shape[1]
#     num_db_features = database_features.shape[1]

#     print("Query Features Shape:", query_features.shape)
#     print("Database Features Shape:", database_features.shape)

#     if num_query_features != num_db_features:
#         raise ValueError("Number of features in query and database do not match.")

#     # Compute cosine similarity
#     similarities = cosine_similarity(query_features, database_features)

#     # Get indices of top similar images
#     similar_image_indices = np.argsort(similarities[0])[::-1]
#     return similar_image_indices

# # Load encoder model
# encoder_model_path = "model/encoder_model.h5"
# encoder_model = load_model(encoder_model_path)

# # Streamlit UI
# st.title("Find Related Photos")

# # Function to get the folder path using Tkinter
# def get_folder_path():
#     root = Tk()
#     root.withdraw()
#     folder_path = filedialog.askdirectory()
#     root.destroy()
#     return folder_path

# # Function to get or set the selected folder path using session state
# def get_set_selected_folder_path():
#     if 'selected_folder_path' not in st.session_state:
#         st.session_state.selected_folder_path = None
#     return st.session_state.selected_folder_path

# # Button to select the folder containing subfolders with images or images directly
# selected_folder_path = get_set_selected_folder_path()
# if st.button("Select Folder"):
#     selected_folder_path = get_folder_path()
#     st.session_state.selected_folder_path = selected_folder_path
# st.write(f"Selected Folder: {selected_folder_path}")  # Display the selected folder path below the button

# # File uploader for the reference image
# uploaded_image = st.file_uploader("Upload a reference image", type=["jpg", "png", "jpeg"])

# # Display the uploaded image if it exists
# if uploaded_image is not None:
#     try:
#         uploaded_image_display = Image.open(uploaded_image)
#         st.image(uploaded_image_display, caption='Uploaded Image', use_column_width=True)
#     except Exception as e:
#         st.error(f"Error: {e}")
#         st.error("Unable to open the uploaded image. Please make sure it's a valid image file.")

# # Button to find related photos
# if st.button("Find Related Photos") and (uploaded_image is not None) and (selected_folder_path is not None):
#     # Check if the specified folder exists
#     if not os.path.exists(selected_folder_path):
#         st.error("Specified folder does not exist.")
#     else:
#         # Check if the selected folder contains subfolders
#         subfolders = [f for f in os.listdir(selected_folder_path) if os.path.isdir(os.path.join(selected_folder_path, f))]
#         if subfolders:  # If subfolders are found, extract features from images in subfolders
#             # Extract features from images in the subfolders
#             database_image_features, database_image_mapping = extract_features_from_subfolders(selected_folder_path, encoder_model)
#         else:  # If no subfolders are found, directly extract features from images in the selected folder
#             folder_path = selected_folder_path
#             # Get a list of image files in the selected folder
#             image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#             # Iterate over image files and extract features
#             image_features = []
#             image_mapping = {}
#             for filename in image_files:
#                 image_path = os.path.join(folder_path, filename)
#                 # Extract features from the image
#                 features = feature_extraction(encoder_model, image_path)
#                 # Store the features in the list
#                 image_features.append(features)
#                 # Store the mapping of filename to features
#                 image_mapping[image_path] = features
#             # Convert the features list to numpy array
#             database_image_features = np.array(image_features)
#             # Assign an empty dictionary to database_image_mapping in this scenario
#             database_image_mapping = {}

#         # Extract features from the uploaded reference image
#         uploaded_image_features = feature_extraction(encoder_model, uploaded_image)
        
#         # Find similar images based on features
#         similar_image_indices = find_similar_images(uploaded_image_features, database_image_features)

#         # Display similar images in a grid view with three columns
#         st.subheader("Related Photos:")
#         num_columns = 3  # Number of columns in the grid
#         num_images = len(similar_image_indices)
#         num_rows = (num_images + num_columns - 1) // num_columns  # Calculate number of rows needed

#         # Create a grid layout
#         for i in range(num_rows):
#             cols = st.columns(num_columns)  # Create columns for each row
#             for j in range(num_columns):
#                 idx = i * num_columns + j
#                 if idx < num_images:
#                     # Retrieve image path based on similar image indices
#                     if subfolders:
#                         image_path = list(database_image_mapping.keys())[similar_image_indices[idx]]
#                     else:
#                         image_path = list(image_mapping.keys())[similar_image_indices[idx]]
#                     image = Image.open(image_path)
#                     cols[j].image(image, caption=f"Related Image {image_path}", use_column_width=True)

# import tensorflow as tf
# import keras

# print("TensorFlow version:", tf.__version__)
# print("Keras version:", keras.__version__)




# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# import streamlit as st
# from PIL import Image
# import numpy as np
# import os
# from tensorflow.keras.models import load_model
# from keras import backend as K
# from sklearn.metrics.pairwise import cosine_similarity
# from tkinter import Tk, filedialog
# import json
# import requests
# import base64
# import firebase_admin
# from firebase_admin import firestore
# from firebase_admin import credentials
# from firebase_admin import auth


# app_name = "my_app"
# # Initialize Firebase
# try:
#     # Attempt to initialize the Firebase app with a unique name
#     cred = credentials.Certificate("pondering-5ff7c-c033cfade319.json")
#     firebase_admin.initialize_app(cred, name=app_name)
# except ValueError as e:
#     # Handle the case where the app with the specified name already exists
#     if "already exists" in str(e):
#         print(f"Firebase app with name '{app_name}' already exists.")
#     else:
#         # Handle other errors
#         print(f"Error during initialization: {e}")


# # cred = credentials.Certificate("pondering-5ff7c-c033cfade319.json")
# # firebase_admin.initialize_app(cred)

# # Function for feature extraction
# def feature_extraction(model, image_path, layer=4):
#     """
#     Extract features from the specified layer of the autoencoder model.
#     Arguments:
#     model: Trained autoencoder model.
#     image_path: Path to the image to extract features from.
#     layer: Layer index from which to extract features (default: 4).
#     Returns:
#     extracted_features: 1D array of extracted features.
#     """
#     # Load and preprocess the image
#     image = Image.open(image_path)
#     image = image.convert('RGB')
#     image = image.resize((224, 224))
#     image = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension
    
#     # Create a Keras function to extract features from the specified layer
#     encoded = K.function([model.layers[0].input], [model.layers[layer].output])
    
#     # Extract features from the image using the specified layer
#     encoded_array = encoded([image])[0]
    
#     # Pool the features
#     pooled_array = encoded_array.max(axis=(-2, -1))  # Pool over both height and width dimensions
    
#     return pooled_array.flatten()  # Flatten the pooled array to make it 1D

# # Function to extract features from images in the subfolders
# def extract_features_from_subfolders(folder_path, encoder_model):
#     # Initialize lists to store image features and mapping
#     image_features = []
#     image_mapping = {}

#     # Recursively traverse the main folder
#     for root, dirs, files in os.walk(folder_path):
#         # Iterate over the subfolders
#         for subfolder in dirs:
#             subfolder_path = os.path.join(root, subfolder)
#             # Iterate over the files in each subfolder
#             for filename in os.listdir(subfolder_path):
#                 # Check if the file is an image
#                 if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     # Get the full path to the image file
#                     image_path = os.path.join(subfolder_path, filename)
#                     # Extract features from the image
#                     features = feature_extraction(encoder_model, image_path)
#                     # Store the features in the list
#                     image_features.append(features)
#                     # Store the mapping of filename to features
#                     image_mapping[image_path] = features
    
#     # Convert the features list to numpy array
#     image_features = np.array(image_features)
    
#     return image_features, image_mapping

# # Function to find similar images based on features
# def find_similar_images(query_features, database_features):
#     # Ensure query_features and database_features are 2D arrays
#     if query_features.ndim != 2 or database_features.ndim != 2:
#         query_features = np.atleast_2d(query_features)
#         database_features = np.atleast_2d(database_features)

#     # Check the number of features in query and database
#     num_query_features = query_features.shape[1]
#     num_db_features = database_features.shape[1]

#     print("Query Features Shape:", query_features.shape)
#     print("Database Features Shape:", database_features.shape)

#     if num_query_features != num_db_features:
#         raise ValueError("Number of features in query and database do not match.")

#     # Compute cosine similarity
#     similarities = cosine_similarity(query_features, database_features)

#     # Get indices of top similar images
#     similar_image_indices = np.argsort(similarities[0])[::-1]
#     return similar_image_indices

# # Load encoder model
# encoder_model_path = "model/encoder_model.h5"
# encoder_model = load_model(encoder_model_path)

# # Function to authenticate users with email and password
# def authenticate_user(email, password):
#     try:
#         user = auth.sign_in_with_email_and_password(email, password)
#         return user
#     except Exception as e:
#         return None

# # Function to get the folder path using Tkinter
# def get_folder_path():
#     root = Tk()
#     root.withdraw()
#     folder_path = filedialog.askdirectory()
#     root.destroy()
#     return folder_path

# # Function to get or set the selected folder path using session state
# def get_set_selected_folder_path():
#     if 'selected_folder_path' not in st.session_state:
#         st.session_state.selected_folder_path = None
#     return st.session_state.selected_folder_path

# # Function to convert image to base64
# @st.cache
# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()


# # Streamlit UI
# def main():
#     st.title("Image Similarity Finder")

#     # # Login or Register section
#     # if 'user' not in st.session_state:
#     #     st.session_state.user = None

#     # if st.session_state.user is None:
#     #     st.subheader("Login or Register")
#     #     login_email = st.text_input("Email")
#     #     login_password = st.text_input("Password", type="password")

#     #     register_email = st.text_input("New Email for Registration")
#     #     register_password = st.text_input("New Password for Registration", type="password")
#     #     confirm_password = st.text_input("Confirm Password", type="password")

#     #     login_button_key = "login_button"
#     #     register_button_key = "register_button"

#     #     if st.button("Login", key=login_button_key) and authenticate_user(login_email, login_password) is not None:
#     #         st.session_state.user = True
#     #         st.success("Login successful!")

#     #     if st.button("Register", key=register_button_key) and register_password == confirm_password:
#     #         try:
#     #             user = auth.create_user_with_email_and_password(register_email, register_password)
#     #             st.success("Registration successful! Please log in.")
#     #         except Exception as e:
#     #             st.error(f"Error during registration: {e}")
#     #     elif st.button("Register", key=register_button_key) and register_password != confirm_password:
#     #         st.error("Passwords do not match. Please try again.")

#     # --------------------------------------------------------------
#     st.title('Welcome to :violet[Pondering] :sunglasses:')

#     if 'username' not in st.session_state:
#         st.session_state.username = ''
#     if 'useremail' not in st.session_state:
#         st.session_state.useremail = ''


#     def sign_up_with_email_and_password(email, password, username=None, return_secure_token=True):
#         try:
#             rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
#             payload = {
#                 "email": email,
#                 "password": password,
#                 "returnSecureToken": return_secure_token
#             }
#             if username:
#                 payload["displayName"] = username 
#             payload = json.dumps(payload)
#             r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
#             try:
#                 return r.json()['email']
#             except:
#                 st.warning(r.json())
#         except Exception as e:
#             st.warning(f'Signup failed: {e}')

#     def sign_in_with_email_and_password(email=None, password=None, return_secure_token=True):
#         rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

#         try:
#             payload = {
#                 "returnSecureToken": return_secure_token
#             }
#             if email:
#                 payload["email"] = email
#             if password:
#                 payload["password"] = password
#             payload = json.dumps(payload)
#             print('payload sigin',payload)
#             r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
#             try:
#                 data = r.json()
#                 user_info = {
#                     'email': data['email'],
#                     'username': data.get('displayName')  # Retrieve username if available
#                 }
#                 return user_info
#             except:
#                 st.warning(data)
#         except Exception as e:
#             st.warning(f'Signin failed: {e}')

#     def f(): 
#         try:
#             # user = auth.get_user_by_email(email)
#             # print(user.uid)
#             # st.session_state.username = user.uid
#             # st.session_state.useremail = user.email

#             userinfo = sign_in_with_email_and_password(st.session_state.email_input,st.session_state.password_input)
#             st.session_state.username = userinfo['username']
#             st.session_state.useremail = userinfo['email']

            
#             global Usernm
#             Usernm=(userinfo['username'])
            
#             st.session_state.signedout = True
#             st.session_state.signout = True    
  
            
#         except: 
#             st.warning('Login Failed')

#     def t():
#         st.session_state.signout = False
#         st.session_state.signedout = False   
#         st.session_state.username = ''


        
    
        
#     if "signedout"  not in st.session_state:
#         st.session_state["signedout"] = False
#     if 'signout' not in st.session_state:
#         st.session_state['signout'] = False    
        

        
    
#     if  not st.session_state["signedout"]: # only show if the state is False, hence the button has never been clicked
#         choice = st.selectbox('Login/Signup',['Login','Sign up'])
#         email = st.text_input('Email Address')
#         password = st.text_input('Password',type='password')
#         st.session_state.email_input = email
#         st.session_state.password_input = password

        

        
#         if choice == 'Sign up':
#             username = st.text_input("Enter  your unique username")
            
#             if st.button('Create my account'):
#                 # user = auth.create_user(email = email, password = password,uid=username)
#                 user = sign_up_with_email_and_password(email=email,password=password,username=username)
                
#                 st.success('Account created successfully!')
#                 st.markdown('Please Login using your email and password')
#                 st.balloons()
#         else:
#             # st.button('Login', on_click=f)          
#             st.button('Login', on_click=f)
#     else:
#         st.subheader("Upload Image")
#         uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

#         if uploaded_image is not None:
#             try:
#                 uploaded_image_display = Image.open(uploaded_image)
#                 st.image(uploaded_image_display, caption='Uploaded Image', use_column_width=True)
#             except Exception as e:
#                 st.error(f"Error: {e}")
#                 st.error("Unable to open the uploaded image. Please make sure it's a valid image file.")

#         st.subheader("Find Similar Images")

#         # Button to select the folder containing images
#         selected_folder_path = get_set_selected_folder_path()
#         if st.button("Select Folder"):
#             selected_folder_path = get_folder_path()
#             st.session_state.selected_folder_path = selected_folder_path

#         if selected_folder_path:
#             st.write(f"Selected Folder: {selected_folder_path}")

#         # Button to find similar images
#         if st.button("Find Similar Images") and uploaded_image is not None and selected_folder_path is not None:
#             if not os.path.exists(selected_folder_path):
#                 st.error("Specified folder does not exist.")
#             else:
#                 # Extract features from images in the selected folder
#                 folder_image_features, folder_image_mapping = extract_features_from_subfolders(selected_folder_path, encoder_model)

#                 # Extract features from the uploaded image
#                 uploaded_image_features = feature_extraction(encoder_model, uploaded_image)

#                 # Find similar images based on features
#                 similar_image_indices = find_similar_images(uploaded_image_features, folder_image_features)

#                 # Display similar images
#                 st.subheader("Similar Images")
#                 num_columns = 3
#                 num_images = len(similar_image_indices)
#                 num_rows = (num_images + num_columns - 1) // num_columns

#                 for i in range(num_rows):
#                     cols = st.columns(num_columns)
#                     for j in range(num_columns):
#                         idx = i * num_columns + j
#                         if idx < num_images:
#                             image_path = list(folder_image_mapping.keys())[similar_image_indices[idx]]
#                             image = Image.open(image_path)
#                             cols[j].image(image, caption=f"Similar Image {image_path}", use_column_width=True)

#     # def ap():
#     #     st.write('Posts')

# if __name__ == "__main__":
#     main()


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

# import streamlit as st
# import numpy as np
# from PIL import Image
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# from tensorflow.keras.models import load_model
# import os
# from keras import backend as K
# from tkinter import Tk, filedialog

# from PIL import Image
# import numpy as np
# import io

# # Function for feature extraction
# def feature_extraction(model, image_path, layer=4):
#     """
#     Extract features from the specified layer of the autoencoder model.
#     Arguments:
#     model: Trained autoencoder model.
#     image_path: Path to the image to extract features from.
#     layer: Layer index from which to extract features (default: 4).
#     Returns:
#     extracted_features: 1D array of extracted features.
#     """
#     # Load and preprocess the image
#     image = Image.open(image_path)
#     image = image.convert('RGB')
#     image = image.resize((224, 224))
#     image = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension
    
#     # Create a Keras function to extract features from the specified layer
#     encoded = K.function([model.layers[0].input], [model.layers[layer].output])
    
#     # Extract features from the image using the specified layer
#     encoded_array = encoded([image])[0]
    
#     # Pool the features
#     pooled_array = encoded_array.max(axis=(-2, -1))  # Pool over both height and width dimensions
    
#     return pooled_array.flatten()  # Flatten the pooled array to make it 1D

# # Function to extract features from images in the subfolders
# def extract_features_from_subfolders(folder_path, encoder_model):
#     # Initialize lists to store image features and mapping
#     image_features = []
#     image_mapping = {}

#     # Recursively traverse the main folder
#     for root, dirs, files in os.walk(folder_path):
#         # Iterate over the subfolders
#         for subfolder in dirs:
#             subfolder_path = os.path.join(root, subfolder)
#             # Iterate over the files in each subfolder
#             for filename in os.listdir(subfolder_path):
#                 # Check if the file is an image
#                 if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     # Get the full path to the image file
#                     image_path = os.path.join(subfolder_path, filename)
#                     # Extract features from the image
#                     features = feature_extraction(encoder_model, image_path)
#                     # Store the features in the list
#                     image_features.append(features)
#                     # Store the mapping of filename to features
#                     image_mapping[image_path] = features
    
#     # Convert the features list to numpy array
#     image_features = np.array(image_features)
    
#     return image_features, image_mapping

# # Function to find similar images based on features
# def find_similar_images(query_features, database_features):
#     # Ensure query_features and database_features are 2D arrays
#     if query_features.ndim != 2 or database_features.ndim != 2:
#         query_features = np.atleast_2d(query_features)
#         database_features = np.atleast_2d(database_features)

#     # Check the number of features in query and database
#     num_query_features = query_features.shape[1]
#     num_db_features = database_features.shape[1]

#     print("Query Features Shape:", query_features.shape)
#     print("Database Features Shape:", database_features.shape)

#     if num_query_features != num_db_features:
#         raise ValueError("Number of features in query and database do not match.")

#     # Compute cosine similarity
#     similarities = cosine_similarity(query_features, database_features)

#     # Get indices of top similar images
#     similar_image_indices = np.argsort(similarities[0])[::-1]
#     return similar_image_indices

# # Load encoder model
# encoder_model_path = "model/encoder_model.h5"
# encoder_model = load_model(encoder_model_path)


# # Streamlit UI
# st.title("Find Related Photos")

# # Function to get the folder path using Tkinter
# def get_folder_path():
#     root = Tk()
#     root.withdraw()
#     folder_path = filedialog.askdirectory()
#     root.destroy()
#     return folder_path

# # Function to get or set the selected folder path using session state
# def get_set_selected_folder_path():
#     if 'selected_folder_path' not in st.session_state:
#         st.session_state.selected_folder_path = None
#     return st.session_state.selected_folder_path

# # Button to select the folder containing subfolders with images
# selected_folder_path = get_set_selected_folder_path()
# if st.button("Select Folder"):
#     selected_folder_path = get_folder_path()
#     st.session_state.selected_folder_path = selected_folder_path
# st.write(f"Selected Folder: {selected_folder_path}")  # Display the selected folder path below the button

# # File uploader for the reference image
# uploaded_image = st.file_uploader("Upload a reference image", type=["jpg", "png", "jpeg"])

# # Display the uploaded image if it exists
# if uploaded_image is not None:
#     try:
#         uploaded_image_display = Image.open(uploaded_image)
#         st.image(uploaded_image_display, caption='Uploaded Image', use_column_width=True)
#     except Exception as e:
#         st.error(f"Error: {e}")
#         st.error("Unable to open the uploaded image. Please make sure it's a valid image file.")

# # Button to find related photos
# if st.button("Find Related Photos") and uploaded_image is not None and selected_folder_path:
#     # Check if the specified folder exists
#     if not os.path.exists(selected_folder_path):
#         st.error("Specified folder does not exist.")
#     else:
#         # Extract features from the uploaded reference image
#         uploaded_image_features = feature_extraction(encoder_model, uploaded_image)
        
#         # Extract features from images in the subfolders
#         database_image_features, database_image_mapping = extract_features_from_subfolders(selected_folder_path, encoder_model)

#         # Find similar images based on features
#         similar_image_indices = find_similar_images(uploaded_image_features, database_image_features)

#         # Display similar images in a grid view with three columns
#         st.subheader("Related Photos:")
#         num_columns = 3  # Number of columns in the grid
#         num_images = len(similar_image_indices)
#         num_rows = (num_images + num_columns - 1) // num_columns  # Calculate number of rows needed

#         # Create a grid layout
#         for i in range(num_rows):
#             cols = st.columns(num_columns)  # Create columns for each row
#             for j in range(num_columns):
#                 idx = i * num_columns + j
#                 if idx < num_images:
#                     image_path = list(database_image_mapping.keys())[similar_image_indices[idx]]
#                     image = Image.open(image_path)
#                     cols[j].image(image, caption=f"Related Image {image_path}", use_column_width=True)


# -----------------------------------------11-03-2024  [1]------------------------------------------------------------

# import streamlit as st
# import numpy as np
# from PIL import Image
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# from tensorflow.keras.models import load_model
# import os
# from keras import backend as K
# from tkinter import Tk, filedialog

# from PIL import Image
# import numpy as np
# import io
# import base64

# # Function for feature extraction
# def feature_extraction(model, image_path, layer=4):
#     """
#     Extract features from the specified layer of the autoencoder model.
#     Arguments:
#     model: Trained autoencoder model.
#     image_path: Path to the image to extract features from.
#     layer: Layer index from which to extract features (default: 4).
#     Returns:
#     extracted_features: 1D array of extracted features.
#     """
#     # Load and preprocess the image
#     image = Image.open(image_path)
#     image = image.convert('RGB')
#     image = image.resize((224, 224))
#     image = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension
    
#     # Create a Keras function to extract features from the specified layer
#     encoded = K.function([model.layers[0].input], [model.layers[layer].output])
    
#     # Extract features from the image using the specified layer
#     encoded_array = encoded([image])[0]
    
#     # Pool the features
#     pooled_array = encoded_array.max(axis=(-2, -1))  # Pool over both height and width dimensions
    
#     return pooled_array.flatten()  # Flatten the pooled array to make it 1D

# # Function to extract features from images in the subfolders
# def extract_features_from_subfolders(folder_path, encoder_model):
#     # Initialize lists to store image features and mapping
#     image_features = []
#     image_mapping = {}

#     # Recursively traverse the main folder
#     for root, dirs, files in os.walk(folder_path):
#         # Iterate over the subfolders
#         for subfolder in dirs:
#             subfolder_path = os.path.join(root, subfolder)
#             # Iterate over the files in each subfolder
#             for filename in os.listdir(subfolder_path):
#                 # Check if the file is an image
#                 if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     # Get the full path to the image file
#                     image_path = os.path.join(subfolder_path, filename)
#                     # Extract features from the image
#                     features = feature_extraction(encoder_model, image_path)
#                     # Store the features in the list
#                     image_features.append(features)
#                     # Store the mapping of filename to features
#                     image_mapping[image_path] = features
    
#     # Convert the features list to numpy array
#     image_features = np.array(image_features)
    
#     return image_features, image_mapping

# # Function to find similar images based on features
# def find_similar_images(query_features, database_features):
#     # Ensure query_features and database_features are 2D arrays
#     if query_features.ndim != 2 or database_features.ndim != 2:
#         query_features = np.atleast_2d(query_features)
#         database_features = np.atleast_2d(database_features)

#     # Check the number of features in query and database
#     num_query_features = query_features.shape[1]
#     num_db_features = database_features.shape[1]

#     print("Query Features Shape:", query_features.shape)
#     print("Database Features Shape:", database_features.shape)

#     if num_query_features != num_db_features:
#         raise ValueError("Number of features in query and database do not match.")

#     # Compute cosine similarity
#     similarities = cosine_similarity(query_features, database_features)

#     # Get indices of top similar images
#     similar_image_indices = np.argsort(similarities[0])[::-1]
#     return similar_image_indices

# # Load encoder model
# encoder_model_path = "model/encoder_model.h5"
# encoder_model = load_model(encoder_model_path)

# @st.cache_data
# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()


# img = get_img_as_base64("image.jpg")

# page_bg_image = f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
# background-image: url("https://media.istockphoto.com/id/1186232338/vector/marble-textured-light-colored-beige-vintage-paper-vector-illustration.jpg?s=612x612&w=0&k=20&c=MF4dl0cbwHVYoo8mz_LoaJXmjFambYK3fzdjuPTNCjM=");
# background-size: cover;
# background-position: top left;
# background-repeat: no-repeat;
# background-attachment: local;
# }}

# [data-testid="stImage"]  > img {{
# height: 400px;
# width: 224px!important;
# align-items: center;
# justify-content: center;
# justify-items: center;
# }}

# [data-testid="stImage"] {{
# justify-content: center;
# align-items: center;
# }}

# [data-testid="stSidebar"] > div:first-child {{
# background-image: url("data:image/png;base64,{img}");
# background-position: center; 
# background-repeat: no-repeat;
# background-attachment: fixed;
# }}

# [data-testid="stHeader"] {{
# background: rgba(0,0,0,0);
# }}

# [data-testid="stToolbar"] {{
# right: 2rem;
# }}

# </style>
# """


# def main():
#     # Streamlit UI
    
#     st.title("Find Related Photos")
#     st.markdown(page_bg_image, unsafe_allow_html=True)


#     if 'username' not in st.session_state:
#         st.session_state.username = ''
#     if 'useremail' not in st.session_state:
#         st.session_state.useremail = ''


#     def sign_up_with_email_and_password(email, password, username=None, return_secure_token=True):
#         try:
#             rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
#             payload = {
#                 "email": email,
#                 "password": password,
#                 "returnSecureToken": return_secure_token
#             }
#             if username:
#                 payload["displayName"] = username 
#             payload = json.dumps(payload)
#             r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
#             try:
#                 return r.json()['email']
#             except:
#                 st.warning(r.json())
#         except Exception as e:
#             st.warning(f'Signup failed: {e}')

#     def sign_in_with_email_and_password(email=None, password=None, return_secure_token=True):
#         rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

#         try:
#             payload = {
#                 "returnSecureToken": return_secure_token
#             }
#             if email:
#                 payload["email"] = email
#             if password:
#                 payload["password"] = password
#             payload = json.dumps(payload)
#             print('payload sigin',payload)
#             r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
#             try:
#                 data = r.json()
#                 user_info = {
#                     'email': data['email'],
#                     'username': data.get('displayName')  # Retrieve username if available
#                 }
#                 return user_info
#             except:
#                 st.warning(data)
#         except Exception as e:
#             st.warning(f'Signin failed: {e}')

#     def f(): 
#         try:
#             # user = auth.get_user_by_email(email)

#             # print(user.uid)
#             # st.session_state.username = user.uid
#             # st.session_state.useremail = user.email

#             userinfo = sign_in_with_email_and_password(st.session_state.email_input,st.session_state.password_input)
#             st.session_state.username = userinfo['username']
#             st.session_state.useremail = userinfo['email']

            
#             global Usernm
#             Usernm=(userinfo['username'])
            
#             st.session_state.signedout = True
#             st.session_state.signout = True    
  
            
#         except: 
#             st.warning('Login Failed')

#     def t():
#         st.session_state.signout = False
#         st.session_state.signedout = False   
#         st.session_state.username = ''


        
    
        
#     if "signedout"  not in st.session_state:
#         st.session_state["signedout"] = False
#     if 'signout' not in st.session_state:
#         st.session_state['signout'] = False    
        

        
    
#     if  not st.session_state["signedout"]: # only show if the state is False, hence the button has never been clicked
#         choice = st.selectbox('Login/Signup',['Login','Sign up'])
#         email = st.text_input('Email Address')
#         password = st.text_input('Password',type='password')
#         st.session_state.email_input = email
#         st.session_state.password_input = password

        

        
#         if choice == 'Sign up':
#             username = st.text_input("Enter  your unique username")
            
#             if st.button('Create my account'):
#                 # user = auth.create_user(email = email, password = password,uid=username)
#                 user = sign_up_with_email_and_password(email=email,password=password,username=username)
                
#                 st.success('Account created successfully!')
#                 st.markdown('Please Login using your email and password')
#                 st.balloons()
#         else:
#             # st.button('Login', on_click=f)          
#             st.button('Login', on_click=f)

#     # Function to get the folder path using Tkinter
#     def get_folder_path():
#         root = Tk()
#         root.withdraw()
#         folder_path = filedialog.askdirectory()
#         root.destroy()
#         return folder_path
#         # Function to get or set the selected folder path using session state
#     def get_set_selected_folder_path():
#         if 'selected_folder_path' not in st.session_state:
#             st.session_state.selected_folder_path = None
#         return st.session_state.selected_folder_path
#         # Button to select the folder containing subfolders with images or images directly
#     selected_folder_path = get_set_selected_folder_path()
#     if st.button("Select Folder"):
#         selected_folder_path = get_folder_path()
#         st.session_state.selected_folder_path = selected_folder_path
#     st.write(f"Selected Folder: {selected_folder_path}")  # Display the selected folder path below the button
#         # File uploader for the reference image
#     uploaded_image = st.file_uploader("Upload a reference image", type=["jpg", "png", "jpeg"])
#         # Display the uploaded image if it exists
#     if uploaded_image is not None:
#         try:
#             uploaded_image_display = Image.open(uploaded_image)
#             st.image(uploaded_image_display, caption='Uploaded Image', use_column_width=True)
#         except Exception as e:
#             st.error(f"Error: {e}")
#             st.error("Unable to open the uploaded image. Please make sure it's a valid image file.")
#         # Button to find related photos
#     if st.button("Find Related Photos") and (uploaded_image is not None) and (selected_folder_path is not None):
#             # Check if the specified folder exists
#         if not os.path.exists(selected_folder_path):
#             st.error("Specified folder does not exist.")
#         else:
#                 # Check if the selected folder contains subfolders
#             subfolders = [f for f in os.listdir(selected_folder_path) if os.path.isdir(os.path.join(selected_folder_path, f))]
#             if subfolders:  # If subfolders are found, extract features from images in subfolders
#                     # Extract features from images in the subfolders
#                 database_image_features, database_image_mapping = extract_features_from_subfolders(selected_folder_path, encoder_model)
#             else:  # If no subfolders are found, directly extract features from images in the selected folder
#                 folder_path = selected_folder_path
#                     # Get a list of image files in the selected folder
#                 image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#                     # Iterate over image files and extract features
#                 image_features = []
#                 image_mapping = {}
#                 for filename in image_files:
#                     image_path = os.path.join(folder_path, filename)
#                         # Extract features from the image
#                     features = feature_extraction(encoder_model, image_path)
#                         # Store the features in the list
#                     image_features.append(features)
#                         # Store the mapping of filename to features
#                     image_mapping[image_path] = features
#                     # Convert the features list to numpy array
#                 database_image_features = np.array(image_features)
#                     # Assign an empty dictionary to database_image_mapping in this scenario
#                 database_image_mapping = {}
#                 # Extract features from the uploaded reference image
#             uploaded_image_features = feature_extraction(encoder_model, uploaded_image)
                
#                 # Find similar images based on features     
#             similar_image_indices = find_similar_images(uploaded_image_features, database_image_features)
#                 # Display similar images in a grid view with three columns
#             st.subheader("Related Photos:")
#             num_columns = 3  # Number of columns in the grid
#             num_images = len(similar_image_indices)
#             num_rows = (num_images + num_columns - 1) // num_columns  # Calculate number of rows needed
#                 # Create a grid layout
#             for i in range(num_rows):
#                 cols = st.columns(num_columns)  # Create columns for each row
#                 for j in range(num_columns):
#                     idx = i * num_columns + j
#                     if idx < num_images:
#                             # Retrieve image path based on similar image indices
#                         if subfolders:
#                             image_path = list(database_image_mapping.keys())[similar_image_indices[idx]]
#                         else:
#                             image_path = list(image_mapping.keys())[similar_image_indices[idx]]
#                         image = Image.open(image_path)
#                         cols[j].image(image, caption=f"Related Image {image_path}", use_column_width=True)
# if __name__ == "__main__":
#     main()
    
    
    
#     # -----------------------------------------11-03-2024  [2 ]------------------------------------------------------------
#     # -----------------------------------------11-03-2024  [1 - perfect work ]------------------------------------------------------------


# import streamlit as st
# import numpy as np
# from PIL import Image
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# from tensorflow.keras.models import load_model
# import os
# from keras import backend as K
# from tkinter import Tk, filedialog

# from PIL import Image
# import numpy as np
# import io
# import base64
# import json
# import requests
# # Define global variables
# Usernm = ""

# # Function for feature extraction
# def feature_extraction(model, image_path, layer=4):
#     """
#     Extract features from the specified layer of the autoencoder model.
#     Arguments:
#     model: Trained autoencoder model.
#     image_path: Path to the image to extract features from.
#     layer: Layer index from which to extract features (default: 4).
#     Returns:
#     extracted_features: 1D array of extracted features.
#     """
#     # Load and preprocess the image
#     image = Image.open(image_path)
#     image = image.convert('RGB')
#     image = image.resize((224, 224))
#     image = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension
    
#     # Create a Keras function to extract features from the specified layer
#     encoded = K.function([model.layers[0].input], [model.layers[layer].output])
    
#     # Extract features from the image using the specified layer
#     encoded_array = encoded([image])[0]
    
#     # Pool the features
#     pooled_array = encoded_array.max(axis=(-2, -1))  # Pool over both height and width dimensions
    
#     return pooled_array.flatten()  # Flatten the pooled array to make it 1D

# # Function to extract features from images in the subfolders
# def extract_features_from_subfolders(folder_path, encoder_model):
#     # Initialize lists to store image features and mapping
#     image_features = []
#     image_mapping = {}

#     # Recursively traverse the main folder
#     for root, dirs, files in os.walk(folder_path):
#         # Iterate over the subfolders
#         for subfolder in dirs:
#             subfolder_path = os.path.join(root, subfolder)
#             # Iterate over the files in each subfolder
#             for filename in os.listdir(subfolder_path):
#                 # Check if the file is an image
#                 if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     # Get the full path to the image file
#                     image_path = os.path.join(subfolder_path, filename)
#                     # Extract features from the image
#                     features = feature_extraction(encoder_model, image_path)
#                     # Store the features in the list
#                     image_features.append(features)
#                     # Store the mapping of filename to features
#                     image_mapping[image_path] = features
    
#     # Convert the features list to numpy array
#     image_features = np.array(image_features)
    
#     return image_features, image_mapping

# # Function to find similar images based on features
# def find_similar_images(query_features, database_features):
#     # Ensure query_features and database_features are 2D arrays
#     if query_features.ndim != 2 or database_features.ndim != 2:
#         query_features = np.atleast_2d(query_features)
#         database_features = np.atleast_2d(database_features)

#     # Check the number of features in query and database
#     num_query_features = query_features.shape[1]
#     num_db_features = database_features.shape[1]

#     print("Query Features Shape:", query_features.shape)
#     print("Database Features Shape:", database_features.shape)

#     if num_query_features != num_db_features:
#         raise ValueError("Number of features in query and database do not match.")

#     # Compute cosine similarity
#     similarities = cosine_similarity(query_features, database_features)

#     # Get indices of top similar images
#     similar_image_indices = np.argsort(similarities[0])[::-1]
#     return similar_image_indices

# # Load encoder model
# encoder_model_path = "model/encoder_model.h5"
# encoder_model = load_model(encoder_model_path)

# @st.cache_data
# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()


# img = get_img_as_base64("image.jpg")

# page_bg_image = f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
# background-image: url("https://media.istockphoto.com/id/1186232338/vector/marble-textured-light-colored-beige-vintage-paper-vector-illustration.jpg?s=612x612&w=0&k=20&c=MF4dl0cbwHVYoo8mz_LoaJXmjFambYK3fzdjuPTNCjM=");
# background-size: cover;
# background-position: top left;
# background-repeat: no-repeat;
# background-attachment: local;
# }}

# [data-testid="stImage"]  > img {{
# height: 400px;
# width: 224px!important;
# align-items: center;
# justify-content: center;
# justify-items: center;
# }}

# [data-testid="stImage"] {{
# justify-content: center;
# align-items: center;
# }}

# [data-testid="stSidebar"] > div:first-child {{
# background-image: url("data:image/png;base64,{img}");
# background-position: center; 
# background-repeat: no-repeat;
# background-attachment: fixed;
# }}

# [data-testid="stHeader"] {{
# background: rgba(0,0,0,0);
# }}

# [data-testid="stToolbar"] {{
# right: 2rem;
# }}

# </style>
# """

# def main():
#     st.title("Find Related Photos")
#     if 'username' not in st.session_state or not st.session_state.username:
#         login_signup_page()
#     else:
#         image_related_functionality_page()

# def login_signup_page():
#     st.title("Login/Sign up")
#     st.write("Please log in or sign up to continue.")

#     if 'username' not in st.session_state:
#         st.session_state.username = ''
#     if 'useremail' not in st.session_state:
#         st.session_state.useremail = ''

#     choice = st.selectbox('Login/Signup', ['Login', 'Sign up'])
#     email = st.text_input('Email Address')
#     password = st.text_input('Password', type='password')
#     st.session_state.email_input = email
#     st.session_state.password_input = password

#     if choice == 'Sign up':
#         username = st.text_input("Enter your unique username")

#         if st.button('Create my account'):
#             user = sign_up_with_email_and_password(email=email, password=password, username=username)

#             if user:
#                 st.success('Account created successfully!')
#                 st.markdown('Please login using your email and password')
#                 st.balloons()
#             else:
#                 st.warning('Signup failed. Please try again.')

#     else:
#         if st.button('Login', on_click=login):
#             pass

# def login():
#     try:
#         userinfo = sign_in_with_email_and_password(st.session_state.email_input, st.session_state.password_input)
#         if userinfo:
#             st.session_state.username = userinfo['username']
#             st.session_state.useremail = userinfo['email']
#             global Usernm
#             Usernm = userinfo['username']
#             st.session_state.signedout = True
#             st.session_state.signout = True
#         else:
#             st.warning('Login Failed')
#     except Exception as e:
#         st.warning(f'Login failed: {e}')

# def image_related_functionality_page():
#     st.title("Welcome, " + st.session_state.username + "!")
#     # st.write("This is the image related functionality page.")
#     # st.write("Implement your image related functionality here.")

#     # Function to get the folder path using Tkinter
#     def get_folder_path():
#         root = Tk()
#         root.withdraw()
#         folder_path = filedialog.askdirectory()
#         root.destroy()
#         return folder_path
#         # Function to get or set the selected folder path using session state
#     def get_set_selected_folder_path():
#         if 'selected_folder_path' not in st.session_state:
#             st.session_state.selected_folder_path = None
#         return st.session_state.selected_folder_path
#         # Button to select the folder containing subfolders with images or images directly
#     selected_folder_path = get_set_selected_folder_path()
#     if st.button("Select Folder"):
#         selected_folder_path = get_folder_path()
#         st.session_state.selected_folder_path = selected_folder_path
#     st.write(f"Selected Folder: {selected_folder_path}")  # Display the selected folder path below the button
#         # File uploader for the reference image
#     uploaded_image = st.file_uploader("Upload a reference image", type=["jpg", "png", "jpeg"])
#         # Display the uploaded image if it exists
#     if uploaded_image is not None:
#         try:
#             uploaded_image_display = Image.open(uploaded_image)
#             st.image(uploaded_image_display, caption='Uploaded Image', use_column_width=True)
#         except Exception as e:
#             st.error(f"Error: {e}")
#             st.error("Unable to open the uploaded image. Please make sure it's a valid image file.")
#         # Button to find related photos
#     if st.button("Find Related Photos") and (uploaded_image is not None) and (selected_folder_path is not None):
#             # Check if the specified folder exists
#         if not os.path.exists(selected_folder_path):
#             st.error("Specified folder does not exist.")
#         else:
#                 # Check if the selected folder contains subfolders
#             subfolders = [f for f in os.listdir(selected_folder_path) if os.path.isdir(os.path.join(selected_folder_path, f))]
#             if subfolders:  # If subfolders are found, extract features from images in subfolders
#                     # Extract features from images in the subfolders
#                 database_image_features, database_image_mapping = extract_features_from_subfolders(selected_folder_path, encoder_model)
#             else:  # If no subfolders are found, directly extract features from images in the selected folder
#                 folder_path = selected_folder_path
#                     # Get a list of image files in the selected folder
#                 image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#                     # Iterate over image files and extract features
#                 image_features = []
#                 image_mapping = {}
#                 for filename in image_files:
#                     image_path = os.path.join(folder_path, filename)
#                         # Extract features from the image
#                     features = feature_extraction(encoder_model, image_path)
#                         # Store the features in the list
#                     image_features.append(features)
#                         # Store the mapping of filename to features
#                     image_mapping[image_path] = features
#                     # Convert the features list to numpy array
#                 database_image_features = np.array(image_features)
#                     # Assign an empty dictionary to database_image_mapping in this scenario
#                 database_image_mapping = {}
#                 # Extract features from the uploaded reference image
#             uploaded_image_features = feature_extraction(encoder_model, uploaded_image)
                
#                 # Find similar images based on features     
#             similar_image_indices = find_similar_images(uploaded_image_features, database_image_features)
#                 # Display similar images in a grid view with three columns
#             st.subheader("Related Photos:")
#             num_columns = 3  # Number of columns in the grid
#             num_images = len(similar_image_indices)
#             num_rows = (num_images + num_columns - 1) // num_columns  # Calculate number of rows needed
#                 # Create a grid layout
#             for i in range(num_rows):
#                 cols = st.columns(num_columns)  # Create columns for each row
#                 for j in range(num_columns):
#                     idx = i * num_columns + j
#                     if idx < num_images:
#                             # Retrieve image path based on similar image indices
#                         if subfolders:
#                             image_path = list(database_image_mapping.keys())[similar_image_indices[idx]]
#                         else:
#                             image_path = list(image_mapping.keys())[similar_image_indices[idx]]
#                         image = Image.open(image_path)
#                         cols[j].image(image, caption=f"Related Image {image_path}", use_column_width=True)

#     # Add more functionality as needed

# def sign_up_with_email_and_password(email, password, username=None, return_secure_token=True):
#     try:
#         rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
#         payload = {
#             "email": email,
#             "password": password,
#             "returnSecureToken": return_secure_token
#         }
#         if username:
#             payload["displayName"] = username 
#         payload = json.dumps(payload)
#         r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
#         try:
#             return r.json()['email']
#         except:
#             st.warning(r.json())
#     except Exception as e:
#         st.warning(f'Signup failed: {e}')

# def sign_in_with_email_and_password(email=None, password=None, return_secure_token=True):
#     rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

#     try:
#         payload = {
#             "returnSecureToken": return_secure_token
#         }
#         if email:
#             payload["email"] = email
#         if password:
#             payload["password"] = password
#         payload = json.dumps(payload)
#         print('payload sigin',payload)
#         r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
#         try:
#             data = r.json()
#             user_info = {
#                 'email': data['email'],
#                 'username': data.get('displayName')  # Retrieve username if available
#             }
#             return user_info
#         except:
#             st.warning(data)
#     except Exception as e:
#         st.warning(f'Signin failed: {e}')

# if __name__ == "__main__":
#     main()


# -------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from tensorflow.keras.models import load_model
import os
from keras import backend as K
# from tkinter import Tk, filedialog

from PIL import Image
import numpy as np
import io
import base64
import json
import requests

# Define global variables
Usernm = ""

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

# Function to extract features from images in the subfolders
def extract_features_from_subfolders(folder_path, encoder_model):
    # Initialize lists to store image features and mapping
    image_features = []
    image_mapping = {}

    # Recursively traverse the main folder
    for root, dirs, files in os.walk(folder_path):
        # Iterate over the subfolders
        for subfolder in dirs:
            subfolder_path = os.path.join(root, subfolder)
            try:
                # Iterate over the files in each subfolder
                for filename in os.listdir(subfolder_path):
                    # Check if the file is an image
                    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                        # Get the full path to the image file
                        image_path = os.path.join(subfolder_path, filename)
                        # Extract features from the image
                        features = feature_extraction(encoder_model, image_path)
                        # Store the features in the list
                        image_features.append(features)
                        # Store the mapping of filename to features
                        image_mapping[image_path] = features
            except PermissionError as e:
                print(f"PermissionError: {e} - Skipping folder {subfolder_path}")
    
    # Convert the features list to numpy array
    image_features = np.array(image_features)
    
    return image_features, image_mapping


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

# Load encoder model
encoder_model_path = "model/encoder_model.h5"
encoder_model = load_model(encoder_model_path)

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
    # st.write("This is the image related functionality page.")
    # st.write("Implement your image related functionality here.")

    # Function to get the folder path using Tkinter
    
    def get_folder_path():
        root = Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(initialdir="/", title="Select a drive or folder")
        root.destroy()
        return folder_path
    #     # Function to get or set the selected folder path using session state
    # def get_set_selected_folder_path():
    #     if 'selected_folder_path' not in st.session_state:
    #         st.session_state.selected_folder_path = None
    #     return st.session_state.selected_folder_path
    #     # Button to select the folder containing subfolders with images or images directly
    # selected_folder_path = get_set_selected_folder_path()
    # if st.button("Select Folder"):
    #     selected_folder_path = get_folder_path()
    #     st.session_state.selected_folder_path = selected_folder_path
    # st.write(f"Selected Folder: {selected_folder_path}")  # Display the selected folder path below the button
    #     # File uploader for the reference image
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
        
        # Function to get or set the selected folder path using session state
    def get_set_selected_folder_path():
        if 'selected_folder_path' not in st.session_state:
            st.session_state.selected_folder_path = None
        return st.session_state.selected_folder_path
        # Button to select the folder containing subfolders with images or images directly
    selected_folder_path = get_set_selected_folder_path()
    if st.button("Select Folder"):
        selected_folder_path = get_folder_path()
        st.session_state.selected_folder_path = selected_folder_path
    st.write(f"Selected Folder: {selected_folder_path}")  # Display the selected folder path below the button
    if st.button("Find Related Photos") and (uploaded_image is not None) and (selected_folder_path is not None):
            # Check if the specified folder exists
        if not os.path.exists(selected_folder_path):
            st.error("Specified folder does not exist.")
        else:
                # Check if the selected folder contains subfolders
            subfolders = [f for f in os.listdir(selected_folder_path) if os.path.isdir(os.path.join(selected_folder_path, f))]
            if subfolders:  # If subfolders are found, extract features from images in subfolders
                    # Extract features from images in the subfolders
                database_image_features, database_image_mapping = extract_features_from_subfolders(selected_folder_path, encoder_model)
            else:  # If no subfolders are found, directly extract features from images in the selected folder
                folder_path = selected_folder_path
                    # Get a list of image files in the selected folder
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    # Iterate over image files and extract features
                image_features = []
                image_mapping = {}
                for filename in image_files:
                    image_path = os.path.join(folder_path, filename)
                        # Extract features from the image
                    features = feature_extraction(encoder_model, image_path)
                        # Store the features in the list
                    image_features.append(features)
                        # Store the mapping of filename to features
                    image_mapping[image_path] = features
                    # Convert the features list to numpy array
                database_image_features = np.array(image_features)
                    # Assign an empty dictionary to database_image_mapping in this scenario
                database_image_mapping = {}
                # Extract features from the uploaded reference image
            uploaded_image_features = feature_extraction(encoder_model, uploaded_image)
                
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
                            # Retrieve image path based on similar image indices
                        if subfolders:
                            image_path = list(database_image_mapping.keys())[similar_image_indices[idx]]
                        else:
                            image_path = list(image_mapping.keys())[similar_image_indices[idx]]
                        image = Image.open(image_path)
                        cols[j].image(image, caption=f"Related Image {image_path}", use_column_width=True)

    # Add more functionality as needed

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

# ==============================================================================================
# ==============================================================================================
# ==============================================================================================



# import streamlit as st
# import numpy as np
# from PIL import Image
# from sklearn.metrics.pairwise import cosine_similarity
# from tensorflow.keras.models import load_model
# import os
# from keras import backend as K
# from tkinter import Tk, filedialog
# import base64
# import json
# import requests

# # Define global variables
# Usernm = ""

# # Function for feature extraction
# def feature_extraction(model, image_path, layer=4):
#     """
#     Extract features from the specified layer of the autoencoder model.
#     Arguments:
#     model: Trained autoencoder model.
#     image_path: Path to the image to extract features from.
#     layer: Layer index from which to extract features (default: 4).
#     Returns:
#     extracted_features: 1D array of extracted features.
#     """
#     # Load and preprocess the image
#     image = Image.open(image_path)
#     image = image.convert('RGB')
#     image = image.resize((224, 224))
#     image = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension
    
#     # Create a Keras function to extract features from the specified layer
#     encoded = K.function([model.layers[0].input], [model.layers[layer].output])
    
#     # Extract features from the image using the specified layer
#     encoded_array = encoded([image])[0]
    
#     # Pool the features
#     pooled_array = encoded_array.max(axis=(-2, -1))  # Pool over both height and width dimensions
    
#     return pooled_array.flatten()  # Flatten the pooled array to make it 1D

# # Function to extract features from images in the subfolders
# def extract_features_from_subfolders(folder_path, encoder_model):
#     # Initialize lists to store image features and mapping
#     image_features = []
#     image_mapping = {}

#     # Recursively traverse the main folder
#     for root, dirs, files in os.walk(folder_path):
#         # Iterate over the subfolders
#         for subfolder in dirs:
#             subfolder_path = os.path.join(root, subfolder)
#             # Iterate over the files in each subfolder
#             for filename in os.listdir(subfolder_path):
#                 # Check if the file is an image
#                 if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     # Get the full path to the image file
#                     image_path = os.path.join(subfolder_path, filename)
#                     # Extract features from the image
#                     features = feature_extraction(encoder_model, image_path)
#                     # Store the features in the list
#                     image_features.append(features)
#                     # Store the mapping of filename to features
#                     image_mapping[image_path] = features
    
#     # Convert the features list to numpy array
#     image_features = np.array(image_features)
    
#     return image_features, image_mapping

# # Function to find similar images based on features
# def find_similar_images(query_features, database_features):
#     # Ensure query_features and database_features are 2D arrays
#     if query_features.ndim != 2 or database_features.ndim != 2:
#         query_features = np.atleast_2d(query_features)
#         database_features = np.atleast_2d(database_features)

#     # Check the number of features in query and database
#     num_query_features = query_features.shape[1]
#     num_db_features = database_features.shape[1]

#     print("Query Features Shape:", query_features.shape)
#     print("Database Features Shape:", database_features.shape)

#     if num_query_features != num_db_features:
#         raise ValueError("Number of features in query and database do not match.")

#     # Compute cosine similarity
#     similarities = cosine_similarity(query_features, database_features)

#     # Get indices of top similar images
#     similar_image_indices = np.argsort(similarities[0])[::-1]
#     return similar_image_indices

# # Load encoder model
# encoder_model_path = "model/encoder_model.h5"
# encoder_model = load_model(encoder_model_path)

# def main():
#     st.title("Find Related Photos")
#     if 'username' not in st.session_state or not st.session_state.username:
#         login_signup_page()
#     else:
#         image_related_functionality_page()

# def login_signup_page():
#     st.title("Login/Sign up")
#     st.write("Please log in or sign up to continue.")

#     if 'username' not in st.session_state:
#         st.session_state.username = ''
#     if 'useremail' not in st.session_state:
#         st.session_state.useremail = ''

#     choice = st.selectbox('Login/Signup', ['Login', 'Sign up'])
#     email = st.text_input('Email Address', key="email_input")
#     password = st.text_input('Password', type='password', key="password_input")

#     if choice == 'Sign up':
#         username = st.text_input("Enter your unique username")

#         if st.button('Create my account'):
#             user = sign_up_with_email_and_password(email=email, password=password, username=username)

#             if user:
#                 st.success('Account created successfully!')
#                 st.markdown('Please login using your email and password')
#                 st.balloons()
#             else:
#                 st.warning('Signup failed. Please try again.')

#     else:
#         if st.button('Login', on_click=login):
#             pass

#     # JavaScript to blur the input fields after email and password are entered
#     st.markdown(
#         """
#         <script>
#         document.getElementById("email_input").addEventListener("input", function(){
#             this.blur();
#         });
#         document.getElementById("password_input").addEventListener("input", function(){
#             this.blur();
#         });
#         </script>
#         """,
#         unsafe_allow_html=True
#     )


# def login():
#     try:
#         userinfo = sign_in_with_email_and_password(st.session_state.email_input, st.session_state.password_input)
#         if userinfo:
#             st.session_state.username = userinfo['username']
#             st.session_state.useremail = userinfo['email']
#             global Usernm
#             Usernm = userinfo['username']
#             st.session_state.signedout = True
#             st.session_state.signout = True
#         else:
#             st.warning('Login Failed')
#     except Exception as e:
#         st.warning(f'Login failed: {e}')

# def image_related_functionality_page():
#     st.title("Welcome, " + st.session_state.username + "!")
#     # st.write("This is the image related functionality page.")
#     # st.write("Implement your image related functionality here.")

#     # Function to get the folder path using Tkinter
    
#     def get_folder_path():
#         root = Tk()
#         root.withdraw()
#         folder_path = filedialog.askdirectory()
#         root.destroy()
#         return folder_path

#     # Function to get the drive path using Tkinter
#     def get_drive_path():
#         root = Tk()
#         root.withdraw()
#         drive_path = filedialog.askdirectory()
#         root.destroy()
#         return drive_path
    
#     # Function to get or set the selected folder path using session state
#     def get_set_selected_folder_path():
#         if 'selected_folder_path' not in st.session_state:
#             st.session_state.selected_folder_path = None
#         return st.session_state.selected_folder_path

#     # Button to select the folder containing subfolders with images or images directly
#     selected_folder_path = get_set_selected_folder_path()
#     if st.button("Select Folder"):
#         selected_folder_path = get_folder_path()
#         st.session_state.selected_folder_path = selected_folder_path
#     st.write(f"Selected Folder: {selected_folder_path}")  # Display the selected folder path below the button

#     # Button to select a drive containing images
#     if st.button("Select Drive"):
#         selected_drive_path = get_drive_path()
#     st.write(f"Selected Drive: {selected_drive_path}")  # Display the selected drive path

#     # File uploader for the reference image
#     uploaded_image = st.file_uploader("Upload a reference image", type=["jpg", "png", "jpeg"])
    
#     # Display the uploaded image if it exists
#     if uploaded_image is not None:
#         try:
#             uploaded_image_display = Image.open(uploaded_image)
#             st.image(uploaded_image_display, caption='Uploaded Image', use_column_width=True)
#         except Exception as e:
#             st.error(f"Error: {e}")
#             st.error("Unable to open the uploaded image. Please make sure it's a valid image file.")

#     # Button to find related photos
#     if st.button("Find Related Photos") and (uploaded_image is not None) and (selected_folder_path is not None):
#         # Check if the specified folder exists
#         if not os.path.exists(selected_folder_path):
#             st.error("Specified folder does not exist.")
#         else:
#             # Check if the selected folder contains subfolders
#             subfolders = [f for f in os.listdir(selected_folder_path) if os.path.isdir(os.path.join(selected_folder_path, f))]
#             if subfolders:  # If subfolders are found, extract features from images in subfolders
#                 # Extract features from images in the subfolders
#                 database_image_features, database_image_mapping = extract_features_from_subfolders(selected_folder_path, encoder_model)
#             else:  # If no subfolders are found, directly extract features from images in the selected folder
#                 folder_path = selected_folder_path
#                 # Get a list of image files in the selected folder
#                 image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#                 # Iterate over image files and extract features
#                 image_features = []
#                 image_mapping = {}
#                 for filename in image_files:
#                     image_path = os.path.join(folder_path, filename)
#                     # Extract features from the image
#                     features = feature_extraction(encoder_model, image_path)
#                     # Store the features in the list
#                     image_features.append(features)
#                     # Store the mapping of filename to features
#                     image_mapping[image_path] = features
#                 # Convert the features list to numpy array
#                 database_image_features = np.array(image_features)
#                 # Assign an empty dictionary to database_image_mapping in this scenario
#                 database_image_mapping = {}
            
#             # Extract features from the uploaded reference image
#             uploaded_image_features = feature_extraction(encoder_model, uploaded_image)
                
#             # Find similar images based on features     
#             similar_image_indices = find_similar_images(uploaded_image_features, database_image_features)
#             # Display similar images in a grid view with three columns
#             st.subheader("Related Photos:")
#             num_columns = 3  # Number of columns in the grid
#             num_images = len(similar_image_indices)
#             num_rows = (num_images + num_columns - 1) // num_columns  # Calculate number of rows needed
#             # Create a grid layout
#             for i in range(num_rows):
#                 cols = st.columns(num_columns)  # Create columns for each row
#                 for j in range(num_columns):
#                     idx = i * num_columns + j
#                     if idx < num_images:
#                         # Retrieve image path based on similar image indices
#                         if subfolders:
#                             image_path = list(database_image_mapping.keys())[similar_image_indices[idx]]
#                         else:
#                             image_path = list(image_mapping.keys())[similar_image_indices[idx]]
#                         image = Image.open(image_path)
#                         cols[j].image(image, caption=f"Related Image {image_path}", use_column_width=True)

# def sign_up_with_email_and_password(email, password, username=None, return_secure_token=True):
#     try:
#         rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
#         payload = {
#             "email": email,
#             "password": password,
#             "returnSecureToken": return_secure_token
#         }
#         if username:
#             payload["displayName"] = username 
#         payload = json.dumps(payload)
#         r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
#         try:
#             return r.json()['email']
#         except:
#             st.warning(r.json())
#     except Exception as e:
#         st.warning(f'Signup failed: {e}')

# def sign_in_with_email_and_password(email=None, password=None, return_secure_token=True):
#     rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

#     try:
#         payload = {
#             "returnSecureToken": return_secure_token
#         }
#         if email:
#             payload["email"] = email
#         if password:
#             payload["password"] = password
#         payload = json.dumps(payload)
#         print('payload sigin',payload)
#         r = requests.post(rest_api_url, params={"key": "AIzaSyApr-etDzcGcsVcmaw7R7rPxx3A09as7uw"}, data=payload)
#         try:
#             data = r.json()
#             user_info = {
#                 'email': data['email'],
#                 'username': data.get('displayName')  # Retrieve username if available
#             }
#             return user_info
#         except:
#             st.warning(data)
#     except Exception as e:
#         st.warning(f'Signin failed: {e}')

# if __name__ == "__main__":
#     main()
