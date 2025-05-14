import streamlit as st
import pandas as pd
import numpy as np

menu_items = ['Basic', 'Chart', 'AI 응용']
with st.sidebar:
  menu = st.selectbox(label='Menu', options=menu_items)
  st.divider()
  st.text('강의자료')
  with open('AI응용소프트웨어 개발.pdf', 'rb') as f:
    st.download_button('Download PDF', f, file_name='note.pdf')

st.title("DCU 체험 100년 대학")

if menu == 'Basic':
  def func(r):
    a = 3.14 * r ** 2
    return a

  st.header('원 면적 구하기')

  radius = st.number_input(label='Radius') 

  if radius:
    area = func(radius)
    st.write(f'원 면적 = {area}')
  with st.expander('Source Code:'):
    st.code('''
    # Source code
    import streamlit as st

    def func(r):
      a = 3.14 * r ** 2
      return a

    st.header('원 면적 구하기')

    radius = st.number_input(label='Radius') 

    if radius:
      area = func(radius)
      st.write(f'원 면적 = {area}')
    ''', 'python')

if menu == 'Chart':
  st.write("Streamlit supports a wide range of data visualizations, including [Plotly, Altair, and Bokeh charts](https://docs.streamlit.io/develop/api-reference/charts). 📊 And with over 20 input widgets, you can easily make your data interactive!")

  all_users = ["Alice", "Bob", "Charly"]
  with st.container(border=True):
      users = st.multiselect("Users", all_users, default=all_users)
      rolling_average = st.toggle("Rolling average")

  np.random.seed(42)
  data = pd.DataFrame(np.random.randn(20, len(users)), columns=users)
  if rolling_average:
      data = data.rolling(7).mean().dropna()

  tab1, tab2 = st.tabs(["Chart", "Dataframe"])
  tab1.line_chart(data, height=250)
  tab2.dataframe(data, height=250, use_container_width=True)
  
if menu == 'AI 응용':
  import streamlit as st
  import mediapipe as mp
  from mediapipe.framework.formats import landmark_pb2
  import numpy as np
  from PIL import Image

  BaseOptions = mp.tasks.BaseOptions
  GestureRecognizer = mp.tasks.vision.GestureRecognizer
  GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  # Create a gesture recognizer instance with the image mode:
  options = GestureRecognizerOptions(
      base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
      running_mode=VisionRunningMode.IMAGE)

  image = st.camera_input(label='Take a picture')
  # image = st.file_uploader(label='Take a picture')
 
  if image is not None:
    image = Image.open(image)
    image = np.array(image)

    with GestureRecognizer.create_from_options(options) as recognizer:
    # The detector is initialized. Use it here.
    # ...
      
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
      annotated_image = image.copy()

      gesture_recognition_result = recognizer.recognize(mp_image)
      if len(gesture_recognition_result.gestures) > 0:
        top_gesture = gesture_recognition_result.gestures[0][0]
        multi_hand_landmarks = gesture_recognition_result.hand_landmarks

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        for hand_landmarks in multi_hand_landmarks:
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
          ])

          mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
          
        st.image(annotated_image)
        st.markdown(f'#### {top_gesture.category_name} {top_gesture.score:.2f}')
      else:
        st.image(annotated_image)
        st.markdown(f'No Gesture')

  with st.expander('File Download'):
    st.markdown('''<a href="gesture_recognizer.task" download>Task File</a> 
    ''', unsafe_allow_html=True)
  with st.expander("Source Code"):
    st.code('''
import streamlit as st
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from PIL import Image

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the image mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE)

image = st.camera_input(label='Take a picture')
# image = st.file_uploader(label='Take a picture')

if image is not None:
  image = Image.open(image)
  image = np.array(image)

  with GestureRecognizer.create_from_options(options) as recognizer:
  # The detector is initialized. Use it here.
  # ...
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    annotated_image = image.copy()

    gesture_recognition_result = recognizer.recognize(mp_image)
    if len(gesture_recognition_result.gestures) > 0:
      top_gesture = gesture_recognition_result.gestures[0][0]
      multi_hand_landmarks = gesture_recognition_result.hand_landmarks

      mp_hands = mp.solutions.hands
      mp_drawing = mp.solutions.drawing_utils
      mp_drawing_styles = mp.solutions.drawing_styles

      for hand_landmarks in multi_hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
        
      st.image(annotated_image)
      st.markdown(f'#### {top_gesture.category_name} {top_gesture.score:.2f}')
    else:
      st.image(annotated_image)
      st.markdown(f'No Gesture')
    ''', 'python')
    
