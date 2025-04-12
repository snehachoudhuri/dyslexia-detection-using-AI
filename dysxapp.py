import numpy as np
import pickle
import streamlit as st

if 'landing_page' not in st.session_state:
    st.session_state.landing_page = True

if st.session_state.landing_page:
    with st.container():
        st.markdown('<div class="landing-container">', unsafe_allow_html=True)
        st.markdown("<h1>Welcome to the AI-Powered Medical Diagnosis System</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p>Our advanced AI platform assists in detecting Dyslexic conditions.<br><br>"
            "<strong>Disclaimer:</strong> This tool is for informational purposes only and is not a substitute for professional medical advice.</p>",
            unsafe_allow_html=True
        )
        if st.button("Enter"):
            st.session_state.landing_page = False
            st.rerun()  # Use st.rerun() if you're on Streamlit >=1.18
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop() 

# Load model
with open('dysx_model.sav', 'rb') as file:
    dysx_model = pickle.load(file)

def main():
    st.title('Dyslexia Predictor')

    # Input fields
    Language_vocab = st.number_input('Level of Language Vocabulary (range: 0(best) to 1(worst))', min_value=0.0, value=0.0)
    Memory = st.number_input('Memory Level (range: 0(best) to 1(worst))', min_value=0.0, value=0.0)
    Speed = st.number_input('Speed Level (range: 0(best) to 1(worst))', min_value=0.0, value=0.0)
    Visual_discrimination = st.number_input('Visual Discrimination Score (range: 0(best) to 1(worst))', min_value=0.0, value=0.0)
    Audio_Discrimination = st.number_input('Audio Discrimination Score (range: 0(best) to 1(worst))', min_value=0.0, value=0.0)
    Survey_Score = st.number_input('Survey Score according to questionnaire (range: 0(best) to 1(worst))', min_value=0.0, value=0.0)


    diagnosis = ''

    if st.button('Test Result'):
        input_data = np.array([Language_vocab, Memory, Speed, Visual_discrimination, Audio_Discrimination, Survey_Score])
        input_data_reshaped = input_data.reshape(1, -1)

        prediction = dysx_model.predict(input_data_reshaped)

        if prediction[0] == 0:
            diagnosis = 'The individual is likely NOT Dyslexic.'
        elif prediction[0] == 1:
            diagnosis = 'The individual shows signs of Dyslexia. Please consult a specialist.'
        elif prediction[0] == 2:
            diagnosis = 'The individual may have SEVERE Dyslexia. Urgently consult a specialist.'
        else:
            diagnosis = 'Unexpected result from prediction.'

        st.success(diagnosis)

if __name__ == '__main__':
    main()
