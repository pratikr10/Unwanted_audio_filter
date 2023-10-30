import socket
import pyaudio
import re
import string
import nltk
import speech_recognition as sr
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.models import load_model
import pickle
import winsound
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pydub import AudioSegment
import io

# Initialize the Snowball stemmer
stemmer = SnowballStemmer("english")
nltk.download('stopwords')

# Define audio capture parameters
RECORD_SECONDS = 5  # Duration of audio capture in seconds

# Load your custom model (replace 'your_model_filename' with the actual filename)
loaded_model = load_model("hate&abusive_model.h5")

# Load your custom tokenizer (replace 'your_tokenizer_filename' with the actual filename)
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# Set up the server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '192.168.159.153'  # Listen on all available network interfaces
port = 12345
server_socket.bind((host, port))
server_socket.listen(1)
print("Server listening on {}:{}".format(host, port))

# Initialize PyAudio for audio playback
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, output=True)

# Function to receive and play audio with offensive word detection
def receive_audio(conn):
    while True:
        audio_data = conn.recv(1024)
        if not audio_data:
            break
        # Perform offensive word detection on transcribed text
        transcribed_text = recognize_audio(audio_data)
        process_offensive_words(transcribed_text)
        stream.write(audio_data)

# Function to recognize transcribed text from audio data
def recognize_audio(audio_data):
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(io.BytesIO(audio_data)) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except sr.WaitTimeoutError:
        print("Timeout: No speech detected after 5 seconds.")
    except sr.UnknownValueError:
        print("Google Web Speech Recognition could not understand audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech Recognition service; {e}")

    return ""

# Function to process offensive words and play beep sound
def process_offensive_words(text):
    # Clean and preprocess the transcribed text (same cleaning code as before)
    cleaned_text = clean_text(text)
    words = cleaned_text.split()

    # Define a threshold for offensive word detection
    offensive_threshold = 0.5

    # Process each word individually
    for word in words:
        # Tokenize the word using the custom tokenizer
        seq = loaded_tokenizer.texts_to_sequences([word])
        padded = pad_sequences(seq, maxlen=300)

        # Classify the word using the custom model
        prediction = loaded_model.predict(padded)

        if prediction >= offensive_threshold:
            # Load a short beep sound
            winsound.Beep(1000, 500)
def clean_text(text):
    # Remove punctuation and convert to lowercase
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    # Tokenize the text
    words = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # Join the cleaned words back into a single string
    cleaned_text = " ".join(words)

    return cleaned_text
# Accept incoming connections and receive audio
while True:
    client_conn, client_addr = server_socket.accept()
    print("Connection from: {}".format(client_addr))
    receive_audio(client_conn)
