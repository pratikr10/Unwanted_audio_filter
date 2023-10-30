import socket
import pyaudio
import wave

# Create a socket connection to the server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('192.168.43.3', 12345)  # Replace 'server_ip' with your server's IP address
client_socket.connect(server_address)

# Initialize the audio stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

# Record audio and send it to the server
while True:
    audio_data = stream.read(1024)
    client_socket.sendall(audio_data)

# Close the audio stream and socket when done
stream.stop_stream()
stream.close()
client_socket.close()