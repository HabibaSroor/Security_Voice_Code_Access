from UI.HABIBA_UI import Ui_MainWindow
import feature_extraction_functions
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
import matplotlib.pyplot as plt
import sys
import wave
import pyaudio
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QWidget, QGridLayout, QGraphicsScene, QGraphicsView
import numpy as np
from scipy.io.wavfile import read
import pickle

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.fs = 44100
        self.seconds = 3
        self.filename = "new.wav"
        self.p = pyaudio.PyAudio()
        self.recording = False

        self.lock_urls = ['UI/padlock-2.png', 'UI/padlock.png']
        self.person_strings = ["aya", "luna", "habiba", "other", "ayaSalah", "nouran"]
        self.person_labels_list = {"aya":self.ui.aya_eyad_lbl, "luna":self.ui.luna_lbl, "habiba":self.ui.habiba_lbl, 
                                       "other":self.ui.other_lbl, "ayaSalah":self.ui.aya_salah_lbl, "nouran":self.ui.nouran_lbl}
        self.password_strings = ["open the door", "grant me access", "open middel gate"]
        self.password_labels = {"open the door":self.ui.OTD_lbl, "grant me access":self.ui.GMA_lbl, "open middel gate":self.ui.OMG_lbl}

        self.connect_signals_slots()

    def connect_signals_slots(self):
        self.ui.record_btn.clicked.connect(self.toggle_recording)
        self.ui.pushButton_ok.clicked.connect(self.choose_peoples)

    def toggle_recording(self):
        """
        Toggles between starting and stopping the recording.

        Checks if recording is in progress. If not recording, starts recording by
        calling the start_recording function. If recording, stops recording by
        calling the stop_recording function.
        """
        # Check if recording is in progress
        if not self.recording:
            # Start recording
            self.start_recording()
        else:
            # Stop recording
            self.stop_recording()

    def start_recording(self):
        """
        Initiates the recording process.

        Sets the recording flag to True, updates the UI with a new icon for the
        record button, initializes PyAudio, and sets up the audio stream for capturing
        audio data using the specified sample format, channels, sampling rate, chunk size,
        and callback function.

        Parameters:
        - self.sample_format (int): The sample format for recording (e.g., pyaudio.paInt16).
        - self.channels (int): The number of audio channels (1 for mono, 2 for stereo, etc.).
        - self.fs (int): The sampling rate (frames per second).
        - self.chunk (int): The number of frames per buffer.
        - self.callback (callable): The callback function to process incoming audio data.
        """
        self.recording = True
        # remov eurl
        new_icon = QIcon('UI/microphone copy.png')
        self.ui.record_btn.setIcon(new_icon)

        # Initialize PyAudio and set up the audio stream
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.stream = self.p.open(format=self.sample_format,
                                      channels=self.channels,
                                      rate=self.fs,
                                      frames_per_buffer=self.chunk,
                                      input=True,
                                      stream_callback=self.callback)

    def callback(self, in_data,frame_count, time_info, status):
        # Audio callback function - called by the stream whenever there's new audio data
        self.frames.append(in_data)
        return None, pyaudio.paContinue



    def stop_recording(self):
        """
        Stops the recording process.

        Sets the recording flag to False, stops and closes the audio stream, terminates
        the PyAudio instance, updates the UI with a new icon for the record button, and
        writes the recorded audio frames to a WAV file. Additionally, it invokes functions
        to generate a spectrogram and perform predictions on the recorded audio.

        """
        self.recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        new_icon = QIcon('UI/microphone-black-shape.png')
        self.ui.record_btn.setIcon(new_icon)

        # Write the recorded audio frames to a WAV file
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        self.to_spectrogram()
        self.predict_words(test_audio = "new.wav")
        self.predict_person(test_audio = "new.wav")

        # self.ui.label_3.setText("Recording completed. Press 'Record' to record again.")
        self.ui.record_btn.setEnabled(True)
    


    def to_spectrogram(self):
        """
        Converts recorded audio to a spectrogram and displays it.

        Reads the recorded audio from the 'new.wav' file, extracts the sample rate
        and audio data using the `scipy.io.wavfile.read` function, and then displays
        the spectrogram using the `display_spectrogram` function.

        Parameters:
        - "new.wav" (str): File name of the recorded audio.
        """
        sample_rate, audio_data = read("new.wav")
        self.display_spectrogram(audio_data, sample_rate)



    def display_spectrogram(self, audio_data, Fs):
        """
        Displays the spectrogram of the provided audio data in the UI.

        Utilizes Matplotlib to generate the spectrogram plot, saves it as an image,
        resizes the image using QPixmap, and sets the resized image to a QLabel in the UI.

        Parameters:
        - audio_data (numpy.ndarray): Audio data for which the spectrogram is to be generated.
        - Fs (int): Sampling rate of the audio data.
        """
        plt.specgram(audio_data, NFFT=1024, Fs = Fs, noverlap=512, cmap='viridis')
        # # Hide the numbers on the axes
        # plt.gca().axes.get_xaxis().set_visible(False)
        # plt.gca().axes.get_yaxis().set_visible(False)
        # plt.axis('off')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.savefig('spectrogram.png',  transparent=True)  # Save the spectrogram image
        plt.close()

        # Display the spectrogram image in the UI
        pixmap = QPixmap('spectrogram.png')
        pixmap_resized = pixmap.scaled(500, 250)
        # Set the pixmap to the label
        self.ui.spectro_lbl.setPixmap(pixmap_resized)

    def predict_person(self, test_audio):
        """
        Predicts the person from the given test audio and updates the UI.

        Parameters:
        - test_audio (str): File path of the test audio.
        """
        # Extract features from the test audio
        test_features = feature_extraction_functions.features_extractor(test_audio)

        # Load the model for predicting persons
        model = pickle.load(open("Models/model2.pkl", "rb"))

        # Use predict_proba to get class probabilities
        class_probabilities = model.predict_proba(test_features)[0]
        
        max_probability_label = model.classes_[np.argmax(class_probabilities)]
       
        # Convert probabilities to percentages
        _probabilities = {label: (probability) * 100 for label, probability in zip(model.classes_, class_probabilities)}
        
        self.modify_labels(_probabilities)

        # Update the UI with the lock image based on the predicted person
        pixmap = QPixmap(self.lock_urls[max_probability_label in self.chosen_people])
        self.ui.label.setPixmap(pixmap)

    def modify_labels(self, percentages):
        """
        Modifies labels in the UI based on predicted probabilities.

        Parameters:
        - percentages (dict): Dictionary of predicted probabilities for each person.
        """
        for person_name in self.person_strings:
            self.person_labels_list[person_name].setText(f"{percentages[person_name]:.2f}%")




    def predict_words(self, test_audio):
        """
        Predicts the words from the given test audio and updates the UI.

        Parameters:
        - test_audio (str): File path of the test audio.
        """
        # Extract features from the test audio
        test_features = feature_extraction_functions.features_extractor(test_audio)

        # Load the model for predicting words
        password_model = pickle.load(open("Models/password_model3.pkl", "rb"))
        
        # Use predict_proba to get class probabilities
        class_probabilities = password_model.predict_proba(test_features)[0]
        max_probability = class_probabilities[np.argmax(class_probabilities)]

        # Update the UI with the lock image based on the predicted person
        pixmap = QPixmap(self.lock_urls[not (max_probability >= 0.499)])
        self.ui.label.setPixmap(pixmap)

        # Convert probabilities to percentages
        _probabilities = {label: (probability) * 100 for label, probability in zip(password_model.classes_, class_probabilities)}

        self.modify_labels_password(_probabilities)
       


    def modify_labels_password (self, percentages):
        """
        Modifies labels in the UI based on predicted probabilities for passwords.

        Parameters:
        - percentages (dict): Dictionary of predicted probabilities for each password.
        """

        for password in self.password_strings:
            self.password_labels[password].setText(f"{percentages[password]:.2f}%")
        


    def choose_peoples(self):
        """
        Retrieves the selected items from the listWidget and updates the chosen_people list.

        The function is designed to be connected to a signal in response to user interaction
        (ok button click, list item selection).

        Updates:
        - self.chosen_people: List of selected people.
        """

        selected_items = self.ui.listWidget.selectedItems()

        if selected_items:
            self.chosen_people = [item.text() for item in selected_items]



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.setWindowTitle("Security Voice-code Access")
    mainWindow.show()
    sys.exit(app.exec())