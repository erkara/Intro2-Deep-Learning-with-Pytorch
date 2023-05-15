import shutil
import torch
import imageio
import torch.nn.functional as F
import numpy as np
import audioread
import pytube
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sn
import librosa
import librosa.display
from IPython.display import clear_output, display, YouTubeVideo
import matplotlib.pyplot as plt
import csv
import pandas as pd
from pydub import AudioSegment
from yt_dlp.utils import DownloadError
from yt_dlp import YoutubeDL
import warnings
warnings.filterwarnings("ignore")
SAMPLE_RATE = 48000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================CREATE AUDIO DATASET FROM YOUTUBE URLS=====================

# Function to download audio from a YouTube link
def download_audio(video_url, audio_name=None):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
        'outtmpl': 'temp_audio.%(ext)s',
        'quiet': True
    }
    if audio_name is not None:
        ydl_opts['outtmpl'] = f'{audio_name}.%(ext)s'

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except DownloadError as e:
        print(f"Error downloading audio from URL: {video_url}\nError details: {e}")

import math


def _split_audio(filename, party, person, max_duration=None, split_length=1, min_last_chunk_duration=5, ):
    audio = AudioSegment.from_wav(filename)
    audio_length = len(audio) // 1000

    if max_duration is not None:
        audio_length = min(audio_length, int(max_duration * 60))

    num_chunks = audio_length // split_length

    last_chunk_duration = audio_length % split_length

    chunks = []

    for i in range(num_chunks):
        chunk_start_time = i * split_length * 1000
        chunk_end_time = chunk_start_time + split_length * 1000
        chunk_filename = f"{party}_{person}_{i}.wav"
        chunk = audio[chunk_start_time:chunk_end_time]
        chunk.export(chunk_filename, format="wav")
        chunks.append(chunk_filename)

    if last_chunk_duration >= min_last_chunk_duration:
        last_chunk_start_time = num_chunks * split_length * 1000
        last_chunk_end_time = last_chunk_start_time + last_chunk_duration * 1000
        last_chunk_filename = f"{party}_{person}_{num_chunks}.wav"
        last_chunk = audio[last_chunk_start_time:last_chunk_end_time]
        last_chunk.export(last_chunk_filename, format="wav")
        chunks.append(last_chunk_filename)

    return chunks


# Function to move audio chunks to label folders and return their paths
def _move_chunks_to_folders(chunks, label, party, person, root_dir):
    label_dir = os.path.join(root_dir, label)

    if not os.path.exists(label_dir):
        os.mkdir(label_dir)  # Create new folder

    person_chunks = []
    for existing_chunk in os.listdir(label_dir):
        if existing_chunk.startswith(f"{party}_{person}_") and existing_chunk.endswith(".wav"):
            person_chunks.append(int(existing_chunk.split("_")[-1].split(".")[0]))

    # Find the next available chunk ID
    if person_chunks:
        chunk_id = max(person_chunks) + 1
    else:
        chunk_id = 0

    paths = []
    for chunk in chunks:
        dest_path = os.path.join(label_dir, f"{party}_{person}_{chunk_id}.wav")
        os.rename(chunk, dest_path)
        paths.append(dest_path)
        chunk_id += 1

    return paths


def process_csv(dataframe, class_to_idx, root_dir="speech_data", max_duration=1, split_length=10):
    # Create the root directory if it does not exist, otherwise remove the existing one
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    else:
        shutil.rmtree(root_dir)  # Remove existing root_dir
        os.mkdir(root_dir)  # Recreate root_dir

    data = []

    num_rows = dataframe.shape[0]
    for i, row in dataframe.iterrows():
        label, party, person, speech_url = row['label'], row['party'], row['person'], row['speech_url']
        download_audio(speech_url)

        if os.path.exists('temp_audio.wav'):
            chunks = _split_audio('temp_audio.wav', party, person, max_duration, split_length)
            paths = _move_chunks_to_folders(chunks, label, party, person, root_dir)

            for path in paths:
                data.append({"label": label, "path": path, "id": class_to_idx[label]})

            # Remove the temporary audio file
            os.remove('temp_audio.wav')
        else:
            print(f"Failed to download audio for {speech_url} at {label}-{party}-{person}")

        print(f"processed {i}/{num_rows} samples")

    # Create the DataFrame
    df = pd.DataFrame(data, columns=["label", "path", "id"])
    # Save the DataFrame to a CSV file
    df.to_csv("speech_dataframe.csv", index=False)
    print("cleaned dataframe saved to speech_dataframe.csv")


# ==========================DATA LOADING=======================================================


def load_data(data, duration, offset):
    signals = []
    for i, file_path in enumerate(data.path):
        try:
            audio, sample_rate = librosa.load(file_path, duration=duration, offset=offset, sr=SAMPLE_RATE)
            signal = np.zeros((int(duration * SAMPLE_RATE),))
            signal[:len(audio)] = audio
            signals.append(signal)
            print(f"\rloading {i + 1}/{len(data)} audio files", end="")
        except (librosa.util.exceptions.ParameterError, audioread.exceptions.DecodeError):
            print(f"\nbroken audio at: {file_path} and dropping row {i}")
            data.drop(i, inplace=True)
            data.reset_index(drop=True, inplace=True)
            continue

    signals = np.stack(signals, axis=0)
    return data, signals


def split_data(data, signals):
    """
    train-valid-test-->80:10:10
    """
    X_train, X_test, Y_train, Y_test = train_test_split(signals, data['id'].values, test_size=0.2,
                                                        random_state=42, stratify=data['id'])

    return X_train, Y_train, X_test, Y_test


def addAWGN(signal, num_bits=16, augmented_num=2, snr_low=15, snr_high=30):
    signal_len = len(signal)
    # Generate White Gaussian noise
    noise = np.random.normal(size=(augmented_num, signal_len))
    # Normalize signal and noise
    norm_constant = 2.0 ** (num_bits - 1)
    signal_norm = signal / norm_constant
    noise_norm = noise / norm_constant
    # Compute signal and noise power
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    # Random SNR: Uniform [15, 30] in dB
    target_snr = np.random.randint(snr_low, snr_high)
    # Compute K (covariance matrix) for each noise 
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, augmented_num)) * K
    # Generate noisy signal
    return signal + K.T * noise


def addAWGN_train(data, X_train, Y_train):
    aug_signals = []
    aug_labels = []
    for i in range(X_train.shape[0]):
        signal = X_train[i, :]
        augmented_signals = addAWGN(signal)
        for j in range(augmented_signals.shape[0]):
            aug_labels.append(data.loc[i, "id"])
            aug_signals.append(augmented_signals[j, :])
            data = data.append(data.iloc[i], ignore_index=True)
        print(f"\radding AWGN noise {i + 1}/{X_train.shape[0]} files", end="")
    aug_signals = np.stack(aug_signals, axis=0)
    X_train = np.concatenate([X_train, aug_signals], axis=0)
    aug_labels = np.stack(aug_labels, axis=0)
    Y_train = np.concatenate([Y_train, aug_labels])

    return X_train, Y_train


def getMELspectrogram(audio, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=SAMPLE_RATE,
                                              n_fft=1024,
                                              win_length=512,
                                              window='hamming',
                                              hop_length=256,
                                              n_mels=n_mels,
                                              fmax=SAMPLE_RATE / 2
                                              )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


# this is correct
def compute_mel_spectrogram(X, n_mels=128):
    mel_data = []
    print(f"\ncalculating mel spectrograms for {X.shape[0]} files")
    for i in range(X.shape[0]):
        mel_spectrogram = getMELspectrogram(X[i, :], n_mels)
        mel_data.append(mel_spectrogram)
        print(f"\rprocessed {i + 1}/{X.shape[0]} files", end='')
    print('')
    return np.stack(mel_data, axis=0)


def scale_data(X_train, X_test):
    """
    X-->(b,h,w)
    """
    scaler = StandardScaler()
    # (b,h,w)-->(b, c=1, h, w)
    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)

    b, c, h, w = X_train.shape
    # (b, c, h, w) --> (b, c * h * w)
    X_train = np.reshape(X_train, newshape=(b, -1))

    # get the mean and standard deviation of the training data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # fit the scaler on the training data only and transform all sets
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(np.reshape(X_test, newshape=(X_test.shape[0], -1)))

    # (b, c * h * w)--> (b, c, h, w)
    X_train = np.reshape(X_train, newshape=(b, c, h, w))
    X_test = np.reshape(X_test, newshape=(X_test.shape[0], c, h, w))

    print(f"\nscaled the data")

    return X_train, X_test, mean, std


def process_full_data(data, duration, offset, n_mels=128):
    # load the dataset
    data, signals = load_data(data, duration, offset)
    print('\ndata loaded...')

    # split the dataset
    X_train, Y_train, X_test, Y_test = split_data(data, signals)
    print(f"\nafter split X_train: {X_train.shape}\n")

    # add AWGN to train
    X_train, Y_train = addAWGN_train(data, X_train, Y_train)
    print(f"\nafter noise augmented X_train: {X_train.shape}\n")

    # calculate mel-specs
    X_train = compute_mel_spectrogram(X_train, n_mels)
    X_test = compute_mel_spectrogram(X_test, n_mels)

    print(f"\nafter mel-spec X_train: {X_train.shape}\n")

    # scale-data
    X_train, X_test, mean, std = scale_data(X_train, X_test)

    print("finished processing dataset\n")
    print(f"X_train:{X_train.shape} Y_train:{Y_train.shape}\n")
    print(f"X_test:{X_test.shape} Y_test:{Y_test.shape}\n")

    return X_train, Y_train, X_test, Y_test, (mean, std)


def process_audio(audio, mean, std):
    # Convert the audio data into a mel spectrogram in decibels
    mel_spec_db = getMELspectrogram(audio)

    # Get the height and width of the mel spectrogram
    h, w = mel_spec_db.shape[0], mel_spec_db.shape[1]

    # Flatten the mel spectrogram into a 1D array
    mel_spec_flatten = np.reshape(mel_spec_db, newshape=(-1,))

    # Normalize the flattened mel spectrogram using the mean and standard deviation
    mel_spec_scaled = (mel_spec_flatten - mean) / std

    # Reshape the normalized mel spectrogram back to its original 2D shape
    mel_spec_scaled = np.reshape(mel_spec_scaled, newshape=(h, w))

    # Add two additional dimensions to the mel spectrogram to match the expected input shape of the model
    x_processed = np.reshape(mel_spec_scaled, newshape=(1, 1, h, w))

    # Return the processed data
    return x_processed


#

# # Define a function to get the model's prediction for the given audio data
def get_prediction(model, x_processed, idx_to_class):
    # Move the model to the device and set train mode
    model.to(device)
    model.train()

    # Convert the preprocessed audio data into a PyTorch tensor and move it to the device
    x_tensor = torch.tensor(x_processed).float().to(device)

    # Get the model's output for the input tensor
    output = model(x_tensor)

    # Calculate the softmax probabilities for the output tensor
    probs = F.softmax(output)

    # Get the top `num_classes` probabilities and their corresponding indices
    top_probabilities, top_indices = probs.topk(len(idx_to_class))

    # Convert the probabilities and indices to numpy arrays and extract the top values
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_probabilities = [round(elem, 2) for elem in top_probabilities]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]

    # Convert the top indices to class names using the `idx_to_class` dictionary
    top_classes = [idx_to_class[index] for index in top_indices]

    # Return the top probabilities and their corresponding class names
    return top_probabilities, top_classes


#
def download_youtube_audio(video_url, audio_name='sample_audio.vaw'):
    yt = pytube.YouTube(video_url)

    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download()

    # get the base name of the file (without the directory path)
    basename = os.path.basename(out_file)

    # create a new name for the file by replacing the file extension with '.wav'
    new_file = os.path.join(os.path.dirname(out_file), audio_name)

    if os.path.exists(new_file):
        os.remove(new_file)

    # rename
    os.rename(basename, new_file)

    print(f"{basename} downloaded and saved as {audio_name}")


def display_video(video_url):
    video_id = video_url.split("/")[-1]
    # load the YouTube video
    # Set the width and height of the video player
    video_width = 450
    video_height = 250

    # Embed the video in the Jupyter notebook
    video = YouTubeVideo(video_id, width=video_width, height=video_height)

    # display the video and the dynamic pie chart
    display(video)


# Define a function to predict the class probabilities for an audio clip
def predict_on_audio(model, audio_path, mean, std, idx_to_class,
                     window_size=3, offset=10, hop_size=1, save_anime=False):
    # load audio file
    audio_path = audio_path + '.wav'
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, offset=offset, mono=True)

    # calculate the number of frames
    audio_length = len(audio)
    frame_size = window_size * sr
    hop_length = hop_size * sr
    num_frames = int(np.ceil((audio_length - frame_size) / hop_length) + 1)

    # initialize predictions array
    predictions = np.zeros((num_frames, len(idx_to_class)))

    # initialize dictionary to store count and sum of probabilities for each class
    class_prob_sum = {class_id: 0.0 for class_id in idx_to_class.values()}
    class_prob_count = {class_id: 0 for class_id in idx_to_class.values()}

    # If the user wants to save an animation of the predictions, create a directory to store the PNG files
    if save_anime:
        image_filenames = []
        temp_dir = "temp_dir"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
        else:
            os.makedirs(temp_dir)

    # Process audio in overlapping frames
    for i, start_sample in enumerate(range(0, audio_length - frame_size + 1, hop_length)):
        end_sample = start_sample + frame_size
        audio_frame = audio[start_sample:end_sample]

        # Preprocess the audio chunk
        processed_chunk = process_audio(audio_frame, mean, std)

        # Get the top predicted classes and their probabilities for the audio chunk
        top_probabilities, top_classes = get_prediction(model, processed_chunk, idx_to_class)

        # Update the running confidance average for each class
        for class_name, class_prob in zip(top_classes, top_probabilities):
            class_prob_sum[class_name] += class_prob
            class_prob_count[class_name] += 1
        class_prob_avg = {class_name: round(class_prob_sum[class_name] / class_prob_count[class_name], 3)
                          for class_name in idx_to_class.values()}

        # pie chart showing the class probabilities for the audio chunk
        plt.clf()
        plt.pie(class_prob_avg.values(), labels=class_prob_avg.keys(), autopct='%1.0f%%',
                textprops={'fontsize': 16})

        # save an animation of the predictions, save the current pie chart as a PNG file
        if save_anime:
            image_name = f"{temp_dir}/frame{i}.png"
            plt.savefig(image_name)
            image_filenames.append(image_name)

        # Display the current pie chart
        display(plt.gcf())
        clear_output(wait=True)
        time.sleep(0.5)

    # save an animation of the predictions, create an MP4 file from the PNG files and delete the directory
    if save_anime:
        images = [imageio.imread(filename) for filename in image_filenames]
        imageio.mimsave('pie_chart_animation.mp4', images, fps=3)
        shutil.rmtree(temp_dir)

    # Print a message indicating that the prediction is complete
    print('done!')
