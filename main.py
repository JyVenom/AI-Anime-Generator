import base64
import math
import os
import shutil
from tkinter import Label, Entry, Tk, Button, filedialog, Listbox, END

import cv2
import moviepy.editor as mpe
import openai
import requests
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import TextToSpeechV1
from pydub import AudioSegment

# setup
root = Tk()
root.geometry('400x400')
data_dir = "./data"
output_dir = "./out"

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise Exception("Missing OpenAI API Key")
openai.api_key = openai_api_key

stability_engine_id = "stable-diffusion-v1-5"
stability_api_host = os.getenv('API_HOST', 'https://api.stability.ai')
stability_api_key = os.getenv("STABILITY_API_KEY")
if stability_api_key is None:
    raise Exception("Missing Stability API key")

ibm_api_key = os.getenv("IBM_API_KEY")
if ibm_api_key is None:
    raise Exception("Missing IBM API Key")
authenticator = IAMAuthenticator(ibm_api_key)
tts = TextToSpeechV1(authenticator=authenticator)
tts.set_service_url(
    'https://api.us-south.text-to-speech.watson.cloud.ibm.com/instances/786f78a9-376d-4759-a6a5-bcf110898126')

# variables
scenes = []
images = []
num_scenes = 0


# functions
def generate():
    # get user input
    global num_scenes
    num_scenes = entry1.get()
    subject = entry2.get()

    # set up directories
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    print("Generating Story...")
    generate_scenes(subject)
    print("Generating Pictures...")
    generate_images()
    print("Generating Audio...")
    generate_audio()
    print("Stitching Together Video...")
    create_video()
    print("Done. Exiting...")
    root.destroy()


def generate_scenes(subject):
    prompt = f'Generate a manga story line with {num_scenes} scenes. Separate each scene with a new line. The manga ' \
             f'will have 4 characters, and be about {subject}. For each scene, give me both a description of the ' \
             f'scene in natural language and a prompt I could use on a text to image ai to generate the scene. When ' \
             f'writing the image prompt, only use keywords. Make it specific. Make sure to make a prompt that ' \
             f'matches the scene and is in anime style. Do not use periods in the prompt, instead separating the ' \
             f'keywords with commas. Use only keywords. No need for "joiner words" that make the sentence readable. ' \
             f'Add ", anime style" to the end of the prompt. Answer exactly in the form of "Scene 1: [content]" and ' \
             f'"Prompt: [content]" replacing [content] with the text you generate and with exactly two new lines ' \
             f'between scenes and prompts'
    print(f'Prompt: {prompt}')
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    print(f'GPT Response: {response}')
    sentences = response['choices'][0]['message']['content'].split('\n\n')
    sentences = [sentence.strip() for sentence in sentences]
    sentences = [sentence[sentence.index(': ') + 2:] for sentence in sentences]

    for i in range(len(sentences)):
        if i % 2 == 0:
            scenes.append(sentences[i])
        else:
            images.append(sentences[i])

    print(f'Scenes: {scenes}')
    print(f'Image prompts: {images}')


def generate_images():
    for i, prompt in enumerate(images):
        response = requests.post(
            f"{stability_api_host}/v1/generation/{stability_engine_id}/text-to-image",
            headers={
                "Content-Type": "application/json",
                # "Accept": "application/json",
                "Authorization": f"Bearer {stability_api_key}"
            },
            json={
                "text_prompts": [
                    {
                        "text": prompt
                    }
                ],
                # "cfg_scale": 7,
                "clip_guidance_preset": "FAST_BLUE",
                "height": 512,
                "width": 512,
                "samples": 1,
                "steps": 10,
                "style_preset": "anime"
            },
        )

        if response.status_code != 200:
            raise Exception("Non-200 response: " + str(response.text))

        print(f'StabilityAI Response: {response}')
        data = response.json()
        print(f'Data: {data}')

        with open(os.path.join(data_dir, f"frame_{i:04}.png"), "wb") as f:
            f.write(base64.b64decode(data["artifacts"][0]["base64"]))


def generate_audio():
    for i, caption in enumerate(scenes):
        voice = 'en-US_AllisonV3Voice'
        accept = 'audio/mp3'

        # Synthesize speech from a text input
        text = caption
        response = tts.synthesize(text, voice=voice, accept=accept).get_result()

        # Save the audio output to a file
        with open(os.path.join(data_dir, f'audio_{i:04}.mp3'), 'wb') as audio_file:
            audio_file.write(response.content)


def create_video():
    num_frames = len(scenes)

    # params
    fps = 30

    # generate scene video clips
    for i in range(num_frames):
        frame_path = os.path.join(data_dir, f"frame_{i:04}.png")
        frame = cv2.imread(frame_path)
        height, width, channels = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = os.path.join(data_dir, f"scene_{i:04}.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        audio_path = os.path.join(data_dir, f'audio_{i:04}.mp3')
        audio = AudioSegment.from_mp3(audio_path)
        audio_frames = math.ceil(len(audio) / 1000 * fps)

        for j in range(audio_frames):
            out.write(frame)
        out.release()

    # add audio to scene clips
    for i in range(num_frames):
        scene_path = os.path.join(data_dir, f"scene_{i:04}.mp4")
        audio_path = os.path.join(data_dir, f'audio_{i:04}.mp3')
        video = mpe.VideoFileClip(scene_path)
        audio = mpe.AudioFileClip(audio_path)
        video = video.set_audio(audio)
        subs = mpe.TextClip(scenes[i])
        video = mpe.CompositeVideoClip([video, subs.set_position(('center', 'bottom'))])
        video = video.set_duration(video.duration)
        video.write_videofile(os.path.join(data_dir, f"scene_{i:04}.mp4"))

    # concatenate scene clips
    clips = []
    for i in range(num_frames):
        scene_path = os.path.join(data_dir, f"scene_{i:04}.mp4")
        clip = mpe.VideoFileClip(scene_path)
        clips.append(clip)

    # export final video
    final_clip = mpe.concatenate_videoclips(clips)
    final_clip.write_videofile(os.path.join(output_dir, "final.mp4"))


def validate_entry(text):
    if text.isdigit():
        return True
    else:
        return False


def open_file_dialog():
    filenames = filedialog.askopenfilenames(initialdir="/", title="Select Files")
    # print the number of files selected
    # print(len(filenames), "files selected:", filenames)

    # make a clean data folder
    if os.path.exists("user_data/"):
        shutil.rmtree("user_data/")
    os.mkdir("user_data/")

    # iterate through the files and copy them to the "data" folder
    for filename in filenames:
        shutil.copy(filename, "user_data/")
        listbox.insert(END, filename)


# gui
label1 = Label(root, text="Enter number of panes:")
label1.grid(row=0, column=0, padx=10, pady=10)

entry1 = Entry(root, validate="key", validatecommand=(root.register(validate_entry), '%S'))
entry1.grid(row=0, column=1, padx=10, pady=10)

label2 = Label(root, text="Enter subject of Anime:")
label2.grid(row=1, column=0, padx=10, pady=10)

entry2 = Entry(root, validate="key")
entry2.grid(row=1, column=1, padx=10, pady=10)

listbox = Listbox(root)
listbox.grid(row=2, column=0)

button = Button(root, text="Upload Files", command=open_file_dialog)
button.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="W")

button = Button(root, text="Send", command=generate)
button.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="W")

# main
if __name__ == '__main__':
    root.mainloop()
