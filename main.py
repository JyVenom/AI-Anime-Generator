import base64
import math
import os
import shutil
from tkinter import Label, Entry, Tk, Button, filedialog, Listbox, END, IntVar, Radiobutton

import cv2
import moviepy.editor as mpe
import openai
import requests
from gtts import gTTS
from moviepy.video.VideoClip import TextClip
from moviepy.video.tools.subtitles import SubtitlesClip
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

stability_engine_id = "stable-diffusion-v1-5"  # "stable-diffusion-v1-5" or "stable-diffusion-512-v2-1"
stability_api_host = os.getenv('API_HOST', 'https://api.stability.ai')
stability_api_key = os.getenv("STABILITY_API_KEY")
if stability_api_key is None:
    raise Exception("Missing Stability API key")

# variables
scenes = []
images = []
num_scenes = 0
language = 'English'


# functions
def generate():
    # get user input
    global num_scenes
    global language
    num_scenes = entry1.get()  # for some reason 4 is fine but 5 is not (it doesn't have two spaces between the scenes)
    subject = entry2.get()
    if v.get() == 0:
        language = "English"
    else:
        language = "Chinese"

    # set up directories
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    print(f'Generating {num_scenes} scenes about {subject} in {language}')
    generate_scenes(subject)
    if language != 'English':
        print(f'Translating story into {language}')
        translate_scenes()
    print("Generating Images...")
    generate_images()
    print("Generating Audio...")
    generate_audio()
    print("Generating Video...")
    create_video()
    print("Done!")
    root.destroy()


def generate_scenes(subject):
    prompt = f'Generate a manga story line with {num_scenes} scenes. The manga will have 4 characters, and be about ' \
             f'{subject}. For each scene, give me both a description of the scene in natural language and a prompt I ' \
             f'could use on a text-to-image AI to generate the scene. When writing the image prompt, only use ' \
             f'keywords. Make it specific. Make sure to make a prompt that matches the scene and is in anime style. ' \
             f'Do not use periods in the prompt, instead separating the keywords with commas. No need for "joiner ' \
             f'words" that make the sentence readable. Add ", anime style" to the end of the prompt. Separate each ' \
             f'scene and its corresponding prompt with exactly two new line. Also separate each prompt and the next ' \
             f'scene with exactly two new lines. Answer exactly in the form of "Scene 1: [content] \\n\\n Prompt: ' \
             f'[content] \\n\\n Scene 2: [content] \\n\\n Prompt 2: [content]" and so on, replacing [content] with ' \
             f'the text you generate. Do not include the quotation marks. Do not include new lines after the last line.'
    print(f'Prompt: {prompt}')
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
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


def translate_scenes():
    for i, scene in enumerate(scenes):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f'Translate the following scene into {language}: {scene}'}]
        )
        translation = response['choices'][0]['message']['content']
        print(f'Translation: {translation}')
        scenes[i] = translation


def generate_images():
    for i, prompt in enumerate(images):
        response = requests.post(
            f"{stability_api_host}/v1/generation/{stability_engine_id}/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {stability_api_key}"
            },
            json={
                "text_prompts": [{"text": prompt}],
                "clip_guidance_preset": "FAST_BLUE",  # not sure what this is yet, need to do further research
                "height": 512,
                "width": 512,
                "samples": 1,
                "steps": 10,
                # "height": 576,  # high-res params
                # "width": 1024,
                # "samples": 1,
                # "steps": 50,
                "style_preset": "anime"
            },
        )

        if response.status_code != 200:
            raise Exception("Non-200 response: " + str(response.text))

        print(f'StabilityAI Response: {response}')
        data = response.json()

        with open(os.path.join(data_dir, f"frame_{i:04}.png"), "wb") as f:
            f.write(base64.b64decode(data["artifacts"][0]["base64"]))


def generate_audio():
    for i, caption in enumerate(scenes):
        voice = "en"
        if language == 'Chinese':
            voice = 'zh-CN'

        voiceover = gTTS(text=caption, lang=voice, slow=False)
        voiceover.save(os.path.join(data_dir, f'audio_{i:04}.mp3'))
        print(f'Generated audio for scene {i}')


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

        # add subtitles to scene clips
        subs = [((0, video.duration), scenes[i])]
        generator = lambda txt: TextClip(txt, align='center', method='caption', size=(video.w, None)).set_position(
            ('center', 'bottom')).set_duration(video.duration)
        subtitles = SubtitlesClip(subs, generator)
        video = mpe.CompositeVideoClip([video, subtitles])

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

label3 = Label(root, text="Language:")
label3.grid(row=2, column=0, padx=10, pady=10)

v = IntVar()

rb1 = Radiobutton(root, text="English", variable=v, value=0)
rb1.grid(row=2, column=1, padx=10, pady=10)

rb2 = Radiobutton(root, text="Chinese", variable=v, value=1)
rb2.grid(row=2, column=2, padx=10, pady=10)

listbox = Listbox(root)
listbox.grid(row=3, column=0)

button = Button(root, text="Upload Files", command=open_file_dialog)
button.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="W")

button = Button(root, text="Send", command=generate)
button.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="W")

# main
if __name__ == '__main__':
    root.mainloop()
