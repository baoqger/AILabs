# Import libraries
import base64
import json
import os
import time

import cv2
import streamlit as st
import yt_dlp
from dotenv import load_dotenv
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from openai import AzureOpenAI
from yt_dlp.utils import download_range_func

# Default configuration
SEGMENT_DURATION = 20  # In seconds, set to 0 to not split the video
SYSTEM_PROMPT = "You are a helpful assistant that describes in detail a video based on sampled frames."
USER_PROMPT = "These are the frames from the video."
DEFAULT_TEMPERATURE = 0.5
RESIZE_OF_FRAMES = 2
SECONDS_PER_FRAME = 30
DEFAULT_URL = "https://www.youtube.com/watch?v=Y6kHpAeIr4c"

# Load configuration
load_dotenv(override=True)

# Configuration of Azure OpenAI
aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_apikey = os.environ["AZURE_OPENAI_API_KEY"]
aoai_apiversion = os.environ["AZURE_OPENAI_API_VERSION"]
aoai_model_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
default_system_prompt = os.environ.get("SYSTEM_PROMPT", SYSTEM_PROMPT)
print(f"aoai_endpoint: {aoai_endpoint}, aoai_model_name: {aoai_model_name}")

# Create AOAI client for answer generation
aoai_client = AzureOpenAI(
    api_version=aoai_apiversion,
    azure_endpoint=aoai_endpoint,
    api_key=aoai_apikey,
)


def apply_custom_theme():
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        #MainMenu,
        footer {
            display: none !important;
        }
        .block-container {
            padding-top: 1.25rem;
        }
        .stApp {
            background: linear-gradient(180deg, #0f1117 0%, #121723 100%);
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #253043;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(78, 161, 255, 0.16), rgba(124, 92, 255, 0.14));
            border: 1px solid rgba(78, 161, 255, 0.30);
            border-radius: 20px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            color: #f8fafc;
        }
        .hero-subtitle {
            color: #cbd5e1;
            margin-top: 0.4rem;
            margin-bottom: 0;
        }
        .model-badge {
            display: inline-block;
            margin-top: 0.85rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.35);
            color: #bbf7d0;
            font-size: 0.9rem;
        }
        .section-card {
            background: #161a23;
            border: 1px solid #2a3140;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        .section-title {
            color: #f8fafc;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }
        .muted-text {
            color: #9ca3af;
            margin-bottom: 0;
        }
        .empty-state {
            background: rgba(22, 26, 35, 0.75);
            border: 1px dashed #334155;
            border-radius: 18px;
            padding: 1.4rem;
            text-align: center;
            color: #94a3b8;
        }
        div[data-testid="stMetric"] {
            background: #161a23;
            border: 1px solid #2a3140;
            border-radius: 16px;
            padding: 0.8rem 1rem;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #2a3140;
            border-radius: 16px;
            background: #161a23;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_duration(seconds_value):
    return f"{seconds_value:.1f}s"


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        raise ValueError(f"Unable to detect FPS for video: {video_path}")

    return total_frames / fps


def resolve_segment_path(file_stem):
    for extension in (".mp4", ".mkv", ".webm"):
        candidate = f"{file_stem}{extension}"
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Segment file was not found for {file_stem}")


# Function to encode a local video into frames
def process_video(video_path, seconds_per_frame=SECONDS_PER_FRAME, resize=RESIZE_OF_FRAMES, output_dir=""):
    base64_frames = []

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        fps = 1

    frames_to_skip = max(int(fps * seconds_per_frame), 1)
    curr_frame = 0

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        frame_count = 1

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break

        if resize:
            height, width, _ = frame.shape
            frame = cv2.resize(frame, (max(1, width // resize), max(1, height // resize)))

        _, buffer = cv2.imencode(".jpg", frame)

        if output_dir:
            frame_filename = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count}.jpg",
            )
            print(f"Saving frame {frame_filename}")
            with open(frame_filename, "wb") as file_handle:
                file_handle.write(buffer)
            frame_count += 1

        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip

    video.release()
    print(f"Extracted {len(base64_frames)} frames")
    return base64_frames


# Function to analyze the video with GPT-4o
def analyze_video(base64frames, system_prompt, user_prompt, temperature):
    print(f"SYSTEM PROMPT: [{system_prompt}]")
    print(f"USER PROMPT:   [{user_prompt}]")

    try:
        response = aoai_client.chat.completions.create(
            model=aoai_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {
                    "role": "user",
                    "content": [
                        *map(
                            lambda frame: {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpg;base64,{frame}", "detail": "auto"},
                            },
                            base64frames,
                        ),
                    ],
                },
            ],
            temperature=temperature,
            max_tokens=4096,
        )

        json_response = json.loads(response.model_dump_json())
        response = json_response["choices"][0]["message"]["content"]
    except Exception as ex:
        print(f"ERROR: {ex}")
        response = f"ERROR: {ex}"

    return response


# Split the video in segments of N seconds. If segment_length is 0 the full video is processed.
def split_video(video_path, output_dir, segment_length=180):
    duration = get_video_duration(video_path)

    if segment_length == 0:
        segment_length = max(int(duration), 1)

    for start_time in range(0, int(duration), segment_length):
        end_time = min(start_time + segment_length, duration)
        output_file = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(video_path))[0]}_segment_{start_time}-{int(end_time)}_secs.mp4",
        )
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_file)
        yield output_file, start_time, end_time


def execute_video_processing(
    segment_path,
    start_time,
    end_time,
    system_prompt,
    user_prompt,
    temperature,
    seconds_per_frame,
    resize,
    save_frames,
    preview_placeholder,
    status_container,
):
    segment_name = os.path.basename(segment_path)
    preview_placeholder.video(segment_path)

    total_start = time.time()
    output_dir = "frames" if save_frames else ""

    with status_container.status(f"Processing {segment_name}", expanded=True) as status:
        status.write("Sampling frames from the current video segment.")
        frame_start = time.time()
        base64frames = process_video(
            segment_path,
            seconds_per_frame=seconds_per_frame,
            resize=resize,
            output_dir=output_dir,
        )
        frame_elapsed = time.time() - frame_start
        status.write(f"Extracted {len(base64frames)} sampled frames in {format_duration(frame_elapsed)}.")

        status.write(f"Sending frames to `{aoai_model_name}` for multimodal analysis.")
        analysis_start = time.time()
        analysis = analyze_video(base64frames, system_prompt, user_prompt, temperature)
        analysis_elapsed = time.time() - analysis_start
        total_elapsed = time.time() - total_start

        status.write(f"Model response received in {format_duration(analysis_elapsed)}.")
        status.update(label=f"Completed {segment_name}", state="complete", expanded=False)

    return {
        "segment_name": segment_name,
        "time_window": f"{int(start_time)}s - {int(end_time)}s",
        "frame_count": len(base64frames),
        "frame_extraction_seconds": frame_elapsed,
        "analysis_seconds": analysis_elapsed,
        "total_seconds": total_elapsed,
        "analysis": analysis,
    }


def render_results(summary, results):
    st.markdown("## Analysis Results")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Segments", summary["segments"])
    metric_columns[1].metric("Frames Sampled", summary["frames"])
    metric_columns[2].metric("Elapsed", format_duration(summary["elapsed_seconds"]))
    metric_columns[3].metric("Model", summary["model"])

    st.caption(
        f"Source: {summary['source_type']} | "
        f"Segment length: {summary['segment_length']} | "
        f"Sampling: every {summary['seconds_per_frame']}s | "
        f"Resize ratio: {summary['resize']}"
    )

    export_payload = json.dumps({"summary": summary, "results": results}, indent=2)
    st.download_button(
        "Download analysis JSON",
        data=export_payload,
        file_name="video-analysis-results.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("### Segment Details")
    for index, result in enumerate(results, start=1):
        with st.expander(f"Segment {index} | {result['time_window']}", expanded=index == 1):
            detail_columns = st.columns(3)
            detail_columns[0].metric("Sampled Frames", result["frame_count"])
            detail_columns[1].metric("Frame Extraction", format_duration(result["frame_extraction_seconds"]))
            detail_columns[2].metric("Model Analysis", format_duration(result["analysis_seconds"]))

            if result["analysis"].startswith("ERROR:"):
                st.error(result["analysis"])
            else:
                st.markdown(result["analysis"])


# Streamlit User Interface
st.set_page_config(
    page_title="Video Analysis with Azure OpenAI",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_theme()

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []
if "analysis_summary" not in st.session_state:
    st.session_state.analysis_summary = None

header_columns = st.columns([0.8, 4.6])
with header_columns[0]:
    st.image("microsoft.png", width=88)
with header_columns[1]:
    st.markdown(
        f"""
        <div class="hero-card">
            <p class="hero-title">Video Analysis Dashboard</p>
            <p class="hero-subtitle">
                Upload a clip or analyze a supported video URL, then review AI-generated descriptions segment by segment.
            </p>
            <span class="model-badge">Azure OpenAI model: {aoai_model_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("## Configure Analysis")
    st.caption("Set the video source first, then tune sampling and prompt behavior if needed.")

    with st.form("analysis_form"):
        file_or_url = st.radio("Video source", ["File", "URL"], horizontal=True)

        video_file = None
        url = ""
        continuous_transmission = False

        if file_or_url == "File":
            video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        else:
            url = st.text_input("Video URL", value=DEFAULT_URL)
            continuous_transmission = st.checkbox(
                "Continuous transmission",
                value=False,
                help="Useful for long livestream-like sources where you want repeated short slices.",
            )

        initial_split = SEGMENT_DURATION if file_or_url == "URL" and continuous_transmission else 0
        seconds_split = int(
            st.number_input(
                "Segment length (seconds)",
                min_value=0,
                value=int(initial_split),
                step=10,
                help="0 processes the full video as a single segment.",
            )
        )

        with st.expander("Advanced Settings"):
            seconds_per_frame = float(
                st.number_input(
                    "Seconds per frame",
                    min_value=0.1,
                    value=float(SECONDS_PER_FRAME),
                    step=0.5,
                    help="Sample one frame every N seconds. Lower values capture more detail but increase cost.",
                )
            )
            resize = int(
                st.number_input(
                    "Frame resize ratio",
                    min_value=0,
                    value=int(RESIZE_OF_FRAMES),
                    step=1,
                    help="0 keeps original size. Higher values shrink each frame before sending it to the model.",
                )
            )
            save_frames = st.checkbox('Save sampled frames to the "frames" folder', value=False)
            temperature = float(
                st.slider(
                    "Model temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=float(DEFAULT_TEMPERATURE),
                    step=0.1,
                )
            )
            system_prompt = st.text_area("System prompt", value=default_system_prompt, height=120)
            user_prompt = st.text_area("User prompt", value=USER_PROMPT, height=100)

        submitted = st.form_submit_button(
            "Analyze video",
            use_container_width=True,
            type="primary",
        )

    st.caption("Choose a video file or enter a video URL, then click Analyze video.")

# Prepare working directories
output_dir = "segments"
os.makedirs(output_dir, exist_ok=True)

main_columns = st.columns([1.45, 1])
preview_placeholder = main_columns[0].empty()

with main_columns[1]:
    st.markdown("### Current Setup")
    setup_columns = st.columns(2)
    setup_columns[0].metric("Source", file_or_url)
    setup_columns[1].metric("Segment Length", "Full video" if seconds_split == 0 else f"{seconds_split}s")
    setup_columns = st.columns(2)
    setup_columns[0].metric("Sampling", f"{seconds_per_frame}s/frame")
    setup_columns[1].metric("Resize", "Original" if resize == 0 else f"1/{resize}")
    st.metric("Temperature", f"{temperature:.1f}")
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Workflow</div>
            <p class="muted-text">1. Choose a file or URL. 2. Adjust advanced settings if needed. 3. Run analysis and review segment summaries below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with main_columns[0]:
    st.markdown("### Source Preview")
    if file_or_url == "File" and video_file is not None:
        preview_placeholder.video(video_file)
        st.caption(f"Ready to analyze `{video_file.name}`")
    elif file_or_url == "URL" and url.strip():
        preview_placeholder.markdown(
            f"""
            <div class="empty-state">
                <strong>URL source selected</strong><br><br>
                The app will download and analyze segments from the provided link after you start the run.<br><br>
                <code>{url}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        preview_placeholder.markdown(
            """
            <div class="empty-state">
                Upload a video or enter a supported URL to start a run.
            </div>
            """,
            unsafe_allow_html=True,
        )

if submitted:
    st.session_state.analysis_results = []
    st.session_state.analysis_summary = None
    progress_bar = st.progress(0, text="Preparing analysis...")
    live_status_container = st.container()
    overall_start = time.time()
    results = []

    print("PARAMETERS:")
    print(f"file_or_url: {file_or_url}, seconds to split: {seconds_split}")
    print(
        f"seconds_per_frame: {seconds_per_frame}, resize ratio: {resize}, "
        f"save_frames: {save_frames}, temperature: {temperature}"
    )

    try:
        if file_or_url == "URL" and not url.strip():
            raise ValueError("Please enter a video URL before running the analysis.")

        if file_or_url == "URL":
            base_ydl_opts = {
                "format": "(bestvideo[vcodec^=av01]/bestvideo[vcodec^=vp9]/bestvideo)+bestaudio/best",
                "force_keyframes_at_cuts": True,
                "quiet": True,
                "no_warnings": True,
            }

            if not continuous_transmission:
                info_dict = yt_dlp.YoutubeDL(base_ydl_opts).extract_info(url, download=False)
                video_duration = int(info_dict.get("duration", 0))
                segment_duration = video_duration if seconds_split == 0 else int(seconds_split)
            else:
                video_duration = int(48 * 60 * 60)
                segment_duration = 180 if seconds_split == 0 else int(seconds_split)

            segment_starts = list(range(0, video_duration, max(segment_duration, 1)))

            for index, start in enumerate(segment_starts, start=1):
                end = min(start + segment_duration, video_duration)
                file_stem = os.path.join(output_dir, f"segment_{start}-{int(end)}")
                progress_bar.progress(
                    (index - 1) / max(len(segment_starts), 1),
                    text=f"Downloading segment {index} of {len(segment_starts)}",
                )

                with live_status_container.status(
                    f"Downloading segment {index} of {len(segment_starts)}",
                    expanded=True,
                ) as download_status:
                    download_status.write(f"Requested time window: {start}s to {int(end)}s.")
                    segment_opts = {
                        **base_ydl_opts,
                        "outtmpl": {"default": f"{file_stem}.%(ext)s"},
                        "download_ranges": download_range_func(None, [(start, end)]),
                    }
                    yt_dlp.YoutubeDL(segment_opts).download([url])
                    download_status.update(
                        label=f"Downloaded segment {index} of {len(segment_starts)}",
                        state="complete",
                        expanded=False,
                    )

                segment_path = resolve_segment_path(file_stem)
                results.append(
                    execute_video_processing(
                        segment_path,
                        start,
                        end,
                        system_prompt,
                        user_prompt,
                        temperature,
                        seconds_per_frame,
                        resize,
                        save_frames,
                        preview_placeholder,
                        live_status_container,
                    )
                )
                progress_bar.progress(
                    index / max(len(segment_starts), 1),
                    text=f"Processed segment {index} of {len(segment_starts)}",
                )

                if os.path.exists(segment_path):
                    os.remove(segment_path)

        else:
            if video_file is None:
                raise ValueError("Please upload a video file before running the analysis.")

            os.makedirs("temp", exist_ok=True)
            video_path = os.path.join("temp", video_file.name)
            with open(video_path, "wb") as file_handle:
                file_handle.write(video_file.getbuffer())

            segments = list(split_video(video_path, output_dir, seconds_split))
            for index, (segment_path, start, end) in enumerate(segments, start=1):
                progress_bar.progress(
                    (index - 1) / max(len(segments), 1),
                    text=f"Processing segment {index} of {len(segments)}",
                )
                results.append(
                    execute_video_processing(
                        segment_path,
                        start,
                        end,
                        system_prompt,
                        user_prompt,
                        temperature,
                        seconds_per_frame,
                        resize,
                        save_frames,
                        preview_placeholder,
                        live_status_container,
                    )
                )
                progress_bar.progress(
                    index / max(len(segments), 1),
                    text=f"Processed segment {index} of {len(segments)}",
                )
                if os.path.exists(segment_path):
                    os.remove(segment_path)

            if os.path.exists(video_path):
                os.remove(video_path)

        summary = {
            "segments": len(results),
            "frames": sum(item["frame_count"] for item in results),
            "elapsed_seconds": time.time() - overall_start,
            "model": aoai_model_name,
            "source_type": file_or_url,
            "segment_length": "Full video" if seconds_split == 0 else f"{seconds_split}s",
            "seconds_per_frame": seconds_per_frame,
            "resize": "Original" if resize == 0 else f"1/{resize}",
        }
        st.session_state.analysis_results = results
        st.session_state.analysis_summary = summary
        progress_bar.progress(1.0, text="Analysis complete.")
        st.success("Video analysis completed.")
    except Exception as ex:
        print(f"ERROR: {ex}")
        st.error(f"ERROR: {ex}")

if st.session_state.analysis_summary and st.session_state.analysis_results:
    render_results(st.session_state.analysis_summary, st.session_state.analysis_results)