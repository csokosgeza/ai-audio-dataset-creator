import asyncio
import sys

# Asyncio event loop policy beállítása Windows-ra (Python 3.8+)
if sys.platform == "win32" and sys.version_info >= (3, 8, 0):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import yaml
import os
import torch # A torch.cuda.is_available() ellenőrzéshez
import shutil # Almappák törléséhez (opcionális)
import datetime # Egyedi mappa nevekhez

# Importáljuk a meglévő moduljaink fő funkcióit
# Győződj meg róla, hogy ezek a .py fájlok ugyanabban a mappában vannak, mint az app.py
try:
    from extract_audio import extract_audio_from_video, check_ffmpeg
    from isolate_vocals import isolate_vocals_with_demucs
    from transcribe_segment import transcribe_and_segment
except ImportError as e:
    st.error(f"Hiba a modulok importálásakor: {e}. Győződj meg róla, hogy az `extract_audio.py`, `isolate_vocals.py`, és `transcribe_segment.py` fájlok ugyanabban a mappában vannak, mint az `app.py`, és a virtuális környezet aktív a szükséges csomagokkal.")
    st.stop()


# === Alap Streamlit Beállítások ===
st.set_page_config(page_title="AI Dataset Creator", layout="wide", initial_sidebar_state="expanded")
st.title("🎙️ AI Audio Dataset Creator 🎞️")
st.markdown("Készíts adathalmazt videókból AI modellek (pl. TTS, RVC) tanításához. A folyamat kinyeri a hangot, opcionálisan eltávolítja a háttérzajt, szétválasztja a beszélőket, majd rövid szegmensekre bontja és átírja a beszédet.")

# === Konfiguráció Kezelése ===
CONFIG_PATH = 'config.yaml'

def load_default_config():
    """Betölti az alapértelmezett konfigurációt a YAML fájlból."""
    if not os.path.exists(CONFIG_PATH):
        st.error(f"Alapértelmezett konfigurációs fájl ({CONFIG_PATH}) nem található! Hozz létre egyet a megadott struktúrával, vagy mentsd el a beállításokat az oldalsávon.")
        return {} # Vagy egy minimális default config
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Hiba az alapértelmezett konfiguráció olvasása közben: {e}")
        return {}

def save_current_config_to_yaml(current_config, path=CONFIG_PATH):
    """Elmenti az aktuális UI beállításokat a YAML fájlba (felülírja!)."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(current_config, f, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper) # Biztonságosabb dumper
        st.sidebar.success(f"Konfiguráció mentve ide: {os.path.abspath(path)}")
    except Exception as e:
        st.sidebar.error(f"Hiba a konfiguráció mentése közben: {e}")

config_defaults = load_default_config()
if not config_defaults:
    st.warning("Nem sikerült betölteni a config.yaml fájlt. Alapértelmezett értékekkel próbálkozunk, de mentsd el a beállításaidat!")
    # Minimális alapértelmezett értékek, ha a fájl nem létezik vagy hibás
    config_defaults = {
        'input_video': 'data/',
        'output_base_dir': 'output',
        'output_raw_audio_filename': 'audio_raw.wav',
        'output_clean_audio_filename': 'audio_clean.wav',
        'output_segments_dirname': 'segments',
        'output_metadata_filename': 'metadata.csv',
        'target_sample_rate': 24000,
        'language': 'hu',
        'whisper_model_size': 'large-v3-turbo',
        'min_segment_duration_ms': 3000,
        'max_segment_duration_ms': 13000,
        'use_demucs': True,
        'use_diarization': True,
        'huggingface_token': 'YOUR_HUGGINGFACE_TOKEN_HERE',
        'confidence_threshold': 0.7,
        'device': 'auto',
        'output_dir_structure': 'speaker_separated'
    }


# === Oldalsáv a Konfigurációhoz ===
with st.sidebar:
    st.header("⚙️ Feldolgozási Beállítások")

    # --- Adathalmaz Mód ---
    st.subheader("0. Adathalmaz Cél")
    dataset_mode = st.radio(
        "Válassz feldolgozási módot:",
        ("Új adathalmaz létrehozása", "Meglévő adathalmazhoz adás"),
        key="dataset_mode_radio"
    )

    base_output_for_datasets = config_defaults.get('output_base_dir', 'output')
    os.makedirs(base_output_for_datasets, exist_ok=True)

    target_dataset_name_ui = "" # Inicializálás
    new_dataset_name_suggestion = f"dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if dataset_mode == "Új adathalmaz létrehozása":
        target_dataset_name_ui = st.text_input("Új adathalmaz neve:", value=new_dataset_name_suggestion, key="new_dataset_name")
    else: # Meglévő adathalmazhoz adás
        existing_datasets = [d for d in os.listdir(base_output_for_datasets) if os.path.isdir(os.path.join(base_output_for_datasets, d)) and d != "temp_uploads"] # temp_uploads kizárása
        if not existing_datasets:
            st.warning(f"Nincsenek meglévő adathalmazok a '{base_output_for_datasets}' mappában. Hozz létre egy újat!")
            # Automatikusan átváltunk "Új adathalmaz létrehozása" módra, ha nincs meglévő
            st.session_state.dataset_mode_radio = "Új adathalmaz létrehozása" # UI frissítése
            target_dataset_name_ui = st.text_input("Új adathalmaz neve (mivel nincs meglévő):", value=new_dataset_name_suggestion, key="new_dataset_name_fallback")
        else:
            target_dataset_name_ui = st.selectbox("Válassz meglévő adathalmazt:", options=existing_datasets, key="existing_dataset_select")

    # --- Bemenet ---
    st.subheader("1. Bemenet")
    uploaded_files = st.file_uploader(
        "Videófájl(ok) feltöltése:",
        type=['mp4', 'mkv', 'mov', 'avi', 'webm'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    input_video_path_ui = st.text_input(
        "Vagy add meg a videófájl/mappa elérési útját:",
        value=config_defaults.get('input_video', 'data/'),
        key="path_input"
    )

    # --- Audio Paraméterek ---
    st.subheader("3. Audio Paraméterek")
    target_sr_ui = st.selectbox(
        "Cél mintavételezési frekvencia (Hz):",
        options=[16000, 22050, 24000, 44100, 48000],
        index=[16000, 22050, 24000, 44100, 48000].index(config_defaults.get('target_sample_rate', 24000)),
        key="sample_rate"
    )
    language_ui = st.text_input("Nyelv (ISO kód, pl. 'hu', 'en'):", value=config_defaults.get('language', 'hu'), key="language")

    # --- Whisper Modell ---
    st.subheader("4. Whisper Modell")
    whisper_model_size_ui = st.selectbox(
        "Whisper modell mérete:",
        options=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3-turbo"],
        index=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3-turbo"].index(config_defaults.get('whisper_model_size', 'large-v3-turbo')),
        key="whisper_model"
    )

    # --- Szegmentálás ---
    st.subheader("5. Szegmentálás")
    min_dur_ms_ui = st.number_input("Minimum szegmenshossz (ms):", min_value=500, max_value=30000, value=config_defaults.get('min_segment_duration_ms', 3000), step=100, key="min_duration")
    max_dur_ms_ui = st.number_input("Maximum szegmenshossz (ms):", min_value=1000, max_value=60000, value=config_defaults.get('max_segment_duration_ms', 13000), step=100, key="max_duration")

    # --- Opcionális Funkciók ---
    st.subheader("6. Opcionális Funkciók")
    use_demucs_ui = st.checkbox("Vokál izoláció (Demucs)", value=config_defaults.get('use_demucs', True), key="use_demucs")
    use_diarization_ui = st.checkbox("Beszélők szétválasztása (Diarizáció)", value=config_defaults.get('use_diarization', True), key="use_diarization")
    hf_token_ui = st.text_input("Hugging Face Token (diarizációhoz):", value=config_defaults.get('huggingface_token', 'YOUR_HUGGINGFACE_TOKEN_HERE'), type="password", key="hf_token")

    # --- Minőség és Eszköz ---
    st.subheader("7. Minőség és Eszköz")
    confidence_threshold_ui = st.slider("Konfidencia küszöb (átirat ellenőrzéshez):", min_value=0.0, max_value=1.0, value=config_defaults.get('confidence_threshold', 0.7), step=0.05, key="confidence_threshold")

    device_options = ["auto", "cpu"]
    default_device_index = 0 # Alapértelmezett az 'auto'
    if torch.cuda.is_available():
        device_options.append("cuda")
        # Próbáljuk meg beállítani az alapértelmezettet a config alapján, ha érvényes
        saved_device = config_defaults.get('device', 'auto')
        if saved_device in device_options:
            default_device_index = device_options.index(saved_device)
    elif config_defaults.get('device', 'auto') == "cpu": # Ha nincs CUDA, de a config CPU-t mond
         default_device_index = device_options.index("cpu")


    device_ui = st.selectbox(
        "Eszköz (CPU/GPU):",
        options=device_options,
        index=default_device_index,
        key="device_select"
    )

    # === Konfiguráció Mentése Gomb ===
    if st.button("Jelenlegi beállítások mentése config.yaml-ba", key="save_config_button"):
        current_config_for_save = {
            'input_video': input_video_path_ui,
            'output_base_dir': base_output_for_datasets,
            'output_raw_audio_filename': config_defaults.get('output_raw_audio_filename'),
            'output_clean_audio_filename': config_defaults.get('output_clean_audio_filename'),
            'output_segments_dirname': config_defaults.get('output_segments_dirname'),
            'output_metadata_filename': config_defaults.get('output_metadata_filename'),
            'target_sample_rate': target_sr_ui,
            'language': language_ui,
            'whisper_model_size': whisper_model_size_ui,
            'min_segment_duration_ms': min_dur_ms_ui,
            'max_segment_duration_ms': max_dur_ms_ui,
            'use_demucs': use_demucs_ui,
            'use_diarization': use_diarization_ui,
            'huggingface_token': hf_token_ui,
            'confidence_threshold': confidence_threshold_ui,
            'device': device_ui,
            'output_dir_structure': 'speaker_separated' if use_diarization_ui else 'flat'
        }
        save_current_config_to_yaml(current_config_for_save)

# === Fő Tartalom - Feldolgozás Indítása ===
st.markdown("---")
if st.button("🚀 Teljes Feldolgozás Indítása", type="primary", use_container_width=True, key="start_processing_button"):
    if not target_dataset_name_ui or not target_dataset_name_ui.strip():
        st.error("Kérlek, adj meg egy érvényes nevet az új adathalmaznak, vagy válassz egy meglévőt!")
        st.stop()

    # A tényleges kimeneti mappa az adathalmaz nevével a base_output_for_datasets alatt
    actual_output_dataset_dir = os.path.join(base_output_for_datasets, target_dataset_name_ui)
    os.makedirs(actual_output_dataset_dir, exist_ok=True)
    st.info(f"Kimenetek mentése a következő adathalmaz mappába: **{os.path.abspath(actual_output_dataset_dir)}**")

    # Aktuális futtatási konfiguráció (ezt adjuk át a moduloknak)
    runtime_config = {
        'output_base_dir': actual_output_dataset_dir, # Ez lesz az adathalmaz gyökere
        'output_raw_audio_filename': config_defaults.get('output_raw_audio_filename', 'audio_raw.wav'),
        'output_clean_audio_filename': config_defaults.get('output_clean_audio_filename', 'audio_clean.wav'),
        'output_segments_dirname': config_defaults.get('output_segments_dirname', 'segments'),
        'output_metadata_filename': config_defaults.get('output_metadata_filename', 'metadata.csv'), # A közös CSV neve
        'target_sample_rate': target_sr_ui,
        'language': language_ui,
        'whisper_model_size': whisper_model_size_ui,
        'min_segment_duration_ms': min_dur_ms_ui,
        'max_segment_duration_ms': max_dur_ms_ui,
        'use_demucs': use_demucs_ui,
        'use_diarization': use_diarization_ui,
        'huggingface_token': hf_token_ui if hf_token_ui and hf_token_ui.strip() and hf_token_ui != 'YOUR_HUGGINGFACE_TOKEN_HERE' else None,
        'confidence_threshold': confidence_threshold_ui,
        'device': device_ui,
        'output_dir_structure': 'speaker_separated' if use_diarization_ui else 'flat'
    }

    # Bemeneti videó(k) listájának összeállítása
    videos_to_process = []
    temp_uploaded_paths = [] # Ideiglenesen feltöltött fájlok útvonalai törléshez

    # Ideiglenes feltöltési mappa a fő kimeneti mappán belül
    temp_upload_main_dir = os.path.join(base_output_for_datasets, "temp_uploads")

    if uploaded_files:
        os.makedirs(temp_upload_main_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            temp_video_path = os.path.join(temp_upload_main_dir, uploaded_file.name)
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            videos_to_process.append(temp_video_path)
            temp_uploaded_paths.append(temp_video_path)
        st.info(f"{len(uploaded_files)} videó feltöltve feldolgozásra.")
    elif input_video_path_ui and input_video_path_ui.strip():
        path_to_check = input_video_path_ui
        if not os.path.isabs(path_to_check):
            path_to_check = os.path.abspath(path_to_check)

        if os.path.isdir(path_to_check):
            st.info(f"Videók keresése a '{path_to_check}' mappában...")
            for filename in os.listdir(path_to_check):
                if filename.lower().endswith(('.mp4', '.mkv', '.mov', '.avi', '.webm')):
                    videos_to_process.append(os.path.join(path_to_check, filename))
            if not videos_to_process:
                st.warning(f"Nem találhatóak videófájlok a '{path_to_check}' mappában.")
            else:
                st.info(f"{len(videos_to_process)} videó található a mappában.")
        elif os.path.isfile(path_to_check):
            videos_to_process.append(path_to_check)
        else:
            st.error(f"A megadott elérési út nem létezik vagy nem támogatott: {path_to_check}")
            st.stop()
    else:
        st.error("Nincs videó kiválasztva vagy elérési út megadva.")
        st.stop()

    if not videos_to_process:
        st.error("Nincsenek feldolgozandó videók.")
        st.stop()

    # --- Feldolgozási Ciklus Minden Videóra ---
    all_videos_processed_successfully = True
    # Hely a logoknak a főoldalon
    log_placeholder = st.empty()
    progress_bar = st.progress(0)
    total_videos = len(videos_to_process)

    for video_index, current_video_path_original in enumerate(videos_to_process):
        video_basename = os.path.basename(current_video_path_original)
        log_placeholder.info(f"Feldolgozás alatt: {video_basename} ({video_index + 1}/{total_videos})")
        progress_bar.progress((video_index + 1) / total_videos)

        video_name_no_ext = os.path.splitext(video_basename)[0]
        safe_video_name = "".join(c if c.isalnum() else "_" for c in video_name_no_ext)

        # A videó-specifikus kimeneti mappa az ADATHALMAZ mappán BELÜL jön létre
        # Ez lesz a feldolgozó szkriptek `output_base_dir`-je
        video_processing_output_dir = os.path.join(actual_output_dataset_dir, safe_video_name)
        os.makedirs(video_processing_output_dir, exist_ok=True)

        # Konfiguráció az aktuális videóhoz
        video_config = runtime_config.copy()
        video_config['input_video'] = current_video_path_original
        video_config['output_base_dir'] = video_processing_output_dir # A szkriptek ide mentik a nyers, tiszta, szegmens fájlokat
        # A metadata.csv útvonalát a runtime_config-ból vesszük, hogy közös legyen
        video_config['output_metadata_file_absolute_path'] = os.path.join(actual_output_dataset_dir, runtime_config['output_metadata_filename'])
        # A szegmens fájlok relatív útvonalának tartalmaznia kell a videó nevét is
        video_config['segments_relative_path_prefix'] = safe_video_name


        st.info(f"Indul: {video_basename}")
        st.caption(f"Kimenetek (nyers, tiszta, szegmensek) ide: {os.path.abspath(video_config['output_base_dir'])}")

        raw_audio_file = None
        clean_audio_file_for_this_video = None
        input_for_transcription_this_video = None

        # --- 1. Audio Kinyerése ---
        with st.expander(f"[{safe_video_name}] 1. Audio kinyerése...", expanded=True):
            if not check_ffmpeg():
                st.error("FFmpeg nem található.")
                all_videos_processed_successfully = False; continue
            raw_audio_file = extract_audio_from_video(video_config)
            if raw_audio_file: st.success(f"Nyers audio: {raw_audio_file}")
            else:
                st.error("Audio kinyerés sikertelen."); all_videos_processed_successfully = False; continue

        # --- 2. Vokál Izoláció (Demucs) ---
        if raw_audio_file:
            if video_config['use_demucs']:
                with st.expander(f"[{safe_video_name}] 2. Vokál izoláció (Demucs)...", expanded=True):
                    clean_audio_file_for_this_video = isolate_vocals_with_demucs(video_config, raw_audio_file)
                    if clean_audio_file_for_this_video:
                        st.success(f"Tiszta vokál: {clean_audio_file_for_this_video}")
                        input_for_transcription_this_video = clean_audio_file_for_this_video
                    else:
                        st.warning("Demucs hiba. Nyers audió használata.")
                        input_for_transcription_this_video = raw_audio_file
            else:
                st.info("Demucs kihagyva.")
                input_for_transcription_this_video = raw_audio_file

            # --- 3. Transzkripció és Szegmentálás ---
            if input_for_transcription_this_video:
                with st.expander(f"[{safe_video_name}] 3. Transzkripció és szegmentálás...", expanded=True):
                    st.write(f"WhisperX futtatása: {os.path.basename(input_for_transcription_this_video)}")
                    # A transcribe_segment.py-nek a video_config-ot adjuk át.
                    # Ennek tartalmaznia kell a 'output_metadata_file_absolute_path' és
                    # 'segments_relative_path_prefix' kulcsokat a közös CSV és helyes relatív utakhoz.
                    transcribe_and_segment(video_config)
                    st.success("Transzkripció és szegmentálás befejezve ehhez a videóhoz!")
            else:
                st.error("Nincs audio fájl a transzkripcióhoz."); all_videos_processed_successfully = False
        else:
            all_videos_processed_successfully = False

    # Ideiglenes feltöltött videók törlése
    if temp_uploaded_paths:
        st.markdown("---")
        st.write("Ideiglenes feltöltött fájlok törlése...")
        for temp_path in temp_uploaded_paths:
            if os.path.exists(temp_path):
                try: os.remove(temp_path); st.caption(f"Törölve: {temp_path}")
                except Exception as e: st.warning(f"Hiba törléskor ({temp_path}): {e}")
        # Ideiglenes mappa törlése, ha üres
        if os.path.exists(temp_upload_main_dir) and not os.listdir(temp_upload_main_dir):
            try: os.rmdir(temp_upload_main_dir); st.caption(f"Ideiglenes mappa törölve: {temp_upload_main_dir}")
            except Exception as e: st.warning(f"Hiba az ideiglenes mappa törlésekor: {e}")


    st.markdown("---")
    if all_videos_processed_successfully and videos_to_process:
        st.balloons()
        st.header("🎉 Feldolgozás Befejezve! 🎉")
        st.success(f"Az összes videó feldolgozása sikeresen megtörtént.")
        st.write(f"A közös metaadatok (átiratok) a `{os.path.abspath(os.path.join(actual_output_dataset_dir, runtime_config['output_metadata_filename']))}` fájlban találhatók.")
        st.write(f"A szegmentált audio fájlok a `{os.path.abspath(actual_output_dataset_dir)}` mappán belüli, videó nevű almappákban (`segments` almappákon belül) vannak.")
    elif videos_to_process:
        st.warning("Egy vagy több videó feldolgozása során hiba történt. Ellenőrizd a fenti üzeneteket.")
    else:
        st.info("Nem volt feldolgozandó videó.")

with st.sidebar:
    st.markdown("---")
    st.info("Fejlesztette: AI Asszisztens")
