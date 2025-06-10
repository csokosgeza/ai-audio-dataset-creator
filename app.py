import asyncio
import sys

# Asyncio event loop policy beállítása Windows-ra (Python 3.8+)
if sys.platform == "win32" and sys.version_info >= (3, 8, 0):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import yaml
import os
import torch  # A torch.cuda.is_available() ellenőrzéshez
import shutil  # Almappák törléséhez (opcionális)
import datetime  # Egyedi mappa nevekhez

# Importáljuk a meglévő moduljaink fő funkcióit
try:
    from extract_audio import process_media_file, check_ffmpeg
    from isolate_vocals import isolate_vocals_with_demucs
    from transcribe_segment import transcribe_and_segment
except ImportError as e:
    st.error(
        f"Hiba a modulok importálásakor: {e}. Győződj meg róla, hogy az `extract_audio.py`, `isolate_vocals.py`, és `transcribe_segment.py` fájlok ugyanabban a mappában vannak, mint az `app.py`, és a virtuális környezet aktív a szükséges csomagokkal.")
    st.stop()

# === Alap Streamlit Beállítások ===
st.set_page_config(page_title="AI Dataset Creator", layout="wide", initial_sidebar_state="expanded")
st.title("🎙️ AI Audio Dataset Creator 🎞️")
st.markdown(
    "Készíts adathalmazt videókból AI modellek (pl. TTS, RVC) tanításához. A folyamat kinyeri a hangot, opcionálisan eltávolítja a háttérzajt, szétválasztja a beszélőket, majd rövid szegmensekre bontja és átírja a beszédet.")

# === Konfiguráció Kezelése ===
CONFIG_PATH = 'config.yaml'


def load_default_config():
    if not os.path.exists(CONFIG_PATH):
        st.error(
            f"Alapértelmezett konfigurációs fájl ({CONFIG_PATH}) nem található! Hozz létre egyet a `config.example.yaml` alapján, vagy mentsd el a beállításokat az oldalsávon.")
        return {
            'input_path': 'data/', 'output_base_dir': 'output',
            'output_raw_audio_filename': 'audio_raw.wav', 'output_clean_audio_filename': 'audio_clean.wav',
            'output_segments_dirname': 'segments', 'output_metadata_filename': 'metadata.csv',
            'target_sample_rate': 24000, 'language': 'hu', 'whisper_model_size': 'large-v3-turbo',
            'min_segment_duration_ms': 3000, 'max_segment_duration_ms': 13000,
            'use_demucs': True, 'use_diarization': True,
            'huggingface_token': 'YOUR_HUGGINGFACE_TOKEN_HERE',
            'confidence_threshold': 0.7, 'device': 'auto',
            'output_dir_structure': 'speaker_separated'
        }
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Hiba az alapértelmezett konfiguráció olvasása közben: {e}")
        return {}


def save_current_config_to_yaml(current_config, path=CONFIG_PATH):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(current_config, f, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper)
        st.sidebar.success(f"Konfiguráció mentve ide: {os.path.abspath(path)}")
    except Exception as e:
        st.sidebar.error(f"Hiba a konfiguráció mentése közben: {e}")


config_defaults = load_default_config()
if not config_defaults:
    st.warning(
        "Nem sikerült betölteni a config.yaml fájlt. Alapértelmezett értékekkel próbálkozunk, de mentsd el a beállításaidat!")
    config_defaults = {
        'input_video': 'data/', 'output_base_dir': 'output',
        'output_raw_audio_filename': 'audio_raw.wav', 'output_clean_audio_filename': 'audio_clean.wav',
        'output_segments_dirname': 'segments', 'output_metadata_filename': 'metadata.csv',
        'target_sample_rate': 24000, 'language': 'hu', 'whisper_model_size': 'large-v3-turbo',
        'min_segment_duration_ms': 3000, 'max_segment_duration_ms': 13000,
        'use_demucs': True, 'use_diarization': True,
        'huggingface_token': 'YOUR_HUGGINGFACE_TOKEN_HERE',
        'confidence_threshold': 0.7, 'device': 'auto',
        'output_dir_structure': 'speaker_separated'
    }

# === Globális Változók a UI Állapothoz ===
base_output_for_datasets = config_defaults.get('output_base_dir', 'output')
os.makedirs(base_output_for_datasets, exist_ok=True)
temp_upload_main_dir = os.path.join(base_output_for_datasets, "temp_uploads")  # Ideiglenes feltöltések mappája
os.makedirs(temp_upload_main_dir, exist_ok=True)
existing_datasets = [d for d in os.listdir(base_output_for_datasets) if
                     os.path.isdir(os.path.join(base_output_for_datasets, d)) and d != "temp_uploads"]

# === Oldalsáv a Konfigurációhoz ===
with st.sidebar:
    st.header("⚙️ Feldolgozási Beállítások")

    # --- Adathalmaz Mód ---
    st.subheader("0. Adathalmaz Cél")
    dataset_mode_options = ("Új adathalmaz létrehozása", "Meglévő adathalmazhoz adás")

    if 'dataset_mode_radio_state' not in st.session_state:
        st.session_state.dataset_mode_radio_state = dataset_mode_options[0]

    # Ha nincs meglévő adathalmaz, és a "Meglévőhöz adás" van kiválasztva, automatikusan váltson "Új"-ra
    if not existing_datasets and st.session_state.dataset_mode_radio_state == dataset_mode_options[1]:
        st.session_state.dataset_mode_radio_state = dataset_mode_options[0]
        # Nem hívunk rerun-t itt, a radio widget indexe majd frissül

    current_radio_index = dataset_mode_options.index(st.session_state.dataset_mode_radio_state)

    dataset_mode = st.radio(
        "Válassz feldolgozási módot:",
        dataset_mode_options,
        index=current_radio_index,  # Az aktuális session state alapján
        key="dataset_mode_radio_state_widget"  # Másik kulcs, hogy ne ütközzön a state-tel
    )
    # Frissítjük a session state-et a widget aktuális értékével (ha változott)
    st.session_state.dataset_mode_radio_state = dataset_mode

    target_dataset_name_ui = ""
    new_dataset_name_suggestion = f"dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if st.session_state.dataset_mode_radio_state == "Új adathalmaz létrehozása":
        target_dataset_name_ui = st.text_input("Új adathalmaz neve:", value=new_dataset_name_suggestion,
                                               key="new_dataset_name_input")
    else:  # Meglévő adathalmazhoz adás
        if not existing_datasets:  # Ezt az esetet a radio gombnak már kezelnie kellett volna
            st.warning(f"Nincsenek meglévő adathalmazok. Kérlek, válts \"Új adathalmaz létrehozása\" módra.")
            target_dataset_name_ui = ""  # Nincs mit kiválasztani
        else:
            selected_existing_dataset = st.selectbox(
                "Válassz meglévő adathalmazt:",
                options=existing_datasets,
                index=0 if existing_datasets else None,  # Csak akkor van index, ha van opció
                key="existing_dataset_select_box"
            )
            target_dataset_name_ui = selected_existing_dataset if selected_existing_dataset else ""

    # --- Bemenet ---
    # ... (többi UI elem változatlan) ...
    st.subheader("1. Bemenet")
    uploaded_files = st.file_uploader(
        "Videó/hangfájl(ok) feltöltése:", type=['mp4', 'mkv', 'mov', 'avi', 'webm', 'mp3', 'wav', 'flac', 'm4a'], # Változás: audio formátumok hozzáadva
        accept_multiple_files=True, key="file_uploader_widget"
    )
    input_path_ui = st.text_input( # Változás: input_video_path_ui -> input_path_ui
        "Vagy add meg a médiafájl/mappa elérési útját:", # Változás: videófájl -> médiafájl
        value=config_defaults.get('input_path', 'data/'), key="path_input_widget" # Változás: input_video -> input_path
    )

    st.subheader("3. Audio Paraméterek")
    target_sr_ui = st.selectbox(
        "Cél mintavételezési frekvencia (Hz):", options=[16000, 22050, 24000, 44100, 48000],
        index=[16000, 22050, 24000, 44100, 48000].index(config_defaults.get('target_sample_rate', 24000)),
        key="sample_rate_select"
    )
    language_ui = st.text_input("Nyelv (ISO kód, pl. 'hu', 'en'):", value=config_defaults.get('language', 'hu'),
                                key="language_input")

    st.subheader("4. Whisper Modell")
    whisper_model_size_ui = st.selectbox(
        "Whisper modell mérete:", options=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3-turbo"],
        index=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3-turbo"].index(
            config_defaults.get('whisper_model_size', 'large-v3-turbo')), key="whisper_model_select"
    )

    st.subheader("5. Szegmentálás")
    min_dur_ms_ui = st.number_input("Minimum szegmenshossz (ms):", min_value=500, max_value=30000,
                                    value=config_defaults.get('min_segment_duration_ms', 3000), step=100,
                                    key="min_duration_input")
    max_dur_ms_ui = st.number_input("Maximum szegmenshossz (ms):", min_value=1000, max_value=60000,
                                    value=config_defaults.get('max_segment_duration_ms', 13000), step=100,
                                    key="max_duration_input")

    st.subheader("6. Opcionális Funkciók")
    use_demucs_ui = st.checkbox("Vokál izoláció (Demucs)", value=config_defaults.get('use_demucs', True),
                                key="use_demucs_checkbox")
    use_diarization_ui = st.checkbox("Beszélők szétválasztása (Diarizáció)",
                                     value=config_defaults.get('use_diarization', True), key="use_diarization_checkbox")
    hf_token_ui = st.text_input("Hugging Face Token (diarizációhoz):",
                                value=config_defaults.get('huggingface_token', 'YOUR_HUGGINGFACE_TOKEN_HERE'),
                                type="password", key="hf_token_input")

    st.subheader("7. Minőség és Eszköz")
    confidence_threshold_ui = st.slider("Konfidencia küszöb (átirat ellenőrzéshez):", min_value=0.0, max_value=1.0,
                                        value=config_defaults.get('confidence_threshold', 0.7), step=0.05,
                                        key="confidence_slider")

    device_options = ["auto", "cpu"]
    default_device_index = 0
    if torch.cuda.is_available():
        device_options.append("cuda")
        saved_device = config_defaults.get('device', 'auto')
        if saved_device in device_options: default_device_index = device_options.index(saved_device)
    elif config_defaults.get('device', 'auto') == "cpu":
        default_device_index = device_options.index("cpu")
    device_ui = st.selectbox("Eszköz (CPU/GPU):", options=device_options, index=default_device_index,
                             key="device_select_box")

    if st.button("Jelenlegi beállítások mentése config.yaml-ba", key="save_config_sidebar_button"):
        # ... (mentési logika változatlan)
        current_config_for_save = {
            'input_video': input_path_ui,
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

# A végleges adathalmaz nevének meghatározása a widgetek aktuális állapotából
final_target_dataset_name = ""
if st.session_state.dataset_mode_radio_state_widget == "Új adathalmaz létrehozása":
    final_target_dataset_name = st.session_state.get("new_dataset_name_input", new_dataset_name_suggestion)
    # Ha a fallback input mező volt aktív (mert nem volt meglévő dataset)
    if not existing_datasets and "new_dataset_name_if_none_exist_for_existing_mode" in st.session_state:
        final_target_dataset_name = st.session_state.new_dataset_name_if_none_exist_for_existing_mode
elif existing_datasets and "existing_dataset_select_box" in st.session_state:  # Csak akkor, ha van mit kiválasztani
    final_target_dataset_name = st.session_state.existing_dataset_select_box
# Ha "Meglévő" van kiválasztva, de nincs meglévő (és a fallback inputot használjuk)
elif not existing_datasets and "new_dataset_name_if_none_exist_for_existing_mode" in st.session_state:
    final_target_dataset_name = st.session_state.new_dataset_name_if_none_exist_for_existing_mode

process_button_disabled = not final_target_dataset_name or not final_target_dataset_name.strip()
button_tooltip = "Kérlek, adj meg/válassz egy adathalmaz nevet az oldalsávon!" if process_button_disabled else "Indítsd el az összes videó feldolgozását a beállított adathalmazhoz."

if st.button("🚀 Teljes Feldolgozás Indítása", type="primary", use_container_width=True,
             key="start_processing_main_button", disabled=process_button_disabled, help=button_tooltip):
    # ... (a feldolgozási logika innen változatlan, ahogy az előző válaszban volt) ...
    actual_output_dataset_dir = os.path.join(base_output_for_datasets, final_target_dataset_name)
    os.makedirs(actual_output_dataset_dir, exist_ok=True)
    st.info(f"Kimenetek mentése a következő adathalmaz mappába: **{os.path.abspath(actual_output_dataset_dir)}**")

    runtime_config = {
        'output_base_dir': actual_output_dataset_dir,
        'output_raw_audio_filename': config_defaults.get('output_raw_audio_filename', 'audio_raw.wav'),
        'output_clean_audio_filename': config_defaults.get('output_clean_audio_filename', 'audio_clean.wav'),
        'output_segments_dirname': config_defaults.get('output_segments_dirname', 'segments'),
        'output_metadata_filename': config_defaults.get('output_metadata_filename', 'metadata.csv'),
        'target_sample_rate': target_sr_ui, 'language': language_ui,
        'whisper_model_size': whisper_model_size_ui,
        'min_segment_duration_ms': min_dur_ms_ui, 'max_segment_duration_ms': max_dur_ms_ui,
        'use_demucs': use_demucs_ui, 'use_diarization': use_diarization_ui,
        'huggingface_token': hf_token_ui if hf_token_ui and hf_token_ui.strip() and hf_token_ui != 'YOUR_HUGGINGFACE_TOKEN_HERE' else None,
        'confidence_threshold': confidence_threshold_ui, 'device': device_ui,
        'output_dir_structure': 'speaker_separated' if use_diarization_ui else 'flat'
    }

    media_to_process = [] # Változás: videos_to_process -> media_to_process
    temp_uploaded_paths = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            temp_media_path = os.path.join(temp_upload_main_dir, uploaded_file.name) # Változás
            with open(temp_media_path, "wb") as f: f.write(uploaded_file.getbuffer())
            media_to_process.append(temp_media_path) # Változás
            temp_uploaded_paths.append(temp_media_path)
        st.info(f"{len(uploaded_files)} fájl feltöltve feldolgozásra.")
    elif input_path_ui and input_path_ui.strip(): # Változás
        path_to_check = input_path_ui # Változás
        if not os.path.isabs(path_to_check): path_to_check = os.path.abspath(path_to_check)
        if os.path.isdir(path_to_check):
            st.info(f"Médiafájlok rekurzív keresése a '{path_to_check}' mappában és almappáiban...")
            supported_extensions = ('.mp4', '.mkv', '.mov', '.avi', '.webm', '.mp3', '.wav', '.flac', '.m4a')
            for root, dirs, files in os.walk(path_to_check):
                for filename in files:
                    if filename.lower().endswith(supported_extensions):
                        media_to_process.append(os.path.join(root, filename))
            
            if not media_to_process:
                st.warning(f"Nem találhatóak feldolgozható médiafájlok a '{path_to_check}' mappában és almappáiban.")
            else:
                st.info(f"{len(media_to_process)} fájl található a mappában és almappáiban.")
        elif os.path.isfile(path_to_check):
            media_to_process.append(path_to_check) # Változás
        else:
            st.error(f"A megadott elérési út nem létezik vagy nem támogatott: {path_to_check}"); st.stop()
    else:
        st.error("Nincs médiafájl kiválasztva vagy elérési út megadva."); st.stop() # Változás

    if not media_to_process: st.error("Nincsenek feldolgozandó fájlok."); st.stop() # Változás

    log_placeholder = st.empty()
    progress_bar_overall = st.progress(0, text="Teljes feldolgozás...")
    all_media_processed_successfully = True # Változás

    for media_index, current_media_path in enumerate(media_to_process): # Változás
        media_basename = os.path.basename(current_media_path) # Változás
        log_placeholder.info(f"Feldolgozás alatt: {media_basename} ({media_index + 1}/{len(media_to_process)})") # Változás

        safe_media_name = "".join(c if c.isalnum() else "_" for c in os.path.splitext(media_basename)[0]) # Változás
        media_processing_output_dir = os.path.join(actual_output_dataset_dir, safe_media_name) # Változás
        os.makedirs(media_processing_output_dir, exist_ok=True)

        media_config = runtime_config.copy() # Változás
        media_config['input_path'] = current_media_path # Változás
        media_config['output_base_dir'] = media_processing_output_dir # Változás
        media_config['output_metadata_file_absolute_path'] = os.path.join(actual_output_dataset_dir,
                                                                          runtime_config['output_metadata_filename'])
        media_config['segments_relative_path_prefix'] = safe_media_name # Változás

        st.info(f"Indul: {media_basename}") # Változás
        st.caption(f"Kimenetek (nyers, tiszta, szegmensek) ide: {os.path.abspath(media_config['output_base_dir'])}") # Változás

        raw_audio_file = None;
        clean_audio_file_for_this_media = None; # Változás
        input_for_transcription_this_media = None # Változás
        current_media_success = True # Változás

        with st.expander(f"[{safe_media_name}] 1. Audio előkészítése...", expanded=True): # Változás
            if not check_ffmpeg():
                st.error("FFmpeg nem található.");
                current_media_success = False # Változás
            if current_media_success: # Változás
                raw_audio_file = process_media_file(media_config) # Változás
                if raw_audio_file:
                    st.success(f"Nyers audio: {os.path.basename(raw_audio_file)}")
                else:
                    st.error("Audio előkészítés sikertelen."); current_media_success = False # Változás

        if current_media_success and media_config['use_demucs']: # Változás
            with st.expander(f"[{safe_media_name}] 2. Vokál izoláció (Demucs)...", expanded=True): # Változás
                clean_audio_file_for_this_media = isolate_vocals_with_demucs(media_config, raw_audio_file) # Változás
                if clean_audio_file_for_this_media:
                    st.success(f"Tiszta vokál: {os.path.basename(clean_audio_file_for_this_media)}")
                    input_for_transcription_this_media = clean_audio_file_for_this_media # Változás
                else:
                    st.warning("Demucs hiba. Nyers audió használata.")
                    input_for_transcription_this_media = raw_audio_file # Változás
        elif current_media_success: # Változás
            st.info(f"[{safe_media_name}] Demucs kihagyva.") # Változás
            input_for_transcription_this_media = raw_audio_file # Változás

        if current_media_success and input_for_transcription_this_media: # Változás
            with st.expander(f"[{safe_media_name}] 3. Transzkripció és szegmentálás...", expanded=True): # Változás
                st.write(f"WhisperX futtatása: {os.path.basename(input_for_transcription_this_media)}") # Változás
                transcribe_and_segment(media_config) # Változás
                st.success("Transzkripció és szegmentálás befejezve ehhez a fájlhoz!") # Változás
        elif current_media_success: # Változás
            st.error(f"[{safe_media_name}] Nincs audio fájl a transzkripcióhoz."); # Változás
            current_media_success = False # Változás

        if not current_media_success: # Változás
            all_media_processed_successfully = False # Változás

        progress_bar_overall.progress((media_index + 1) / len(media_to_process), # Változás
                                      text=f"Fájl {media_index + 1}/{len(media_to_process)} feldolgozva.") # Változás

    if temp_uploaded_paths:
        st.markdown("---");
        st.write("Ideiglenes feltöltött fájlok törlése...")
        for temp_path in temp_uploaded_paths:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path); st.caption(f"Törölve: {temp_path}")
                except Exception as e:
                    st.warning(f"Hiba törléskor ({temp_path}): {e}")
        if os.path.exists(temp_upload_main_dir) and not os.listdir(temp_upload_main_dir):
            try:
                shutil.rmtree(temp_upload_main_dir); st.caption(f"Ideiglenes mappa törölve: {temp_upload_main_dir}")
            except Exception as e:
                st.warning(f"Hiba az ideiglenes mappa törlésekor: {e}")

    st.markdown("---")
    if media_to_process:
        if all_media_processed_successfully:
            st.balloons()
            st.header("🎉 Feldolgozás Befejezve! 🎉")
            st.success(f"Az összes fájl feldolgozása sikeresen megtörtént.")
        else:
            st.warning(
                "Egy vagy több fájl feldolgozása során hiba történt. Ellenőrizd a fenti üzeneteket és a terminál logjait.")
        st.write(
            f"A közös metaadatok (átiratok) a `{os.path.abspath(os.path.join(actual_output_dataset_dir, runtime_config['output_metadata_filename']))}` fájlban találhatók.")
        st.write(
            f"A szegmentált audio fájlok a `{os.path.abspath(actual_output_dataset_dir)}` mappán belüli, videó nevű almappákban (`segments` almappákon belül) vannak.")
    else:
        st.info("Nem volt feldolgozandó videó.")

with st.sidebar:
    st.markdown("---")
    st.info("Fejlesztette: AI Asszisztens")
