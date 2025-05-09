import os
import yaml
import subprocess
import sys
import logging
import shutil # Fájlok mozgatásához/átnevezéséhez

# === Logger beállítása ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

# === Konfiguráció betöltése (ugyanaz, mint az extract_audio.py-ban) ===
def load_config(config_path='config.yaml'):
    """Betölti a konfigurációs fájlt."""
    if not os.path.exists(config_path):
        log.error(f"Hiba: A konfigurációs fájl nem található: {config_path}")
        print(f"\nHIBA: A konfigurációs fájl nem található itt: {os.path.abspath(config_path)}")
        print("Győződj meg róla, hogy a config.yaml ugyanabban a mappában van, ahonnan a szkriptet futtatod.")
        sys.exit(1)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        log.info(f"Konfiguráció betöltve innen: {config_path}")
        return config
    except yaml.YAMLError as e:
        log.error(f"Hiba a konfigurációs fájl olvasása közben: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Ismeretlen hiba a konfiguráció betöltésekor: {e}")
        sys.exit(1)

# === Fő funkció ===
def isolate_vocals_with_demucs(config, raw_audio_path):
    """
    Izolálja a vokált a nyers audio fájlból a Demucs segítségével, ha a konfigurációban
    engedélyezve van. A kimenetet a konfigurációban megadott helyre menti.
    """
    use_demucs = config.get('use_demucs', False)
    output_base_dir = config.get('output_base_dir', 'output')
    clean_audio_filename = config.get('output_clean_audio_filename', 'audio_clean.wav')
    output_clean_audio_path = os.path.join(output_base_dir, clean_audio_filename)

    # Ha a Demucs nincs engedélyezve, a "tiszta" audio ugyanaz, mint a nyers
    if not use_demucs:
        log.info("A Demucs használata ki van kapcsolva a konfigurációban. A nyers audio kerül felhasználásra tiszta audioként.")
        # Ha a nyers és a cél tiszta fájl elérési útja nem ugyanaz, másoljuk át
        if os.path.abspath(raw_audio_path) != os.path.abspath(output_clean_audio_path):
            try:
                shutil.copy2(raw_audio_path, output_clean_audio_path) # copy2 megőrzi a metaadatokat is
                log.info(f"Nyers audio átmásolva ide: {output_clean_audio_path}")
                return output_clean_audio_path
            except Exception as e:
                log.error(f"Hiba a nyers audio másolása közben ({raw_audio_path} -> {output_clean_audio_path}): {e}")
                return None
        else:
             # Ha ugyanaz a fájl, nincs teendő
             return raw_audio_path

    log.info(f"Demucs vokál izoláció indítása a fájlon: {raw_audio_path}")

    # Demucs kimeneti mappa meghatározása (a fő kimeneti mappán belül)
    # Célszerű egy külön mappát használni, mert a Demucs több fájlt is generálhat
    demucs_output_dir = os.path.join(output_base_dir, "separated")
    # A Demucs maga is létrehoz egy almappát a modell nevével, pl. "htdemucs"
    demucs_model_name = "htdemucs" # Ezt is lehetne konfigurálhatóvá tenni

    # Demucs parancs összeállítása (parancssori interfész használata javasolt a stabilitás miatt)
    # A '--two-stems vocals' opció csak a vokált és a többit (no_vocals) választja szét.
    demucs_command = [
        sys.executable, # A jelenlegi Python interpretert használjuk a demucs futtatásához
        '-m', 'demucs',
        '--two-stems', 'vocals',
        '-n', demucs_model_name,
        '-o', demucs_output_dir, # Kimeneti gyökérmappa
        raw_audio_path
    ]
    # GPU/CPU kezelése: A demucs parancssori eszköz automatikusan kezeli (ha a torch helyesen van telepítve)
    # Nem kell explicit device flaget hozzáadni itt általában.

    try:
        log.info(f"Demucs parancs futtatása: {' '.join(demucs_command)}")
        # Demucs futtatása alfolyamatként
        process = subprocess.run(demucs_command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        log.info("Demucs sikeresen lefutott.")
        log.debug(f"Demucs kimenet (stdout):\n{process.stdout}")
        if process.stderr:
            log.debug(f"Demucs kimenet (stderr):\n{process.stderr}")

        # A várt kimeneti vokál fájl elérési útjának meghatározása
        # A Demucs általában a <kimeneti_mappa>/<modell_neve>/<eredeti_fájlnév_kiterj.nélkül>/vocals.wav struktúrát hozza létre
        raw_audio_basename = os.path.splitext(os.path.basename(raw_audio_path))[0]
        expected_vocals_path = os.path.join(demucs_output_dir, demucs_model_name, raw_audio_basename, "vocals.wav")

        if os.path.exists(expected_vocals_path):
            log.info(f"Demucs által generált vokál fájl megtalálva: {expected_vocals_path}")
            # Vokál fájl átnevezése/mozgatása a végleges helyére
            shutil.move(expected_vocals_path, output_clean_audio_path)
            log.info(f"Tiszta vokál sikeresen elmentve ide: {os.path.abspath(output_clean_audio_path)}")

            # Opcionális: A Demucs által létrehozott többi fájl és mappa törlése
            try:
                demucs_specific_output_folder = os.path.join(demucs_output_dir, demucs_model_name, raw_audio_basename)
                if os.path.exists(demucs_specific_output_folder):
                     shutil.rmtree(demucs_specific_output_folder)
                     log.info(f"Demucs ideiglenes mappa törölve: {demucs_specific_output_folder}")
                # Ha a 'separated/htdemucs' mappa üres, azt is törölhetjük
                model_output_folder = os.path.join(demucs_output_dir, demucs_model_name)
                if os.path.exists(model_output_folder) and not os.listdir(model_output_folder):
                    os.rmdir(model_output_folder)
                # Ha a 'separated' mappa üres, azt is törölhetjük
                if os.path.exists(demucs_output_dir) and not os.listdir(demucs_output_dir):
                     os.rmdir(demucs_output_dir)
            except Exception as cleanup_err:
                log.warning(f"Hiba a Demucs ideiglenes fájlok törlése közben: {cleanup_err}")

            return output_clean_audio_path
        else:
            log.error(f"Hiba: A Demucs lefutott, de a várt vokál fájl nem található itt: {expected_vocals_path}")
            return None

    except subprocess.CalledProcessError as e:
        log.error(f"Hiba történt a Demucs futtatása közben:")
        log.error(f"Parancs: {' '.join(e.cmd)}")
        log.error(f"Visszatérési kód: {e.returncode}")
        log.error(f"Demucs hibaüzenet (stdout):\n{e.stdout}")
        log.error(f"Demucs hibaüzenet (stderr):\n{e.stderr}")
        print("\nHIBA: A Demucs nem futott le sikeresen. Ellenőrizd a fenti logokat a részletekért.")
        print("Lehetséges okok: Helytelen PyTorch/CUDA telepítés, kevés memória/VRAM, vagy a Demucs modell letöltési hibája.")
        return None
    except FileNotFoundError:
         log.error(f"Hiba: A '{sys.executable} -m demucs' parancs nem található. Biztosan telepítve van a 'demucs' csomag a '{sys.executable}' Python környezetbe?")
         print("\nHIBA: A demucs parancs nem található. Telepítsd a `pip install demucs` paranccsal az aktív Conda környezetbe.")
         return None
    except Exception as e:
        log.error(f"Váratlan hiba a vokál izoláció közben: {e}")
        return None

# === Szkript belépési pontja ===
if __name__ == "__main__":
    log.info("=== Vokál Izoláló Szkript Indítása ===")
    config = load_config()
    if config:
        output_base_dir = config.get('output_base_dir', 'output')
        raw_audio_filename = config.get('output_raw_audio_filename', 'audio_raw.wav')
        raw_audio_path = os.path.join(output_base_dir, raw_audio_filename)

        if not os.path.exists(raw_audio_path):
            log.error(f"Hiba: A bemeneti nyers audio fájl nem található: {raw_audio_path}")
            log.error("Futtasd először az 'extract_audio.py' szkriptet.")
            sys.exit(1)

        clean_audio_path = isolate_vocals_with_demucs(config, raw_audio_path)

        if clean_audio_path:
            log.info(f"=== Vokál izoláció sikeresen befejezve (vagy kihagyva). Kimeneti fájl: {clean_audio_path} ===")
        else:
            log.error("=== Vokál izolációs folyamat sikertelen. ===")
    else:
        log.error("=== Konfiguráció betöltése sikertelen, a szkript leáll. ===")