import customtkinter as ctk
import os
import subprocess
import shutil
import pandas as pd
import numpy as np
import joblib
import requests
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from demoparser2 import DemoParser as dp
from tqdm import tqdm
import threading
import dearpygui.dearpygui as dpg
import webbrowser

#################
# A-AC by 4urxra
###############

version = ""
app_version = "1.0"

app_update = False

folder_path = os.path.join(os.environ['USERPROFILE'], 'a-ac')

if os.path.exists(os.path.join(folder_path, "scripts")):
    with open(os.path.join(folder_path, "version.txt"), "r") as f:
        version = f.read().strip()
else:
    # remove old folder
    shutil.rmtree(folder_path, ignore_errors=True)
    version = "0.0"

# Check for updates (get latest version from GitHub https://github.com/Dream23322/aurora-background/blob/main/other/latest.txt)
try:
    latest_version = subprocess.check_output(
        ["curl", "-s", "-k", "https://raw.githubusercontent.com/Dream23322/aurora-background/main/other/latest.txt"]
    ).decode("utf-8").strip()
    print("1")
    if latest_version != version:
        # Delete current folder, this forces redownload of latest version
        shutil.rmtree(os.path.join(os.environ['USERPROFILE'], 'a-ac'), ignore_errors=True)

    print("2")

    latest_app_version = subprocess.check_output(
        ["curl", "-s", "-k", "https://raw.githubusercontent.com/Dream23322/aurora-background/main/other/app-version.txt"]
    ).decode("utf-8").strip()

    print("3")

    if latest_app_version != app_version:
        app_update = True

except subprocess.CalledProcessError as e:
    print("Failed to check for updates. Please visit the GitHub page to download the latest version.\nCheck internet connection. \nError details:", e)

# Debug
print(f"Current AI Version: {version}"
      f"\nCurrent App Version: {app_version}"
      f"\nLatest AI Version: {latest_version}"
      f"\nLatest App Version: {latest_app_version}")

# Create folder named "a-ac" in user's home directory if it doesn't exist yet
folder_path = os.path.join(os.environ['USERPROFILE'], 'a-ac')
if not os.path.exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)

    os.makedirs(os.path.join(folder_path, "temp"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "temp/demo-holder"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "temp/parsed"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "temp/processed"), exist_ok=True)

    os.makedirs(os.path.join(folder_path, "model"), exist_ok=True)
    # Download model file
    subprocess.run(["curl", "-k", "https://raw.githubusercontent.com/Dream23322/aurora-background/main/model/aim-assist-model.h5", "-o", os.path.join(folder_path, "model/aim-assist-model.h5")])
    subprocess.run(["curl", "-k", "https://raw.githubusercontent.com/Dream23322/aurora-background/main/model/scaler.pkl", "-o", os.path.join(folder_path, "model/scaler.pkl")])

# Sanity check
if not os.path.exists(os.path.join(folder_path, "model/aim-assist-model.h5")) or not os.path.exists(os.path.join(folder_path, "model/scaler.pkl")):
    raise FileNotFoundError("Model files not found in the a-ac/model directory.")

# All done :D

# The background part for checking and all that fancy stuff
AACconfig = {
    "MODEL_PATH" : os.path.join(folder_path, "model/aim-assist-model.h5"),
    "SCALER_PATH" : os.path.join(folder_path, "model/scaler.pkl"),
    "INPUT_DIR" : os.path.join(folder_path, "temp/demo-holder"),
    "API_KEY" : "81F7CFCFAF256132B497EE4D4F655879",
    "CHEAT_THRESHOLD" : 0.5
}

class AuroraBackground():
    def __init__(self):
        self.config = AACconfig
        self.model = None
        self.scaler = None
        self.errors = ""

    def load(self):
        self.model = load_model(self.config["MODEL_PATH"])
        self.scaler = joblib.load(self.config["SCALER_PATH"])

    def parse_demos(self):
        parse_script = os.path.join(folder_path, "scripts/parser.py")
        process_script = os.path.join(folder_path, "scripts/processor.py")

        subprocess.run(["python", parse_script,
                        "--input", self.config["INPUT_DIR"],
                        "--output", os.path.join(folder_path, "temp/parsed")])

        subprocess.run(["python", process_script,
                        "--input", os.path.join(folder_path, "temp/parsed"),
                        "--output", os.path.join(folder_path, "temp/processed")])

    def refresh_folders(self):
        for folder in [os.path.join(folder_path, "temp/demo-holder"),
                       os.path.join(folder_path, "temp/parsed"),
                       os.path.join(folder_path, "temp/processed")]:
            if os.path.exists(folder):
                shutil.rmtree(folder)

    def get_steam_name(self, steamid):
        if not self.config["API_KEY"]:
            print("No Steam API key found")
            return "Unknown"
        try:
            params = {"key": self.config["API_KEY"], "steamids": steamid}
            r = requests.get("https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/", params=params, timeout=5)
            r.raise_for_status()
            players = r.json().get("response", {}).get("players", [])
            if players:
                return players[0].get("personaname")
        except Exception as e:
            self.errors += f"Error fetching Steam name for {steamid}: {str(e)}\n"
        return "Unknown"
    
    def copy_demo_to_folder(self, demo_path):
        dest_path = os.path.join(self.config["INPUT_DIR"], os.path.basename(demo_path))
        shutil.copy2(demo_path, dest_path)
    
    def demo_selector(self):
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        Tk().withdraw()
        try:
            demo_path = askopenfilename(title="Select a CS2 Demo", filetypes=[("CS2 Demos", "*.dem")], initialdir=r"\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\replays")
            self.copy_demo_to_folder(demo_path)
        except Exception as e:
            print(e)
            demo_path = askopenfilename(title="Select a CS2 Demo", filetypes=[("CS2 Demos", "*.dem")])
            self.copy_demo_to_folder(demo_path)

    def demo_exists(self):
        for f in os.listdir(self.config["INPUT_DIR"]):
            if f.endswith(".dem"):
                return True
        return False
    
    def predict_file(self, file_path):
        df = pd.read_csv(file_path)
        df = df.drop(columns=["tick","label"], errors="ignore")
        X = self.scaler.transform(df.values)

        X_lstm = X.reshape(1, X.shape[0], X.shape[1])
        
        y_pred = self.model.predict(X_lstm, verbose=0)
        mean_prob = float(np.mean(y_pred))
        suspicious = mean_prob > self.config["CHEAT_THRESHOLD"]
        return suspicious, mean_prob

    def check_demo_folder(self, input_dir):
        player_stats = {}

        for player_folder in os.listdir(input_dir):
            player_path = os.path.join(input_dir, player_folder)
            if not os.path.isdir(player_path):
                continue

            steamid = player_folder
            player_stats[steamid] = {"segments": 0, "suspicious": 0, "probs": []}

            for f in os.listdir(player_path):
                if not f.endswith(".csv"):
                    continue
                file_path = os.path.join(player_path, f)
                try:
                    suspicious, prob = self.predict_file(file_path)
                    player_stats[steamid]["segments"] += 1
                    player_stats[steamid]["probs"].append(prob)
                    player_stats[steamid]["suspicious"] += int(suspicious)

                    status = "[!] Suspicious" if suspicious else "[✓] Legit"
                    print(f"{f}: {status} ({prob*100:.2f}%)")
                except Exception as e:
                    print(f"[X] Failed to process {file_path}: {e}")

        # ===== OUTPUT CREATION =====
        output = FullResponse(error=(len(self.errors) not in [0, None]), error_msg=self.errors)
        for steamid, stats in player_stats.items():
            clean_id = steamid.replace("user_", "")
            steam_link = f"https://www.cs2guard.com/player/{clean_id}"
            avg_prob = np.mean(stats["probs"]) * 100 if stats["probs"] else 0

            output.add_player(PlayerOutput(
                steamid=steamid,
                steamname=self.get_steam_name(steamid),
                percent=avg_prob,
                segments=stats["segments"],
                amount=stats["suspicious"]
            ))

        return output

class PlayerOutput():
    def __init__(self, steamid, steamname, percent, segments, amount):
        self.steamid = steamid
        self.steamname = steamname
        self.percent = percent
        self.segments = segments
        self.amount = amount

class FullResponse():
    def __init__(self, error: bool, error_msg: str):
        self.error = error
        self.error_msg = error_msg
        self.players = []

    def add_player(self, player: PlayerOutput):
        self.players.append(player)

class Parser():
    def __init__(self):
        pass

    def euclidean_distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def parse_demo_folder(self, input_dir, cheater_ids, blacklist_ids, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for filename in tqdm(os.listdir(input_dir), desc="Parsing demos"):
            if not filename.endswith('.dem'):
                continue
            demo_path = os.path.join(input_dir, filename)
            demo_base = os.path.splitext(filename)[0]

            try:
                parser = dp(demo_path)
                events = parser.parse_event("player_death", player=["X", "Y", "Z", "pitch", "yaw", "steamid"])
                ticks_df = parser.parse_ticks(["tick", "steamid", "X", "Y", "Z", "pitch", "yaw"])

                for _, event in events.iterrows():
                    attacker = event.get("attacker_steamid")
                    victim = event.get("user_steamid")
                    tick = event["tick"]

                    if not attacker or not victim:
                        continue

                    if int(attacker) in blacklist_ids:
                        continue

                    attacker_int = int(attacker)
                    label = 1 if attacker_int in cheater_ids else 0

                    start_tick = tick - 300
                    end_tick = tick

                    # Slice attacker window
                    attacker_window = ticks_df[
                        ticks_df["tick"].between(start_tick, end_tick) &
                        (ticks_df["steamid"] == attacker_int)
                    ].drop_duplicates(subset="tick")

                    if attacker_window.empty:
                        continue

                    # Reindex for consistent 300-tick window
                    full_index = list(range(start_tick, end_tick))
                    attacker_window = (
                        attacker_window.set_index("tick")
                        .reindex(full_index)
                        .ffill()
                        .reset_index()
                        .rename(columns={"index": "tick"})
                    )

                    # Add label and metadata
                    attacker_window["steamid"] = attacker_int
                    attacker_window["label"] = label

                    # Weapon info
                    weapon = event.get("weapon", "unknown").lower()
                    attacker_window["weapon_name"] = weapon
                    attacker_window["weapon_type"] = self.map_weapon_group(weapon)

                    # Kill distance (Euclidean in X/Y)
                    dist = self.euclidean_distance(
                        event.get("attacker_X", 0),
                        event.get("attacker_Y", 0),
                        event.get("user_X", 0),
                        event.get("user_Y", 0)
                    )
                    attacker_window["kill_distance"] = dist

                    # Aim angle delta at kill (use last and second-last tick)
                    if attacker_window.shape[0] >= 2:
                        pitch_delta = (
                            attacker_window["pitch"].iloc[-1] -
                            attacker_window["pitch"].iloc[-2]
                        )
                        yaw_delta = (
                            attacker_window["yaw"].iloc[-1] -
                            attacker_window["yaw"].iloc[-2]
                        )
                    else:
                        pitch_delta = yaw_delta = 0

                    attacker_window["pitch_delta_at_kill"] = pitch_delta
                    attacker_window["yaw_delta_at_kill"] = yaw_delta

                    # Player speed: Euclidean distance between last two positions
                    if attacker_window.shape[0] >= 2:
                        dx = (
                            attacker_window["X"].iloc[-1] -
                            attacker_window["X"].iloc[-2]
                        )
                        dy = (
                            attacker_window["Y"].iloc[-1] -
                            attacker_window["Y"].iloc[-2]
                        )
                        dz = (
                            attacker_window["Z"].iloc[-1] -
                            attacker_window["Z"].iloc[-2]
                        )
                        speed = np.sqrt(dx**2 + dy**2 + dz**2)
                    else:
                        speed = 0

                    attacker_window["player_speed"] = speed

                    # Save
                    user_dir = os.path.join(output_dir, f"user_{attacker}")
                    os.makedirs(user_dir, exist_ok=True)

                    csv_name = f"{demo_base}_kill_{start_tick}_to_{end_tick}.csv"
                    csv_path = os.path.join(user_dir, csv_name)
                    attacker_window.to_csv(csv_path, index=False)

            except Exception as e:
                print(f"Failed to parse {filename}: {e}")

    def map_weapon_group(self, weapon_name):
        if not isinstance(weapon_name, str):
            return "unknown"
        weapon_name = weapon_name.lower()
        if any(w in weapon_name for w in ["deagle", "glock", "usp", "p250", "tec9", "cz75", "five", "revolver"]):
            return "pistol"
        elif any(w in weapon_name for w in ["ak47", "m4a", "galil", "famas", "aug", "sg", "scar", "bizon"]):
            return "rifle"
        elif any(w in weapon_name for w in ["awp", "ssg", "scout"]):
            return "sniper"
        elif any(w in weapon_name for w in ["ump", "mac", "mp7", "mp9", "mp5"]):
            return "smg"
        elif any(w in weapon_name for w in ["m249", "negev"]):
            return "lmg"
        elif any(w in weapon_name for w in ["nova", "xm", "mag", "sawedoff"]):
            return "shotgun"
        elif any(w in weapon_name for w in ["knife", "zeus"]):
            return "melee"
        else:
            return "unknown"
        
    def run(self):
        cheater_ids = {
    
        }

        blacklist_ids = {

        }

        output_dir = os.path.join(folder_path, "temp/parsed")

        input_dir = os.path.join(folder_path, "temp/demo-holder")
        print(f"Parsing folder: {input_dir}")
        self.parse_demo_folder(
            input_dir=input_dir,
            cheater_ids=cheater_ids,
            blacklist_ids=blacklist_ids,
            output_dir=output_dir
        )

        return True

class Processor():
    def __init__(self):
        # Fit a global encoder (shared across calls)
        self.weapon_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.WEAPON_GROUPS = ['pistol', 'rifle', 'sniper', 'smg', 'lmg', 'shotgun', 'melee', 'grenade', 'unknown']
        self.weapon_encoder.fit(np.array(self.WEAPON_GROUPS).reshape(-1, 1))

    def engineer_features(self, df):
        df = df.copy()

        if 'steamid' in df.columns:
            df = df.drop(columns=['steamid'])
            
        # Derivatives
        df['pitch_velocity'] = df['pitch'].diff() / df['tick'].diff()
        df['yaw_velocity'] = df['yaw'].diff() / df['tick'].diff()
        df['pitch_acceleration'] = df['pitch_velocity'].diff() / df['tick'].diff()
        df['yaw_acceleration'] = df['yaw_velocity'].diff() / df['tick'].diff()
        df['pitch_jerk'] = df['pitch_acceleration'].diff() / df['tick'].diff()
        df['yaw_jerk'] = df['yaw_acceleration'].diff() / df['tick'].diff()

        # Angular snap
        df['snap_magnitude'] = np.sqrt(df['pitch_delta_at_kill']**2 + df['yaw_delta_at_kill']**2)

        # Speed stability
        df['speed_rolling_std'] = df['player_speed'].rolling(window=10, min_periods=1).std()
        df['speed_rolling_mean'] = df['player_speed'].rolling(window=10, min_periods=1).mean()

        # Jumpiness
        df['position_delta'] = np.sqrt(df['X'].diff()**2 + df['Y'].diff()**2 + df['Z'].diff()**2)
        df['position_jumpiness'] = df['position_delta'].rolling(window=10, min_periods=1).std()

        # Angular motion
        df['cumulative_pitch'] = df['pitch'].cumsum()
        df['cumulative_yaw'] = df['yaw'].cumsum()
        df['angle_magnitude'] = np.sqrt(df['pitch'].diff()**2 + df['yaw'].diff()**2)

        # Flip detection
        df['yaw_change_sign'] = np.sign(df['yaw_velocity'].diff())
        df['pitch_change_sign'] = np.sign(df['pitch_velocity'].diff())
        df['direction_flips'] = (df['yaw_change_sign'].diff().abs() > 0).astype(int)
        df['flip_rate'] = df['direction_flips'].rolling(window=10).sum()

        # Rolling stats
        df['yaw_rolling_std'] = df['yaw'].rolling(window=10, min_periods=1).std()
        df['pitch_rolling_std'] = df['pitch'].rolling(window=10, min_periods=1).std()
        df['yaw_rolling_mean'] = df['yaw'].rolling(window=10, min_periods=1).mean()
        df['pitch_rolling_mean'] = df['pitch'].rolling(window=10, min_periods=1).mean()

        # Peak detection
        df['pitch_peaks'] = ((df['pitch_velocity'].diff().shift(-1) < 0) &
                            (df['pitch_velocity'].diff() > 0)).astype(int)
        df['yaw_peaks'] = ((df['yaw_velocity'].diff().shift(-1) < 0) &
                        (df['yaw_velocity'].diff() > 0)).astype(int)

        # Summary stats
        for col in ['pitch', 'yaw', 'pitch_velocity', 'yaw_velocity', 'angle_magnitude']:
            df[f'{col}_mean'] = df[col].mean()
            df[f'{col}_std'] = df[col].std()
            df[f'{col}_min'] = df[col].min()
            df[f'{col}_max'] = df[col].max()
            df[f'{col}_range'] = df[f'{col}_max'] - df[f'{col}_min']
            df[f'{col}_skew'] = df[col].skew()
            df[f'{col}_kurtosis'] = df[col].kurt()

        # Encode weapon_type
        if 'weapon_type' in df.columns:
            encoded_weapons = self.weapon_encoder.transform(df[['weapon_type']])
            encoded_df = pd.DataFrame(encoded_weapons, columns=[f'weapon_{cls}' for cls in self.weapon_encoder.categories_[0]])
            df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Drop string-based columns
        df = df.drop(columns=[col for col in ['name', 'weapon_name', 'weapon_type'] if col in df.columns])

        return df.dropna()

    def process_all_segments(self, input_dir, output_dir):
        for user_folder in os.listdir(input_dir):
            user_path = os.path.join(input_dir, user_folder)
            if not os.path.isdir(user_path):
                continue

            # Keep SteamID subfolder in output
            user_output_dir = os.path.join(output_dir, user_folder)
            os.makedirs(user_output_dir, exist_ok=True)

            for file_name in os.listdir(user_path):
                if not file_name.endswith('.csv'):
                    continue

                file_path = os.path.join(user_path, file_name)
                try:
                    df = pd.read_csv(file_path)
                    processed = self.engineer_features(df)

                    save_path = os.path.join(user_output_dir, f"engineered_{file_name}")
                    processed.to_csv(save_path, index=False)
                except Exception as e:
                    print(f"[!] Error processing {file_path}: {e}")

    def run(self):
        base_input = os.path.join(folder_path, "temp/parsed")
        base_output = os.path.join(folder_path, "temp/processed")

        print("Processing segments with SteamID folders retained...")
        self.process_all_segments(base_input, base_output)


# The UI and all that stuff
class AuroraApp():
    def __init__(self, background: AuroraBackground, parser: Parser, processor: Processor):
        super().__init__()

        self.background = background
        self.parser = parser
        self.processor = processor

        self.theme = (217, 156, 195)

        dpg.create_context()

        with dpg.theme() as container_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Tab, self.theme, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, self.theme, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, self.theme, category=dpg.mvThemeCat_Core),
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, self.theme, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, self.theme, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, self.theme, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, self.theme, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, self.theme, category=dpg.mvThemeCat_Core)

        dpg.create_viewport(title=f'A-AC [v{version}]', width=700, height=500)



        with dpg.window(tag="Primary Window"):
            dpg.bind_item_theme("Primary Window", container_theme)
            with dpg.tab_bar():
                with dpg.tab(label="AI Aim Assist"):
                    dpg.add_spacer(width=75)

                    dpg.add_text("Aurora Anti-Cheat")
                    dpg.add_text("by 4urxra and yviler", bullet=True)

                    if app_update:
                        dpg.add_spacer(height=10)
                        dpg.add_separator()
                        dpg.add_spacer(height=10)
                        dpg.add_text("A new version of Aurora Anti-Cheat is available!\nPlease visit the GitHub page to download the latest version.", color=(255, 0, 0))
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="GitHub User", callback=lambda: webbrowser.open("https://github.com/Dream23322/aurora-user/releases/latest"))

                    dpg.add_spacer(height=10)
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    dpg.add_text("Demo Selection:")

                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Select Demo", tag="select_demo_button", callback=lambda: self.select_demo())
                        self.has_file = dpg.add_text("Demo: No demo selected", tag="demo_status_text")

                    dpg.add_spacer(height=10)
                    dpg.add_separator()
                    dpg.add_spacer(height=10)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Run Analysis", tag="run_analysis_button", callback=lambda: threading.Thread(target=self.run).start())

                        self.progress_bar = dpg.add_progress_bar(tag="loading_bar", width=500, default_value=0.0, overlay="Progress")

                    dpg.add_spacer(height=5)
                    dpg.add_separator()

                    self.output_labels = []
                    # self.output_buttons = []

                    for i in range(10):
                        with dpg.group(horizontal=True):
                            self.label_id = dpg.add_text("")
                            self.output_labels.append(self.label_id)

                            # self.button_id = dpg.add_button(label="Profile", tag=f"profile_button_{i}", show=False)
                            # self.output_buttons.append(self.button_id)

                    dpg.add_spacer(height=10)

                    self.error_label = dpg.add_text("")

                    dpg.add_spacer(height=10)

                with dpg.tab(label="Classic Anti-Cheat"):
                    dpg.add_text("Classic style anti-cheat, looking for bad values, instead of using machine learning.")

                    

                with dpg.tab(label="About"):
                    dpg.add_text("Aurora Anti-Cheat (A-AC)")
                    dpg.add_text(f"Ai-Version: {version} | App Version: {app_version}")
                    dpg.add_text("Developed by 4urxra and yviler")
                    dpg.add_spacer(height=10)
                    dpg.add_text("A-AC is an advanced anti-cheat detection tool for Counter-Strike 2 demos, \nleveraging machine learning to identify potential aim-assist cheats.")
                    dpg.add_spacer(height=10)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="GitHub User", callback=lambda: webbrowser.open("https://github.com/Dream23322/aurora-user"))
                        dpg.add_button(label="GitHub Background", callback=lambda: webbrowser.open("https://github.com/Dream23322/aurora-background"))
                    dpg.add_spacer(height=10)
                    dpg.add_text("This software may produce false positives. Do not take this as absolute proof of cheating.")

        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 1)
                dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 20)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 1)
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, (107, 110, 248))
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (107, 110, 248))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (107, 110, 248))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (107, 110, 248))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (107, 110, 248))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (107, 110, 248))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (107, 110, 248))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (107, 110, 248))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (71, 71, 77))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (71, 71, 77))
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (107, 110, 248))
            
        dpg.bind_theme(global_theme)

        dpg.create_context()
        dpg.show_viewport()

        dpg.setup_dearpygui()
        dpg.set_primary_window("Primary Window", True)
        dpg.start_dearpygui()

    def select_demo(self):
        self.background.demo_selector()
        dpg.set_value(self.has_file, "Demo: Demo selected")

    def run(self):
        if not self.background.demo_exists():
            dpg.set_value(self.error_label, "Error: No demo file found in demo-holder folder, please select a valid demo!")
            print("No demo file found in demo-holder folder!")
            return
        
        dpg.set_value(self.progress_bar, 0.1)
        self.parser.run()
        dpg.set_value(self.progress_bar, 0.2)
        self.processor.run()
        dpg.set_value(self.progress_bar, 0.4)
        output = self.background.check_demo_folder(
            os.path.join(folder_path, "temp/processed")
        )

        dpg.set_value(self.progress_bar, 0.9)

        count = 0

        for player in output.players:
            # self.result_display.insert(ctk.END, f"{player.steamname} | {player.percent:.2f}%\n")
            # self.result_display.insert(ctk.END, f"> Data: SA: {player.segments} | SS: {player.amount} | SI: {player.steamid}\n")
            # self.result_display.insert(ctk.END, "----------------------------------------\n")

            dpg.set_value(self.output_labels[count], f"{player.steamname} | {player.percent:.2f}% | ID: {player.steamid.replace('user_', '')}")
            # dpg.configure_item(f"profile_button_{count}", show=True, callback=lambda u=f"{player.steamid}": webbrowser.open(f"https://www.cs2guard.com/player/{player.steamid.replace('user_', '')}"))
            count += 1

        dpg.set_value(self.progress_bar, 1.0)

if __name__ == "__main__":
    background = AuroraBackground()
    background.load()
    parser = Parser()
    processor = Processor()
    ui = AuroraApp(background, parser, processor)