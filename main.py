# neuro_path_pipeline.py
"""
NeuroPath Wandering Detection System
Includes: GeoLife Parser, Stop Detection, Leader Stream Clustering (Infinite Memory),
Place Registry Noise Filtering, Markov + LSTM Models, Multi-Vector Anomaly Detection,
Interactive HTML Mapping, and Summary Statistics.
"""

import os
import warnings

# Silence TensorFlow and Keras warnings to keep the caretaker dashboard clean
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import folium
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.utils import to_categorical
from collections import defaultdict
import datetime


# -------------------------------
# 0. GeoLife Data Parser
# -------------------------------
def parse_geolife_data(file_content):
    """Parses the standard GeoLife PLT format into (lat, lon, timestamp_seconds)."""
    gps_data = []
    lines = file_content.strip().split('\n')

    for line in lines[6:]:  # Skip 6 lines of headers
        parts = line.split(',')
        if len(parts) >= 7:
            lat, lon = float(parts[0]), float(parts[1])
            date_str, time_str = parts[5], parts[6]
            dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            gps_data.append((lat, lon, dt.timestamp()))

    return gps_data


# -------------------------------
# 1. Stop Detection
# -------------------------------
def detect_stops(gps_data, dist_thresh=0.001, time_thresh=2 * 60):
    """Filters noisy GPS data into meaningful stays (Stops)."""
    stops = []
    i = 0
    while i < len(gps_data):
        lat_i, lon_i, t_i = gps_data[i]
        j = i + 1
        while j < len(gps_data):
            lat_j, lon_j, t_j = gps_data[j]
            dist = np.sqrt((lat_j - lat_i) ** 2 + (lon_j - lon_i) ** 2)
            if dist > dist_thresh:
                break
            j += 1
        duration = gps_data[j - 1][2] - t_i
        if duration >= time_thresh:
            centroid_lat = np.mean([gps_data[k][0] for k in range(i, j)])
            centroid_lon = np.mean([gps_data[k][1] for k in range(i, j)])
            readable_time = datetime.datetime.fromtimestamp(t_i).strftime('%H:%M:%S')

            # Determine time of day shift for temporal anomaly detection
            hour = datetime.datetime.fromtimestamp(t_i).hour
            if 5 <= hour < 12:
                time_shift = "Morning"
            elif 12 <= hour < 17:
                time_shift = "Afternoon"
            elif 17 <= hour < 22:
                time_shift = "Evening"
            else:
                time_shift = "Night"

            stops.append({
                'lat': centroid_lat, 'lon': centroid_lon,
                'start': t_i, 'time_str': readable_time, 'time_shift': time_shift
            })
        i = j
    return stops


# -------------------------------
# 2. Leader Clustering & Place Registry
# -------------------------------
class LeaderClusterer:
    """Infinite memory stream clustering based on distance radius."""

    def __init__(self, radius_threshold=0.0025):
        self.radius = radius_threshold
        self.places = {}
        self.next_id = 0

    def update(self, stop):
        lat, lon = stop['lat'], stop['lon']
        nearest_cid = None
        min_dist = float('inf')

        # Find the closest existing cluster
        for cid, coords in self.places.items():
            dist = np.sqrt((coords['lat'] - lat) ** 2 + (coords['lon'] - lon) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_cid = cid

        # Group into existing cluster if within radius
        if nearest_cid is not None and min_dist <= self.radius:
            place = self.places[nearest_cid]
            weight = place['weight']
            # Pull centroid slightly toward the new stop
            place['lat'] = (place['lat'] * weight + lat) / (weight + 1)
            place['lon'] = (place['lon'] * weight + lon) / (weight + 1)
            place['weight'] += 1
            return nearest_cid, False

        # Spawn a brand new cluster if outside all known radii
        else:
            new_cid = self.next_id
            self.places[new_cid] = {'lat': lat, 'lon': lon, 'weight': 1}
            self.next_id += 1
            return new_cid, True


class PlaceRegistry:
    """Filters noisy raw clusters into stable places based on visit frequency."""

    def __init__(self, stability_threshold=5, merge_radius=0.0025):
        self.places = {}  # place_id -> info
        self.cluster_to_place = {}
        self.next_place_id = 0
        self.stability_threshold = stability_threshold
        self.merge_radius = merge_radius

        self.cluster_visit_counts = defaultdict(int)

    def update(self, cluster_id, stop):
        lat, lon = stop['lat'], stop['lon']

        # Track how stable this cluster is
        self.cluster_visit_counts[cluster_id] += 1

        # If cluster already mapped -> return existing place
        if cluster_id in self.cluster_to_place:
            place_id = self.cluster_to_place[cluster_id]
            self._update_place(place_id, lat, lon)
            return place_id, False

        # Only promote to place if stable enough
        if self.cluster_visit_counts[cluster_id] < self.stability_threshold:
            return None, False

        # Try to merge with existing place
        for pid, place in self.places.items():
            dist = np.sqrt((place['lat'] - lat) ** 2 + (place['lon'] - lon) ** 2)
            if dist < self.merge_radius:
                self.cluster_to_place[cluster_id] = pid
                self._update_place(pid, lat, lon)
                return pid, False

        # Otherwise create new place
        pid = self.next_place_id
        self.places[pid] = {'lat': lat, 'lon': lon, 'visits': 1}
        self.cluster_to_place[cluster_id] = pid
        self.next_place_id += 1

        return pid, True

    def _update_place(self, pid, lat, lon):
        place = self.places[pid]
        w = place['visits']
        place['lat'] = (place['lat'] * w + lat) / (w + 1)
        place['lon'] = (place['lon'] * w + lon) / (w + 1)
        place['visits'] += 1


# -------------------------------
# 3. Sequence Generation
# -------------------------------
def generate_sequences(cluster_ids, n_steps=3):
    X, y = [], []
    for i in range(len(cluster_ids) - n_steps):
        X.append(cluster_ids[i:i + n_steps])
        y.append(cluster_ids[i + n_steps])
    return np.array(X), np.array(y)


# -------------------------------
# 4. Markov Chain (Route Baseline)
# -------------------------------
class MarkovModel:
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.probs = {}

    def update(self, sequence):
        for i in range(len(sequence) - 1):
            curr, nxt = sequence[i], sequence[i + 1]
            self.transitions[curr][nxt] += 1

    def normalize(self):
        for curr, nxt_dict in self.transitions.items():
            total = sum(nxt_dict.values())
            self.probs[curr] = {k: v / total for k, v in nxt_dict.items()}

    def check_anomaly(self, current_place, actual_next, threshold=0.05):
        if current_place not in self.probs:
            return True, 0.0
        prob = self.probs[current_place].get(actual_next, 0.0)
        return (prob < threshold), prob


# -------------------------------
# 5. LSTM (Route Prediction)
# -------------------------------
def train_lstm(cluster_ids, n_steps=2, epochs=30):
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(cluster_ids)

    if len(encoded) <= n_steps: return None, None

    X, y = generate_sequences(encoded, n_steps)
    if len(X) == 0: return None, None

    y = to_categorical(y, num_classes=len(set(encoded)))
    X = np.array(X).reshape((X.shape[0], X.shape[1]))

    model = Sequential([
        Embedding(input_dim=len(set(encoded)), output_dim=16),
        LSTM(32),
        Dense(len(set(encoded)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, verbose=0)
    return model, encoder


def check_lstm_anomaly(model, encoder, sequence, actual_next, threshold=0.05):
    try:
        encoded_seq = encoder.transform(sequence)
        encoded_actual = encoder.transform([actual_next])[0]
    except ValueError:
        return True, 0.0

    encoded_seq = np.array(encoded_seq).reshape((1, len(sequence)))
    probs = model.predict(encoded_seq, verbose=0)[0]

    actual_prob = probs[encoded_actual]
    return (actual_prob < threshold), actual_prob


# -------------------------------
# 6. Map Visualization
# -------------------------------
def plot_clusters_on_map(stops, cluster_ids, output_filename="wandering_map.html"):
    """Generates an interactive HTML map with stops connected chronologically."""
    if not stops:
        return

    avg_lat = sum(stop['lat'] for stop in stops) / len(stops)
    avg_lon = sum(stop['lon'] for stop in stops) / len(stops)

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=14, tiles="CartoDB positron")

    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'darkred',
        'darkblue', 'darkgreen', 'cadetblue', 'darkpurple',
        'pink', 'lightblue', 'lightgreen', 'black'
    ]

    print("\n--- Generating Map ---")

    # Draw path
    path_coords = [[stop['lat'], stop['lon']] for stop in stops]
    folium.PolyLine(path_coords, color="gray", weight=2, opacity=0.5, dash_array="5, 5").add_to(m)

    # Add markers
    for stop, cid in zip(stops, cluster_ids):
        marker_color = colors[cid % len(colors)]
        folium.CircleMarker(
            location=[stop['lat'], stop['lon']],
            radius=8,
            popup=f"<b>Place {cid}</b><br>Time: {stop['time_str']}",
            tooltip=f"Place {cid}",
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.8
        ).add_to(m)

    m.save(output_filename)
    print(f"Map saved successfully! Open '{output_filename}' in your web browser.")


# -------------------------------
# Execution: Caretaker Dashboard Simulation
# -------------------------------
if __name__ == "__main__":
    try:
        with open('combined.plt', 'r') as f:
            raw_file_content = f.read()
    except FileNotFoundError:
        print("Error: 'combined.plt' not found.")
        exit()

    print("--- Parsing GPS Stream ---")
    gps_data = parse_geolife_data(raw_file_content)
    all_stops = detect_stops(gps_data)

    if len(all_stops) == 0:
        print("No stops detected.")
        exit()

    print("\n--- Initializing Caretaker Wandering Detection ---")
    # Both stages use the 0.0025 threshold you requested
    leader_clusterer = LeaderClusterer(radius_threshold=0.0025)
    registry = PlaceRegistry(stability_threshold=2, merge_radius=0.0025)
    markov = MarkovModel()

    history_buffer = []
    visit_counts = defaultdict(int)
    place_time_profiles = defaultdict(set)

    stable_stops = []
    assigned_cluster_ids = []

    lstm_model, encoder = None, None
    N_STEPS = 2
    BURN_IN_PERIOD = 1

    # Initialize statistics tracker
    stats = {
        'total_raw_stops': len(all_stops),
        'stable_stops_processed': 0,
        'spatial_anomalies': 0,
        'destination_anomalies': 0,
        'temporal_anomalies': 0,
        'route_markov_anomalies': 0,
        'route_lstm_anomalies': 0
    }

    for idx, stop in enumerate(all_stops):
        time_str = stop['time_str']
        time_shift = stop['time_shift']
        print(f"\n[{time_str}] User stopped at Lat: {stop['lat']:.5f}, Lon: {stop['lon']:.5f}")

        # Process through Leader Clusterer and Place Registry
        raw_cluster_id, _ = leader_clusterer.update(stop)
        cid, is_new_place = registry.update(raw_cluster_id, stop)

        # Noise Filter: Skip anomaly checks and mapping if the place isn't stable yet
        if cid is None:
            print(f"  -> [Noise Filter] Gathering stability data. Not yet recognized as a stable place.")
            continue

        # If stable, append to our active tracking lists
        stats['stable_stops_processed'] += 1
        stable_stops.append(stop)
        assigned_cluster_ids.append(cid)
        is_training_phase = len(stable_stops) <= BURN_IN_PERIOD

        if is_training_phase:
            print(f"  -> [Warm-up] Observing baseline. Mapped Place {cid} during {time_shift}.")
        else:
            # 1. SPATIAL CHECK
            if is_new_place:
                stats['spatial_anomalies'] += 1
                print(f"  -> [ALERT: SPATIAL] User wandered to a completely unknown location! (Place {cid})")
            else:
                print(f"  -> [Location] Recognized known place: Place {cid}")

                # 2. DESTINATION CHECK
                total_visits = max(1, sum(visit_counts.values()))
                destination_prob = visit_counts[cid] / total_visits
                if destination_prob < 0.05:
                    stats['destination_anomalies'] += 1
                    print(
                        f"  -> [WARNING: DESTINATION] Rare destination. User only goes here {destination_prob:.1%} of the time.")

                # 3. TEMPORAL CHECK
                if time_shift not in place_time_profiles[cid]:
                    stats['temporal_anomalies'] += 1
                    print(
                        f"  -> [ALERT: TEMPORAL] User has NEVER been to Place {cid} during the {time_shift} before!")

            # 4. SEQUENTIAL CHECK
            if len(history_buffer) >= N_STEPS:
                current_seq = history_buffer[-N_STEPS:]
                curr_place = history_buffer[-1]

                is_m_anom, m_prob = markov.check_anomaly(curr_place, cid)
                if is_m_anom:
                    stats['route_markov_anomalies'] += 1
                    print(f"  -> [WARNING: ROUTE (Markov)] Unlikely path taken. Probability: {m_prob:.2%}")

                if lstm_model is not None:
                    is_l_anom, l_prob = check_lstm_anomaly(lstm_model, encoder, current_seq, cid)
                    if is_l_anom:
                        stats['route_lstm_anomalies'] += 1
                        print(f"  -> [WARNING: ROUTE (LSTM)] Unlikely path taken. Probability: {l_prob:.2%}")

        # --- Update System Memory ---
        history_buffer.append(cid)
        visit_counts[cid] += 1
        place_time_profiles[cid].add(time_shift)

        if len(history_buffer) >= 2:
            markov.update(history_buffer[-2:])
            markov.normalize()

        if len(history_buffer) >= N_STEPS + 1 and (
                len(stable_stops) == BURN_IN_PERIOD + 1 or len(stable_stops) % 4 == 0):
            lstm_model, encoder = train_lstm(history_buffer, n_steps=N_STEPS, epochs=20)

    # --- Print Summary Statistics ---
    print("\n" + "=" * 50)
    print("      WANDERING DETECTION SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total Raw Stops Detected:          {stats['total_raw_stops']}")
    print(f"Stable Stops Processed:            {stats['stable_stops_processed']}")
    print(f"Unique Stable Places Identified:   {len(registry.places)}")
    print("-" * 50)
    print("ANOMALY BREAKDOWN:")
    print(f"Spatial Alerts (New Places):       {stats['spatial_anomalies']}")
    print(f"Destination Warnings (Rare):       {stats['destination_anomalies']}")
    print(f"Temporal Alerts (Odd Hours):       {stats['temporal_anomalies']}")
    print(f"Route Warnings (Markov Chain):     {stats['route_markov_anomalies']}")
    print(f"Route Warnings (LSTM Predict):     {stats['route_lstm_anomalies']}")
    print("=" * 50)

    # --- Generate Interactive Map using only stable stops ---
    plot_clusters_on_map(stable_stops, assigned_cluster_ids)
