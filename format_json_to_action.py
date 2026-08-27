import json
import re
import argparse

def convert_to_action_items(data):
    """
    Deeply analyzes the structure of the input data and extracts action segments,
    no matter how deeply nested they are in the JSON.
    """
    items = []
    
    def extract(node):
        if isinstance(node, dict):
            # 1. Frame-by-frame dict: {"1584": ..., "frame1585": {"1": "Action : 0.99"}}
            frame_mapping = {}
            for k, v in node.items():
                if isinstance(k, str):
                    # Matches "123", "frame123", "frame_123", "f123"
                    match = re.fullmatch(r'(?:frame_?|f)?(\d+)', k.lower())
                    if match:
                        frame_mapping[int(match.group(1))] = v
            
            # If a significant portion of the dictionary looks like frame mappings
            if len(frame_mapping) > 0 and len(frame_mapping) >= len(node) * 0.5:
                frames = sorted(frame_mapping.keys())
                start_f = last_f = frames[0]
                
                def get_val(f):
                    v = frame_mapping[f]
                    if isinstance(v, dict):
                        best_action = "Unknown"
                        best_score = -1.0
                        
                        parsed_any = False
                        for val in v.values():
                            if isinstance(val, str):
                                # Extract score from the end of the string (e.g., "Action : 0.99", "1: Action: .99")
                                score_match = re.search(r'([0-9]*\.[0-9]+|[0-9]+)\s*%?$', val.strip())
                                if score_match:
                                    try:
                                        score = float(score_match.group(1))
                                        # The action is everything before the score
                                        action_part = val[:score_match.start()].strip()
                                        # Clean up trailing colons/hyphens and leading numbers/bullets
                                        action_part = re.sub(r'[:\-]+$', '', action_part).strip()
                                        action_part = re.sub(r'^\d+\s*[:\-.]?\s*', '', action_part).strip()
                                        
                                        if action_part:
                                            parsed_any = True
                                            if score > best_score:
                                                best_score = score
                                                best_action = action_part
                                    except ValueError:
                                        pass
                        
                        if parsed_any and best_score >= 0:
                            return best_action, best_score

                        # Fallback for dicts like {"Jumping": 0.99, "Walking": 0.01}
                        try:
                            if all(isinstance(val, (int, float)) or (isinstance(val, str) and re.match(r'^[0-9]*\.?[0-9]+$', val.strip())) for val in v.values()):
                                best = max(v.items(), key=lambda x: float(x[1]))
                                return str(best[0]), float(best[1])
                        except:
                            pass

                        # Fallback for explicit keys
                        for key in ["label", "action_label", "class", "action", "prediction"]:
                            if key in v: return str(v[key]), float(v.get("score", v.get("confidence_score", 1.0)))
                            
                        return "Unknown", 1.0
                        
                    elif isinstance(v, list) and len(v) > 0:
                        if isinstance(v[0], dict):
                            for key in ["label", "action_label", "class", "action"]:
                                if key in v[0]: return str(v[0][key]), float(v[0].get("score", 1.0))
                        elif isinstance(v[0], (list, tuple)) and len(v[0]) >= 2:
                            return str(v[0][0]), float(v[0][1])
                        return str(v[0]), 1.0
                    elif isinstance(v, str):
                        score_match = re.search(r'([0-9]*\.[0-9]+|[0-9]+)\s*%?$', v.strip())
                        if score_match:
                            try:
                                score = float(score_match.group(1))
                                action_part = v[:score_match.start()].strip()
                                action_part = re.sub(r'[:\-]+$', '', action_part).strip()
                                action_part = re.sub(r'^\d+\s*[:\-.]?\s*', '', action_part).strip()
                                if action_part:
                                    return action_part, score
                            except ValueError:
                                pass
                        return str(v), 1.0
                    return str(v), 1.0

                curr_label, curr_score = get_val(start_f)
                scores = [curr_score]
                
                for f in frames[1:]:
                    label, score = get_val(f)
                    # Group if same label (ignoring frame gaps in case of sampled frames)
                    if label == curr_label:
                        last_f = f
                        scores.append(score)
                    else:
                        items.append({"action_label": curr_label, "start_frame": start_f, "end_frame": last_f, "confidence_score": sum(scores)/len(scores)})
                        curr_label, start_f, last_f, scores = label, f, f, [score]
                items.append({"action_label": curr_label, "start_frame": start_f, "end_frame": last_f, "confidence_score": sum(scores)/len(scores)})
                return # Stop recursing this specific node

            # 2. Explicit action item (has both a label and a time indicator)
            has_label = any(k in node for k in ["label", "action_label", "class", "action", "name", "prediction"])
            has_time = any(k in node for k in ["start", "start_frame", "segment", "timestamp", "start_time"])
            
            if has_label and has_time:
                label = node.get("action_label", node.get("label", node.get("class", node.get("action", node.get("name", node.get("prediction", "Unknown"))))))
                if "segment" in node and isinstance(node["segment"], list) and len(node["segment"]) >= 2:
                    sf, ef = node["segment"][0], node["segment"][1]
                else:
                    sf = node.get("start_frame", node.get("start", node.get("timestamp", node.get("start_time", 0))))
                    ef = node.get("end_frame", node.get("end", sf))
                score = node.get("confidence_score", node.get("score", node.get("confidence", 1.0)))
                items.append({"action_label": label, "start_frame": sf, "end_frame": ef, "confidence_score": score})

            # 3. Label -> Segments mapping (e.g., {"Squatting": [[10, 20], [30, 40]]})
            for k, v in node.items():
                if isinstance(v, list) and len(v) > 0:
                    if all(isinstance(x, list) and len(x) >= 2 for x in v):
                        for seg in v:
                            items.append({"action_label": k, "start_frame": seg[0], "end_frame": seg[1], "confidence_score": seg[2] if len(seg)>2 else 1.0})
                    elif len(v) >= 2 and all(isinstance(x, (int, float)) for x in v[:2]):
                        if k not in ["segment", "bbox", "resolution", "color", "size"]:
                            items.append({"action_label": k, "start_frame": v[0], "end_frame": v[1], "confidence_score": v[2] if len(v)>2 else 1.0})

            # Always recurse into values to find nested actions
            for v in node.values():
                extract(v)

        elif isinstance(node, list):
            # 4. List of lists (e.g., [["Squatting", 10, 20], ...])
            if len(node) > 0 and all(isinstance(x, list) and len(x) >= 3 and isinstance(x[0], str) for x in node):
                for seg in node:
                    items.append({"action_label": seg[0], "start_frame": seg[1], "end_frame": seg[2], "confidence_score": seg[3] if len(seg)>3 else 1.0})
                return

            # Recurse into list items
            for item in node:
                extract(item)

    # Start the deep extraction
    extract(data)
    
    # Filter out invalid items and enforce the strict list of possible actions using flexible matching
    def map_label(raw_label):
        raw = str(raw_label).lower().strip()
        if "stand" in raw: return "Stand"
        if "bend" in raw: return "Bending_Down"
        if "walk" in raw: return "Walking"
        if "rais" in raw or "hand" in raw: return "Raising_hand"
        if "squat" in raw: return "Squatting"
        if "run" in raw: return "Running"
        if "jump" in raw: return "Jumping"
        return None
    
    valid_items = []
    for item in items:
        mapped_label = map_label(item["action_label"])
        if mapped_label:
            try:
                item["action_label"] = mapped_label
                item["start_frame"] = int(item["start_frame"])
                item["end_frame"] = int(item["end_frame"])
                item["confidence_score"] = float(item["confidence_score"])
                valid_items.append(item)
            except (ValueError, TypeError):
                pass
                
    # Sort items by start_frame to ensure chronological order
    valid_items.sort(key=lambda x: x["start_frame"])
    
    # Post-process to merge consecutive items with the same label
    merged_items = []
    for item in valid_items:
        if not merged_items:
            merged_items.append(item)
        else:
            last_item = merged_items[-1]
            # Merge if same label and frames are consecutive or close (e.g., gap <= 60 frames)
            if last_item["action_label"] == item["action_label"] and (item["start_frame"] - last_item["end_frame"] <= 60):
                last_item["end_frame"] = max(last_item["end_frame"], item["end_frame"])
                last_item["confidence_score"] = (last_item["confidence_score"] + item["confidence_score"]) / 2.0
            else:
                merged_items.append(item)
                
    return merged_items

def format_to_action_structure(input_path, output_path, fps=30):
    try:
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
        return

    formatted_data = []
    
    # Convert raw data into standard action items
    raw_items = convert_to_action_items(raw_data)
    
    if not raw_items:
        print("Warning: No action items found. Please check the structure of your input JSON.")
    
    for item in raw_items:
        start_frame = item["start_frame"]
        end_frame = item["end_frame"]
        
        duration_frames = end_frame - start_frame + 1
        duration_seconds = round(duration_frames / fps, 2)
        timestamp = round(start_frame / fps, 2)
        
        formatted_item = {
            "action_label": item["action_label"],
            "timestamp": timestamp,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_frames": duration_frames,
            "duration_seconds": duration_seconds,
            "confidence_score": round(item["confidence_score"], 2)
        }
        formatted_data.append(formatted_item)

    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    print(f"Successfully formatted {len(formatted_data)} items and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and format JSON action data to standardized action items")
    parser.add_argument("input_file", help="Path to input JSON file")
    parser.add_argument("output_file", help="Path to output JSON file")
    parser.add_argument("--fps", type=float, default=30, help="Frames per second for timestamp calculation (default: 59.94)")
    
    args = parser.parse_args()
    
    format_to_action_structure(args.input_file, args.output_file, args.fps)
