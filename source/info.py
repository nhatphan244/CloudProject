import os
import json
import re

def get_monster_info(monster_name):
    base_path = os.path.dirname(__file__)
    json_path = os.path.join(base_path, 'monsters_info.json')

    normalized_name = monster_name.strip().lower().replace('_', ' ')

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for name, info in data.items():
        json_normalized_name = name.strip().lower().replace('_', ' ')

        if json_normalized_name == normalized_name:
            cleaned_info = {}
            for key, value in info.items():
                if isinstance(value, str):
                    cleaned_info[key] = re.sub(r'\[\d+\]', '', value).strip()
                else:
                    cleaned_info[key] = value

            cleaned_name = name.replace('_', ' ').strip()
            cleaned_info['Monster Name'] = cleaned_name

            return cleaned_info

    return {"error": f"Monster '{monster_name}' not found."}
