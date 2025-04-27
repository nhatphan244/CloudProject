import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import io
import os
import re

def fetch_monster_data(monster_url, monster_name):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(monster_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    data = {
        "monster_name": monster_name,
        "url": monster_url,
        "hunter_notes": "",
        "classification": "",
        "sightings": "",
        "habitats": [],
        "attack_types": [],
        "weakest_to": [],
        "lore": "",
        "characteristics": "",
        "behavior": "",
        "useful_information": "",
        "ailments": {},
        "hitzones": {},
        "materials": {},
        "combat_tips": "",
    }
    
    # Extract classification
    classification_row = soup.find('th', string=lambda t: t and 'Classification' in t)
    if classification_row and classification_row.parent:
        classification_cell = classification_row.find_next('td')
        if classification_cell:
            data["classification"] = classification_cell.text.strip()
    
    # Extract attack types
    attack_types_row = soup.find('th', string=lambda t: t and 'Attack Types' in t)
    if attack_types_row and attack_types_row.parent:
        attack_cell = attack_types_row.find_next('td')
        if attack_cell:
            # Look for links within the cell which indicate attack types
            attack_links = attack_cell.find_all('a')
            for link in attack_links:
                if link.get('title'):
                    data["attack_types"].append(link.get('title'))
    
    # Extract weakest to
    weakest_row = soup.find('th', string=lambda t: t and 'Weakest To' in t)
    if weakest_row and weakest_row.parent:
        weakest_cell = weakest_row.find_next('td')
        if weakest_cell:
            # Look for links within the cell which indicate weaknesses
            weakness_links = weakest_cell.find_all('a')
            for link in weakness_links:
                if link.get('title'):
                    data["weakest_to"].append(link.get('title'))
    
    # Extract Hunter's Notes
    hunters_notes = soup.find('span', id="Hunter's_Notes")
    if hunters_notes and hunters_notes.parent:
        notes_table = hunters_notes.parent.find_next('table')
        if notes_table:            
            # Extract Characteristics
            char_section = notes_table.find('b', string=lambda t: t and ('Characteristics:' in t))
            if char_section:
                # Get text following the bold tag up to the next paragraph
                next_p = char_section.find_next('p')
                if next_p:
                    char_text = char_section.next_sibling
                    data["characteristics"] += char_text.strip() if char_text else ""

            # Extract Notes
            notes_section = notes_table.find('b', string=lambda t: t and ('Hunting:' in t or 'Threat:' in t or 'Threat Level:' in t or 'Hunters Notes:' in t))
            if notes_section and notes_section.parent:
                data["characteristics"] += notes_section.parent.get_text().replace('Hunting:', '').strip()
                notes_text = notes_section.parent.get_text().replace('Hunters Notes:', '').strip()
                rep = {"Hunting:": "",} # define desired replacements here
                # use these three lines to do the replacement
                rep = dict((re.escape(k), v) for k, v in rep.items()) 
                pattern = re.compile("|".join(rep.keys()))
                notes_text = pattern.sub(lambda m: rep[re.escape(m.group(0))], notes_text)
                data['hunter_notes'] = notes_text
            
            # Extract Useful Information
            info_section = notes_table.find('b', string=lambda t: t and ('Useful Information:' in t or 'Helpful Hints:' in t))
            if info_section and info_section.parent:
                info_text = info_section.parent.get_text().replace('Useful Information:', '').strip()
                info_text = info_section.parent.get_text().replace('Helpful Hints:', '').strip()
                data["useful_information"] = info_text
            
            # Extract Known Habitats
            habitat_section = notes_table.find('b', string=lambda t: t and ('Known Habitats:' in t))
            if habitat_section and habitat_section.parent:
                habitat_text = habitat_section.parent.get_text().replace('Known Habitats:', '').strip()
                data["habitats"].append(habitat_text)
    
    # Extract location information
    gameplay_section = soup.find('span', id='Gameplay_Information')
    if gameplay_section and gameplay_section.parent:
        # Find paragraphs following the Gameplay Information heading
        current = gameplay_section.parent
        paragraphs = []
        
        while current:
            current = current.find_next_sibling()
            if current and current.name == 'p':
                # Skip italicized instructional text
                if not current.find('i'):
                    paragraphs.append(current.text.strip())
            # Stop if we hit another heading
            elif current and current.name in ['h1', 'h2', 'h3', 'h4']:
                break
        
        if paragraphs:
            data["sightings"] = paragraphs[0]
            data["hunter_notes"] += "\n".join(paragraphs[1:])
    
    lore_section = soup.find('span', id='Ecology_&amp;_Lore')    
    if lore_section and lore_section.parent:
        # Find paragraphs following the Gameplay Information heading
        current = lore_section.parent
        paragraphs = []
        
        while current:
            current = current.find_next_sibling()
            if current and current.name == 'p':
                # Skip italicized instructional text
                if not current.find('i'):
                    paragraphs.append(current.text.strip())
            # Stop if we hit another heading
            elif current and current.name in ['h1', 'h2', 'h3', 'h4']:
                break
        
        if paragraphs:
            data["lore"] += "\n".join(paragraphs)
    
    physiology_section = soup.find('span', id='Physiology')    
    if physiology_section and physiology_section.parent:
        # Find paragraphs following the Gameplay Information heading
        current = physiology_section.parent
        paragraphs = []
        
        while current:
            current = current.find_next_sibling()
            if current and current.name == 'p':
                # Skip italicized instructional text
                if not current.find('i'):
                    paragraphs.append(current.text.strip())
            # Stop if we hit another heading
            elif current and current.name in ['h1', 'h2', 'h3', 'h4']:
                break
        
        if paragraphs:
            data["characteristics"] += "\n".join(paragraphs)
    
    behavior_section = soup.find('span', id='Behavior')    
    if behavior_section and behavior_section.parent:
        # Find paragraphs following the Gameplay Information heading
        current = behavior_section.parent
        paragraphs = []
        
        while current:
            current = current.find_next_sibling()
            if current and current.name == 'p':
                # Skip italicized instructional text
                if not current.find('i'):
                    paragraphs.append(current.text.strip())
            # Stop if we hit another heading
            elif current and current.name in ['h1', 'h2', 'h3', 'h4']:
                break
        
        if paragraphs:
            data["behavior"] = "\n".join(paragraphs)
    
    # Extract habitats
    habitats_header = soup.find('th', string=lambda t: t and 'Habitats' in t)
    if habitats_header and habitats_header.parent:
        habitats_row = habitats_header.parent.find_next_sibling()
        if habitats_row:
            habitats_cell = habitats_row.find('td')
            if habitats_cell:
                # Get all habitat links or text
                habitat_links = habitats_cell.find_all('a')
                if habitat_links:
                    data["habitats"] = [h.text.strip() for h in habitat_links]
                else:
                    data["habitats"] = [habitats_cell.text.strip()]
    
    
    # Find tables for weaknesses, ailments, hitzones, materials
    tables = soup.find_all('table')
    for table in tables:
        # Try to identify table by caption or heading above it
        caption = table.find('caption')
        caption_text = caption.text.strip().lower() if caption else ""
        
        # If no caption, try to find a heading before the table
        if not caption_text:
            prev_elem = table.find_previous(['h2', 'h3', 'h4', 'h5'])
            if prev_elem:
                caption_text = prev_elem.text.strip().lower()
        
        # Also check table headers
        headers = [th.text.strip().lower() for th in table.find_all('th')]
        headers_text = " ".join(headers)
        
        try:
            # Fix 1: Use StringIO to wrap the HTML string
            table_html = str(table)
            table_io = io.StringIO(table_html)
            df = pd.read_html(table_io)[0]
            
            # Fix 2: Ensure columns are always string type before converting to dict
            df.columns = df.columns.astype(str)
            
            # Classify the table
            if any(kw in caption_text or kw in headers_text for kw in ['weakness', 'element', 'fire', 'water', 'thunder']):
                data["weaknesses"] = df.to_dict(orient='records')
            
            elif any(kw in caption_text or kw in headers_text for kw in ['hitzone', 'damage', 'cut', 'impact']):
                data["hitzones"] = df.to_dict(orient='records')
            
            elif any(kw in caption_text or kw in headers_text for kw in ['ailment', 'status', 'poison', 'sleep']):
                data["ailments"] = df.to_dict(orient='records')
            
            elif any(kw in caption_text or kw in headers_text for kw in ['material', 'item', 'reward', 'carve', 'drop']):
                table_id = f"Table_{len(data['materials']) + 1}"
                data["materials"][table_id] = df.to_dict(orient='records')
        
        except Exception as e:
            print(f"Error processing table: {e}")
    
    return data

def save_to_json(monster_data, filename):
    # Make sure the file path is in the data directory
    filepath = os.path.join("./data/monsters_wiki", os.path.basename(filename))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(monster_data, f, indent=4, ensure_ascii=False)
    print(f"Data saved to {filepath}")

def __get_wikiinfo__(monster_name):
    url = f"https://monsterhunterwiki.org/wiki/{monster_name}"
    monster_data = fetch_monster_data(url, monster_name)
    
    return monster_data