import os
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote
import urllib.request
from img_crawler import __get_image__
from wiki_crawler import __get_wikiinfo__
import re
from icrawler.builtin import GoogleImageCrawler

class MonsterHunterScraper:
    def __init__(self, debug=True):
        self.base_url = "https://monsterhunterwiki.org"
        self.monsters_info = {}
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.debug = debug
        # Create data directory if it doesn't exist
        self.data_dir = "./data"
        self.debug_file_path = "./data/html_stocks"
        os.makedirs(self.data_dir, exist_ok=True)
        self.log_file = open(os.path.join(self.data_dir, "scraper_log.txt"), "w", encoding="utf-8")
        
    def log(self, message):
        """Print and log a message if debugging is enabled."""
        if self.debug:
            print(message)
            self.log_file.write(f"{message}\n")
            self.log_file.flush()
    
    def get_all_monster_names(self):
        """Scrape the list of all monster names from the Monster Hunter Wiki."""
        url = f"{self.base_url}/wiki/Monster_List#Large_Monster"
        self.log(f"Fetching monster list from {url}")
        
        try:
            response = requests.get(url, headers=self.headers)
            self.log(f"Response status code: {response.status_code}")
            
            # Save the HTML for debugging
            debug_file_path = os.path.join(self.debug_file_path, "monster_list_page.html")
            with open(debug_file_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            self.log(f"Saved monster list HTML to {debug_file_path}")
                    
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Try multiple selectors to find monster names
            monster_names = []
            
            # Method 1: Try to find tables with monster information
            monster_tables = soup.find_all("table", class_="wikitable")
            for table in monster_tables:
                for row in table.find_all("tr")[1:]:  # Skip header row
                    cells = row.find_all("td")
                    if cells and len(cells) > 0:
                        links = cells[0].find_all("a")
                        if links:
                            monster_name = links[0].get_text().strip()
                            if monster_name and monster_name not in monster_names:
                                monster_names.append(monster_name)
            
            # Method 2: Find all links that might be monster pages
            if not monster_names:
                self.log("Method 1 failed, trying Method 2...")
                all_links = soup.find_all("a")
                for link in all_links:
                    href = link.get("href", "")
                    if "/wiki/" in href and "Category:" not in href and "Template:" not in href:
                        monster_name = link.get_text().strip()
                        if monster_name and len(monster_name) > 2 and monster_name not in monster_names:
                            monster_names.append(monster_name)
            
            # Manually add some known monsters as a fallback
            if not monster_names:
                self.log("No monsters found via HTML parsing, adding known monsters...")
                monster_names = [
                    "Rathalos", "Rathian", "Zinogre", "Nargacuga", "Tigrex", 
                    "Diablos", "Rajang", "Gore Magala", "Brachydios", "Lagiacrus"
                ]
            
            non_monster_terms = [
                # Wiki/Website Structure
                "Monster Hunter Wiki", "Page", "Read", "Category", "Main page", "Recent changes",
                "Random page", "Editing Help", "Help about MediaWiki", "Special pages",
                "What links here", "Related changes", "Privacy policy", "About Monster Hunter Wiki", "Disclaimers", "Monster List",

                # Game Titles / Versions
                "Main Series games", "Stories generation", "Frontier", "Monster List (Spinoffs)",
                "Monster Hunter Wilds", "MHWilds", "MHNow", "MHRise", "MHRS", "MHPuzzles", "MH4", "MH2",
                "MHGU", "MHST2", "MH4U", "MHGen", "MH1", "MHF2", "MH3", "MHP3", "MHG", "MHFrontier", "MHGU",
                "MHFU", "MHWorld", "MHOutlanders", "MHWI", "MH3U", "MHF1", "MHOnline", "Game List",
                "Monster Hunter Now", "Monster Hunter Rise Sunbreak", "Monster Hunter World Iceborne",

                # Environmental Life / Categories / Misc.
                "Amphibian", "Endemic Life", "Relicts", "Bird Wyvern", "Brute Wyvern", "Carapaceon", "Cephalopod",
                "Construct", "Demi Elder", "Elder Dragon", "Fanged Beast", "Fanged Wyvern",
                "Flying Wyvern", "Leviathan", "Lynian", "Neopteron", "Piscine Wyvern", "Relict",
                "Snake Wyvern", "Temnoceran", "???", "Fish", "Herbivore", "Wingdrake",
            ]

            
            non_monster_set = set(non_monster_terms)
            monster_names = [term for term in monster_names if term not in non_monster_set]
            
            print(monster_names)
            self.log(f"Found {len(monster_names)} monsters: {monster_names[:10]}...")
            return monster_names
            
        except Exception as e:
            self.log(f"Error getting monster list: {e}")
            # Return some known monsters as a fallback
            return ["Rathalos", "Rathian", "Zinogre", "Nargacuga", "Tigrex"]
    
    def get_monster_info(self, monster_name):
        """Gather information about a specific monster."""
        self.log(f"\n--- Getting info for: {monster_name} ---")
        safe_name = quote(monster_name.replace(' ', '_'))
        search_url = f"{self.base_url}/wiki/{safe_name}"
        self.log(f"URL: {search_url}")
        
        try:
            response = requests.get(search_url, headers=self.headers)
            self.log(f"Response status code: {response.status_code}")
            
            # Save the HTML for debugging
            debug_filename = f"{monster_name.replace(' ', '_')}_page.html"
            debug_file_path = os.path.join(self.debug_file_path, debug_filename)
            with open(debug_file_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            self.log(f"Saved monster page HTML to {debug_file_path}")
            
            # Initialize monster data
            monster_data = __get_wikiinfo__(monster_name=monster_name)
            
            self.log(f"Extracted data for {monster_name}: {json.dumps(monster_data, indent=2)}")
            return monster_data
        
        except Exception as e:
            self.log(f"Error getting info for {monster_name}: {e}")
            return {
                "monster_name": monster_name,
                "url": f"https://monsterhunterwiki.org/wiki/{monster_name}",
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
    
    def download_monster_images(self, monster_name, limit = 200):
        """Download images for a specific monster."""
        self.log(f"\n--- Downloading images for: {monster_name} ---")
        
        # Create folder for monster images if it doesn't exist
        folder_name = monster_name.replace(" ", "_").replace("'", "").replace(":", "_")
        folder_path = os.path.join("monster_images", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        current_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

        if current_count < limit:
            self._download_from_google(monster_name, folder_path, limit - current_count)
        
        # Count the final number of images
        final_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        self.log(f"Downloaded a total of {final_count} images for {monster_name}")
    
    def _download_from_wiki(self, monster_name, folder_path, limit):
        """Download images from the Monster Hunter Wiki."""
        self.log(f"Downloading images from Monster Hunter Wiki for {monster_name}")
        search_url = f"{self.base_url}/wiki/{quote(monster_name.replace(' ', '_'))}"
        
        try:
            response = requests.get(search_url, headers=self.headers)
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find image elements on the wiki page
            img_tags = soup.find_all("img")
            img_urls = []
            
            for img in img_tags:
                src = img.get("src", "")
                if src and not src.endswith((".svg", ".png")):  # Skip icons and logos
                    # Convert relative URLs to absolute
                    if src.startswith("//"):
                        src = "https:" + src
                    elif not src.startswith("http"):
                        src = urljoin(self.base_url, src)
                    
                    img_urls.append(src)
            
            self.log(f"Found {len(img_urls)} potential image URLs on wiki")
            
            # Download images
            count = 0
            for i, img_url in enumerate(img_urls):
                if count >= limit:
                    break
                
                try:
                    img_response = requests.get(img_url, headers=self.headers)
                    
                    # Skip small images (likely icons or thumbnails)
                    content_length = int(img_response.headers.get('Content-Length', 0))
                    if content_length < 10000:  # Skip images smaller than 10KB
                        self.log(f"Skipping small image ({content_length} bytes): {img_url}")
                        continue
                    
                    img_path = os.path.join(folder_path, f"wiki_{monster_name.replace(' ', '_')}_{i+1}.jpg")
                    
                    with open(img_path, "wb") as f:
                        f.write(img_response.content)
                    
                    count += 1
                    if count % 5 == 0:
                        self.log(f"Downloaded {count} wiki images for {monster_name}")
                    
                    # Be nice to the server
                    time.sleep(0.2)
                    
                except Exception as e:
                    self.log(f"Error downloading wiki image {i+1} for {monster_name}: {e}")
            
            self.log(f"Downloaded {count} wiki images for {monster_name}")
            
        except Exception as e:
            self.log(f"Error accessing wiki for images of {monster_name}: {e}")
            
            
    def _download_from_google(self, monster_name, folder_path, limit):
        """Downloads images from Google Images for each monster name."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        print(f"Downloading images for: {monster_name}")
        monster_folder = folder_path
        os.makedirs(monster_folder, exist_ok=True)

        crawler = GoogleImageCrawler(storage={'root_dir': monster_folder})
        crawler.crawl(keyword=monster_name + ' Monster Hunter', max_num=10)
    
    def run_scraper(self, max_monsters=None):
        """Main function to run the scraper."""
        # Create the base directory for monster images
        os.makedirs("monster_images", exist_ok=True)
        
        # Load existing data if available
        if os.path.exists("./data/monsters_info.json"):
            try:
                with open("monsters_info.json", "r", encoding="utf-8") as f:
                    self.monsters_info = json.load(f)
                self.log(f"Loaded existing data for {len(self.monsters_info)} monsters")
            except Exception as e:
                self.log(f"Error loading existing data: {e}")
        
        # Get all monster names
        monster_names = self.get_all_monster_names()
        
        # Limit the number of monsters if specified
        if max_monsters:
            monster_names = monster_names[:max_monsters]
        
        __get_image__(monster_names)
        # Process each monster
        for monster_name in monster_names:
            # Skip if already processed
            if monster_name in self.monsters_info:
                self.log(f"Skipping {monster_name} - already processed")
                continue
            
            # Get monster info
            monster_info = self.get_monster_info(monster_name)
            
            # Add to the dictionary
            self.monsters_info[monster_name] = monster_info
            
            # Download images
            # self.download_monster_images(monster_name, limit=max_monsters)
            
            # Save the JSON after each monster (in case the script gets interrupted)
            with open("monsters_info.json", "w", encoding="utf-8") as f:
                json.dump(self.monsters_info, f, indent=4)
            
            # Be nice to the server
            time.sleep(1)
        
        self.log(f"Completed scraping {len(monster_names)} monsters.")
        self.log(f"Monster information saved to monsters_info.json")
        self.log(f"Monster images saved to monster_images/ directory")
        self.log_file.close()

def create_dummy_content():
    """Create dummy content for testing purposes."""
    print("Creating dummy content for testing...")
    
    # Create dummy monster info
    dummy_monsters = {
        "Rathalos": {
            "weakness": ["Dragon", "Thunder"],
            "strategy": "Target the wings to ground it. Avoid its poisonous talons and fire breath. Use flash bombs when it's airborne.",
            "recommended_gear": "Water or Thunder weapons work well. High Fire Resistance armor is recommended.",
            "locations": ["Ancient Forest", "Elder's Recess"]
        },
        "Zinogre": {
            "weakness": ["Ice", "Water"],
            "strategy": "Attack when it's charging to interrupt. Target the back and legs to topple it. Beware of its lightning attacks.",
            "recommended_gear": "Ice weapons are most effective. Thunder Resistance helps against its attacks.",
            "locations": ["Coral Highlands", "Guiding Lands"]
        },
        "Nargacuga": {
            "weakness": ["Thunder", "Fire"],
            "strategy": "Wait for openings after its attacks. Dodge sideways when it pounces. Break the tail to reduce its range.",
            "recommended_gear": "Thunder weapons and high mobility. Earplugs can help against its roars.",
            "locations": ["Ancient Forest", "Coral Highlands"]
        }
    }
    
    # Save dummy info to JSON
    with open("./data/monsters_info.json", "w", encoding="utf-8") as f:
        json.dump(dummy_monsters, f, indent=4)
    
    # Create dummy image folders
    for monster in dummy_monsters:
        folder_name = monster.replace(" ", "_")
        folder_path = os.path.join("monster_images", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Create a text file explaining the dummy content
        with open(os.path.join(folder_path, "README.txt"), "w") as f:
            f.write(f"This is a placeholder folder for {monster} images.\n")
            f.write("In a real run, this folder would contain images downloaded from various sources.\n")
            f.write("Due to web scraping limitations, actual image downloads may not work as expected.\n")
    
    print("Dummy content created. Check monsters_info.json and monster_images/ directory.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monster Hunter Wiki Scraper")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of monsters to scrape")
    parser.add_argument("--dummy", action="store_true", help="Create dummy content for testing")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.dummy:
        create_dummy_content()
    else:
        scraper = MonsterHunterScraper(debug=args.debug)
        scraper.run_scraper(max_monsters=args.max)