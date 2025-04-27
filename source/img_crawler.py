from icrawler.builtin import BingImageCrawler
from PIL import Image
import time
import os
import re
import concurrent.futures
from threading import Event, Thread
from queue import Queue, Empty

def safe_folder_name(name):
    return re.sub(r'[^\w\-_. ]', '', name).replace(' ', '_')

def process_image(filepath, min_size=(200, 200), max_size=(1024, 1024), max_retries=3):
    """Process a single image with retry capability"""
    filename = os.path.basename(filepath)
    
    # Skip already processed files
    if filename.endswith('.jpg') and os.path.splitext(filename)[0].isdigit():
        return True
        
    retries = 0
    while retries < max_retries:
        try:
            with Image.open(filepath) as img:
                # Filter based on size
                if img.width < min_size[0] or img.height < min_size[1] or \
                   img.width > max_size[0] or img.height > max_size[1]:
                    os.remove(filepath)
                    return False

                # Convert to JPEG and overwrite
                rgb_img = img.convert('RGB')
                new_filename = os.path.splitext(filename)[0] + ".jpg"
                new_filepath = os.path.join(os.path.dirname(filepath), new_filename)
                rgb_img.save(new_filepath, "JPEG")

                # Delete the original file if it was not a jpg
                if filename != new_filename:
                    os.remove(filepath)
                return True
                
        except PermissionError as e:
            print(f"Access error on {filename}, retry {retries+1}/{max_retries}: {e}")
            retries += 1
            time.sleep(1)  # Wait before retrying
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            try:
                os.remove(filepath)
            except:
                pass
            return False
    
    return False

def file_monitor(folder_path, file_queue, stop_event, check_interval=0.5):
    """Monitor a folder for new files and add them to the processing queue"""
    processed_files = set()
    
    while not stop_event.is_set():
        try:
            for filename in os.listdir(folder_path):
                filepath = os.path.join(folder_path, filename)
                if filepath not in processed_files and os.path.isfile(filepath):
                    file_queue.put(filepath)
                    processed_files.add(filepath)
        except Exception as e:
            print(f"Error monitoring folder {folder_path}: {e}")
        
        time.sleep(check_interval)

def image_processor(file_queue, stop_event):
    """Process images from the queue until stopped"""
    while not stop_event.is_set() or not file_queue.empty():
        try:
            filepath = file_queue.get(timeout=1)
            process_image(filepath)
            file_queue.task_done()
        except Empty:
            continue
        except Exception as e:
            print(f"Error in image processor: {e}")

def download_and_process_monster(monster, base_dir, max_images=150):
    """Download and process images for a single monster"""
    print(f"Starting download for: {monster}")
    folder_name = safe_folder_name(monster)
    monster_folder = os.path.join(base_dir, folder_name)
    os.makedirs(monster_folder, exist_ok=True)
    
    # Check how many images are already in the folder
    existing_files = [f for f in os.listdir(monster_folder) if os.path.isfile(os.path.join(monster_folder, f)) 
                      and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    existing_count = len(existing_files)
    
    # If we already have max_images or more, skip downloading
    if existing_count >= max_images:
        print(f"Skipping {monster}: already has {existing_count} images (max: {max_images})")
        return monster
    
    # Calculate how many more images we need
    images_to_download = max_images - existing_count
    print(f"Found {existing_count} existing images for {monster}, downloading {images_to_download} more")
    
    # Setup the processing queue and workers
    file_queue = Queue()
    stop_event = Event()  # Using threading.Event instead of concurrent.futures.Event
    
    # Start the file monitor thread
    monitor_thread = Thread(target=file_monitor, args=(monster_folder, file_queue, stop_event))
    monitor_thread.start()
    
    # Start image processor threads
    processor_threads = []
    for _ in range(2):  # 2 processor threads
        processor = Thread(target=image_processor, args=(file_queue, stop_event))
        processor.start()
        processor_threads.append(processor)
    
    # Start the crawler - only download the additional images needed
    crawler = BingImageCrawler(storage={'root_dir': monster_folder})
    crawler.crawl(keyword=f'{monster} Monster Hunter', max_num=images_to_download)
    
    # Allow some time for final processing
    time.sleep(5)
    
    # Signal workers to stop
    stop_event.set()
    
    # Wait for all threads to complete
    monitor_thread.join()
    for thread in processor_threads:
        thread.join()
    
    # Process any remaining files that might have been missed
    remaining_files = [os.path.join(monster_folder, f) for f in os.listdir(monster_folder)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(process_image, remaining_files)
    
    print(f"Completed processing for: {monster}")
    return monster

def __get_image__(monster_names):
    base_dir = './data/monster_images'
    os.makedirs(base_dir, exist_ok=True)
    
    # Process multiple monsters in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks and get futures
        futures = [
            executor.submit(download_and_process_monster, monster, base_dir)
            for monster in monster_names
        ]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                monster = future.result()
                print(f"✓ {monster} processing completed successfully")
            except Exception as e:
                print(f"× A monster processing task failed: {e}")
    
    print("All monster images have been downloaded and processed.")
    return