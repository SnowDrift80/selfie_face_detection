from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import time
import process4 as p


class ImageHandler(FileSystemEventHandler):
    file_types = ['.jpg', '.png']
    
    def __init__(self, input_dir):
        self.input_dir = input_dir
        
    def on_created(self, event):
        if any(event.src_path.endswith(ft) for ft in self.file_types):
            p.ProcessPhoto.align_and_resize_face(event.src_path, 512)
    
def start_monitoring(input_dir):
    event_handler = ImageHandler(input_dir)
    observer = Observer()
    observer.schedule(event_handler, input_dir, recursive=True)
    observer.start()
    
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    
    
    
start_monitoring("./monitored")