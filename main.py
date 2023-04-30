import time
from download_script import DownloadScript

start_time = time.time()

download = DownloadScript()

download.get_all_users()
download.get_all_maps()
download.get_all_maps_from_user(end_id=3)
download.get_all_zips()
download.unzip_all_zips()

end_time = time.time()
runtime = end_time - start_time

print("Runtime:", runtime, "seconds")
