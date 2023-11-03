import csv
import json
import math
import os
import time
import zipfile

import requests

time.time()


class DownloadScript:
	def __init__(self, file):
		self.terminal_file = file
		self.songFolder = "data"
		self.zipFiles = []
		self.map_info = []
		self.maps_ids = []

	def get_all_users(self, start=0, end=20000):
		# get all users in range of pages
		all_users_json = []
		for page in range(start, end):
			while True:
				try:
					response = requests.get(
						f"https://api.beatsaver.com/users/list/{page}")
					break
				except Exception as e:
					print(e)
					self.terminal_file.write(f"{e}\n")
					time.sleep(10)
			json_users = response.json()
			for user in json_users:
				all_users_json.append([
					user["id"],
					user['stats']["totalMaps"],
				])
			print(f"Users page {page} done")
			self.terminal_file.write(f"Users page {page} done\n")
			self.terminal_file.flush()

		if os.path.exists('saved_data/usersNew.csv'):
			if os.path.exists('saved_data/usersOld.csv'):
				os.remove('saved_data/usersOld.csv')
			os.rename('saved_data/usersNew.csv', 'saved_data/usersOld.csv')
		with open(f'saved_data/usersNew.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerows(all_users_json)

	def get_all_maps(self):
		# get all zip maps from users
		song_index = 1
		all_maps = []

		with open('saved_data/usersNew.csv', 'r') as users_file:
			users = csv.reader(users_file)
			for (user) in users:
				user_maps = 0
				for page in range(0, math.ceil(int(int(user[1]) / 20) + 1)):
					while True:
						try:
							response = requests.get(
								f"https://api.beatsaver.com/maps/uploader/{user[0]}/{page}")
							break
						except Exception as e:
							print(e)
							self.terminal_file.write(f"{e}\n")
							time.sleep(10)
					json_user_maps = response.json()
					user_maps += len(json_user_maps['docs'])
					print(
						f"User {user[0]} maps page {page} done")
					self.terminal_file.write(
						f"User {user[0]} maps page {page} done\n")
					self.terminal_file.flush()

					all_maps.extend(json_user_maps['docs'])

				user_maps_info = [user[0], user[1], user_maps]
				with open('saved_data/user_maps.csv', 'a', newline='') as file:
					writer = csv.writer(file)
					writer.writerow(user_maps_info)

		print("Number of maps: ", len(all_maps))
		self.terminal_file.write(f"Number of maps: {len(all_maps)}\n")

		with open('saved_data/map_info.json', 'w') as file:
			json.dump(all_maps, file)

		with open('saved_data/map_info.json', 'r') as file:
			all_maps = json.load(file)
			print("Number of maps from json: ", len(all_maps))

	def get_all_maps_urls(self):
		# get all zip maps urls
		all_maps_urls = []
		with open('saved_data/map_info.json', 'r') as file:
			all_maps = json.load(file)
			for index, map in enumerate(all_maps):
				all_maps_urls.append([
					f'song{index + 1}',
					map["id"],
					map['versions'][0]["downloadURL"]
				])
				print(f"Map {index + 1}/{len(all_maps)} added")
				self.terminal_file.write(f"Map {index} added\n")
				self.terminal_file.flush()
		with open('saved_data/map_zips.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerows(all_maps_urls)

	def get_all_maps_from_user(self, start_id=None, end_id=None):
		"""
		Download all maps from user
		start_id: start index of all_maps_urls
		end_id: end index of all_maps_urls

		default: download all maps
		"""
		with open('saved_data/map_zips.csv', 'r', newline='') as file:
			reader = csv.reader(file)
			all_maps_urls = list(reader)
			response = None

		for map in all_maps_urls[start_id:end_id]:
			while True:
				try:
					response = requests.get(map[2])
					break
				except Exception as e:
					print(e)
					time.sleep(10)
			with open(f"data/{map[0]}.zip", 'wb') as f:
				f.write(response.content)
			print(f"{map[0]}/{len(all_maps_urls)} downloaded")
			self.terminal_file.write(
				f"{map[0]}/{len(all_maps_urls)} downloaded\n")
			self.terminal_file.flush()

	def get_all_zips(self):
		# get all zip files
		for file_name in os.listdir(self.songFolder):
			if file_name.endswith(".zip"):
				self.zipFiles.append(file_name)
				print(f"{file_name} added")
				self.terminal_file.write(f"{file_name} added\n")

	def unzip_all_zips(self):
		# unzip file
		for index, zip_file in enumerate(self.zipFiles):
			print(f"Unzipping {zip_file} {index + 1}/{len(self.zipFiles)}")
			self.terminal_file.write(
				f"Unzipping {zip_file} {index + 1}/{len(self.zipFiles)}\n")
			try:
				with zipfile.ZipFile(f"data/{zip_file}", "r") as zip_ref:
					zip_ref.extractall(f"data/{zip_file[:-4]}")
					print(f"{zip_file} unzipped")
					self.terminal_file.write(f"{zip_file} unzipped\n")
				# remove song zip file
				os.remove(f"data/{zip_file}")
				print(f"{zip_file} removed")
				self.terminal_file.write(f"{zip_file} removed\n")
			except Exception as e:

				# Printing the file name
				print(f"Error occurred in file song{index + 1}")
				self.terminal_file.write(
					f"Error occurred in file {index}\n")
				continue
