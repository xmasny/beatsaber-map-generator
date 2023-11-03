import csv
import json
import os
import traceback
from collections import Counter

import jsonschema
import librosa
import numpy as np
import requests
from jsonschema import validate


class DataGeneration:
	def __init__(self, file):
		self.terminal_file = file

		with open('saved_data/map_info.json', 'r', encoding='utf-8') as f:
			self.map_info = json.load(f)

		with open('saved_data/map_zips.csv', 'r', encoding='utf-8') as f:
			reader = csv.reader(f)
			self.map_csv = list(reader)

	def save_song_info(self):
		for song in self.map_csv:
			if song[0] in os.listdir('data'):
				song_info = [id for id in self.map_info if id['id'] == song[1]]

				if 'generated' not in os.listdir(f'data/{song[0]}'):
					os.mkdir(f'data/{song[0]}/generated')

				with open(f'data/{song[0]}/generated/SongInfo.dat', 'w') as f:
					json.dump(song_info[0], f)
					message = f"{song[0]} SongInfo.dat saved"
					print(message)
					self.terminal_file.write(f"{message}\n")
					self.terminal_file.flush()
			else:
				message = f"{song[0]} not found"
				print(message)
				self.terminal_file.write(f"{message}\n")
				self.terminal_file.flush()

	def get_song_versions(self):
		versions = []
		for song in self.map_csv:
			try:
				if song[0] in os.listdir('data'):
					directory_path = f"data/{song[0]}"
					info_dat_file = next((filename for filename in os.listdir(
						directory_path) if filename.lower() == 'info.dat'), None)
					print(song[0], "versions")
					self.terminal_file.write(f"{song[0]} versions\n")
					self.terminal_file.flush()

					if info_dat_file is not None:
						info_dat_file_path = os.path.join(
							directory_path, info_dat_file)
						difficulty_beatmaps_filenames = []
						try:
							with open(info_dat_file_path, 'r') as f:
								song_info = json.load(f)

							for difficultyBeatmapSet in song_info['_difficultyBeatmapSets']:
								for difficultyBeatmap in difficultyBeatmapSet['_difficultyBeatmaps']:
									difficulty_beatmaps_filenames.append(
										difficultyBeatmap['_beatmapFilename'])

							for difficultyFilename in difficulty_beatmaps_filenames:
								try:
									with open(f'{directory_path}/{difficultyFilename}', 'r') as f:
										difficulty_data = json.load(f)

										if 'version' in difficulty_data:
											versions.append(
												difficulty_data['version'])
										elif '_version' in difficulty_data:
											versions.append(
												difficulty_data['_version'])
										else:
											message = f"'version' or '_version' not found in {difficultyFilename}"
											print(song[0], message)
											self.terminal_file.write(
												f"{song[0]} {message}\n")
											self.terminal_file.flush()
								except OSError as e:
									message = f"Error opening difficulty file: {e}"
									print(message)
									self.terminal_file.write(f"{message}\n")
									self.terminal_file.flush()
									continue
								except json.JSONDecodeError as e:
									message = f"Error loading JSON data from difficulty file: {e}"
									print(message)
									self.terminal_file.write(f"{message}\n")
									self.terminal_file.flush()
									continue
						except OSError as e:
							message = f"Error opening 'Info.dat' file: {e}"
							print(message)
							self.terminal_file.write(f"{message}\n")
							self.terminal_file.flush()
							continue
						except json.JSONDecodeError as e:
							message = f"Error loading JSON data from 'Info.dat' file: {e}"
							print(message)
							self.terminal_file.write(f"{message}\n")
							self.terminal_file.flush()
							continue
					else:
						message = "'Info.dat' file not found in directory."
						print(song[0], message)
						print(os.listdir(directory_path))
						print('-------------------')
						self.terminal_file.write(f"{song[0]} {message}\n")
						self.terminal_file.write(
							f"{os.listdir(directory_path)}\n")
						self.terminal_file.write("-------------------\n")
						self.terminal_file.flush()
				else:
					message = "not found"
					print(song[0], message)
					self.terminal_file.write(f"{song[0]} {message}\n")
					self.terminal_file.flush()
			except Exception as e:
				message = f"Unexpected error occurred: {e}\n{traceback.format_exc()}"
				print(message)
				self.terminal_file.write(f"{message}\n")
				self.terminal_file.flush()
				continue

		message = f"Versions: {Counter(versions)}"
		print(message)
		self.terminal_file.write(f"{message}\n")
		self.terminal_file.flush()

	def get_song_info_versions(self):
		versions = []
		for song in self.map_csv:
			try:
				if song[0] in os.listdir('data'):
					directory_path = f"data/{song[0]}"
					info_dat_file = next((filename for filename in os.listdir(
						directory_path) if filename.lower() == 'info.dat'), None)
					if info_dat_file is not None:
						info_dat_file_path = os.path.join(
							directory_path, info_dat_file)
						try:
							with open(info_dat_file_path, 'r') as f:
								song_info = json.load(f)

							if 'version' in song_info:
								versions.append(
									song_info['version'])

								print(
									f"{song[0]} info version {song_info['version']}")
								self.terminal_file.write(
									f"{song[0]} info version {song_info['version']}\n")
								self.terminal_file.flush()
							elif '_version' in song_info:
								versions.append(
									song_info['_version'])
								print(
									f"{song[0]} info version {song_info['_version']}")
								self.terminal_file.write(
									f"{song[0]} info version {song_info['_version']}\n")
								self.terminal_file.flush()
							else:
								message = f"'version' or '_version' not found in {info_dat_file}"
								print(song[0], message)
								self.terminal_file.write(
									f"{song[0]} {message}\n")
								self.terminal_file.flush()
						except OSError as e:
							message = f"Error opening 'Info.dat' file: {e}"
							print(message)
							self.terminal_file.write(f"{message}\n")
							self.terminal_file.flush()
							continue
						except json.JSONDecodeError as e:
							message = f"Error loading JSON data from 'Info.dat' file: {e}"
							print(message)
							self.terminal_file.write(f"{message}\n")
							self.terminal_file.flush()
							continue
					else:
						message = "'Info.dat' file not found in directory."
						print(song[0], message)
						print(os.listdir(directory_path))
						print('-------------------')
						self.terminal_file.write(f"{song[0]} {message}\n")
						self.terminal_file.write(
							f"{os.listdir(directory_path)}\n")
						self.terminal_file.write("-------------------\n")
						self.terminal_file.flush()
				else:
					message = "not found"
					print(song[0], message)
					self.terminal_file.write(f"{song[0]} {message}\n")
					self.terminal_file.flush()
			except Exception as e:
				message = f"Unexpected error occurred: {e}\n{traceback.format_exc()}"
				print(message)
				self.terminal_file.write(f"{message}\n")
				self.terminal_file.flush()
				continue

		message = f"Versions: {Counter(versions)}"
		print(message)
		self.terminal_file.write(f"{message}\n")
		self.terminal_file.flush()

	def generate_v3_beatmap(self):
		difficulty_beatmap = {
			"version"                          : "3.2.0",
			"bpmEvents"                        : [],
			"rotationEvents"                   : [],
			"colorNotes"                       : [],
			"bombNotes"                        : [],
			"obstacles"                        : [],
			"sliders"                          : [],
			"basicBeatmapEvents"               : [],
			"colorBoostBeatmapEvents"          : [],
			"lightColorEventBoxGroups"         : [],
			"lightRotationEventBoxGroups"      : [],
			"lightTranslationEventBoxGroups"   : [],
			"useNormalEventsAsCompatibleEvents": True
		}

		incorrect_versions = []
		for song in self.map_csv:
			try:
				if song[0] in os.listdir('data'):
					directory_path = f"data/{song[0]}/"

					song_beatmaps = [filename for filename in os.listdir(
						directory_path) if filename.endswith('.dat') and not filename.lower().endswith('info.dat')]

					for difficulty in song_beatmaps:
						with open(os.path.join(directory_path, difficulty), 'r') as f:
							difficulty_data = json.load(f)

							if '_version' in difficulty_data and difficulty_data['_version'][0] == '2':
								message = f"{directory_path}/{difficulty} version {difficulty_data['_version']}"
								print(message)
								transfer_to_v3(
									difficulty_beatmap.copy(), difficulty_data, directory_path, difficulty)
								self.terminal_file.write(f"{message}\n")
								self.terminal_file.flush()
							elif 'version' in difficulty_data and difficulty_data['version'][0] == '3':
								message = f"{directory_path}/{difficulty} version {difficulty_data['version']}"
								print(message)
								self.terminal_file.write(f"{message}\n")
								self.terminal_file.flush()
							else:
								message = f"{directory_path}/{difficulty} incorrect version"
								print(message)
								incorrect_versions.append(
									f"{directory_path}/{difficulty}")
								self.terminal_file.write(f"{message}\n")
								self.terminal_file.flush()
								with open(os.path.join('incorrect_versions.json'), 'w') as file:
									file.write(json.dumps(incorrect_versions))
			except Exception as e:
				message = f"Unexpected error occurred: {e}\n{traceback.format_exc()}"
				print(message)
				self.terminal_file.write(f"{message}\n")
				self.terminal_file.flush()


def mel_gen_and_save(self):
	for song in self.map_csv:
		try:
			if song[0] in os.listdir('data'):
				directory_path = f"data/{song[0]}/"

				song_file = [filename for filename in os.listdir(
					directory_path) if filename.endswith('.egg')][0]
				audio_data, sample_rate = librosa.load(
					directory_path + song_file)

				# Compute the Mel spectrogram
				mel_spectrogram = librosa.feature.melspectrogram(
					y=audio_data, sr=sample_rate)
				np.save(f'{directory_path}generated/song_mel.npy',
				        mel_spectrogram)

				message = f'Mel spectrogram were generated for song file in {song[0]}'

				print(message)
				self.terminal_file.write(f"{message}\n")
				self.terminal_file.flush()

		except Exception as e:
			message = f"Unexpected error occurred: {e}\n{traceback.format_exc()}"
			print(message)
			self.terminal_file.write(f"{message}\n")
			self.terminal_file.flush()

	def save_tensors(self):
		for song in self.map_csv:
			try:
				if song[0] in os.listdir('data'):
					directory_path = f"data/{song[0]}/"
					song_beatmaps = [filename for filename in os.listdir(directory_path) if
					                 filename.endswith('.dat') and filename.lower() not in ['info.dat', 'songinfo.dat']]

					for difficulty in song_beatmaps:
						difficulty_data = None
						with open(os.path.join(directory_path, difficulty), 'r') as f:
							difficulty_data = json.load(f)

						if 'version' in difficulty_data and difficulty_data['version'][0] == '3':
							generate_tensor(difficulty_data)

						else:
							with open(os.path.join(directory_path, 'generated', difficulty), 'r') as f:
								difficulty_data = json.load(f)

							generate_tensor(difficulty_data)

			except Exception as e:
				message = f"Unexpected error occurred: {e}\n{traceback.format_exc()}"
				print(message)
				self.terminal_file.write(f"{message}\n")
				self.terminal_file.flush()


def generate_tensor():
	return NotImplemented


def transfer_to_v3(difficulty_beatmap, difficulty_data, directory_path, difficulty):
	print('generating v3 beatmap for', difficulty, 'in', directory_path)
	for note in difficulty_data['_notes']:
		if note['_type'] == 0 or note['_type'] == 1:
			difficulty_beatmap['colorNotes'].append({
				'b': note['_time'],
				'x': note['_lineIndex'],
				'y': note['_lineLayer'],
				'c': note['_type'],
				'd': note['_cutDirection'],
				'a': 0
			})

		elif note['_type'] == 3:
			difficulty_beatmap['bombNotes'].append({
				'b': note['_time'],
				'x': note['_lineIndex'],
				'y': note['_lineLayer'],
			})

	for obstacle in difficulty_data['_obstacles']:
		difficulty_beatmap['obstacles'].append({
			'b': obstacle['_time'],
			'd': obstacle['_duration'],
			'x': obstacle['_lineIndex'],
			'w': obstacle['_width'],
			'y': 0 if obstacle['_type'] == 0 else 2,
			'h': 5 if obstacle['_type'] == 0 else 3
		})
	os.makedirs(f'{directory_path}/generated/maps_v2_to_v3', exist_ok=True)
	with open(f'{directory_path}/generated/maps_v2_to_v3/{difficulty}', 'w') as f:
		# URL of the JSON schema
		schema_url = 'https://raw.githubusercontent.com/xmasny/beatmap-schemas/master/schemas/difficulty-v3.schema.json'

		# Fetch the schema from the URL
		response = requests.get(schema_url)

		# Check if the request was successful
		if response.status_code == 200:
			schema = response.json()
		else:
			print(f"Failed to fetch the schema from URL: {schema_url}")
		try:
			# Validate the data against the schema
			validate(instance=difficulty_beatmap, schema=schema)
			print("JSON data is valid.")
			json.dump(difficulty_beatmap, f)
		except jsonschema.exceptions.ValidationError as e:
			print("JSON data is invalid.")
			print(e)
