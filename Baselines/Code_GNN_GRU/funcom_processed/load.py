import json

def load():
	"""
	Loads json data from files and returns a dictionary for each data type
	where the key is function_id and the value is the source code or comment
	"""
	src_path = 'functions.json'
	com_path = 'comments.json'

	with open(src_path, 'r') as fp:
		src = json.load(fp, parse_int=True)

	# newsrc = {}
	# for k, v in src.items():
	# 	newsrc[int(k)] = v

	# del src

	with open(com_path, 'r') as fp:
		com = json.load(fp)


	# newcom = {}
	# for k, v in com.items():
	# 	newcom[int(k)] = v

	# del com

	return (src, com)

def load_comment():
	"""
	Loads json data from files and returns a dictionary
	where the key is function_id and the value is the comment
	"""
	com_path = 'comments.json'

	with open(com_path, 'r') as fp:
		com = json.load(fp)

	return com

def load_function():
	"""
	Loads json data from files and returns a dictionary
	where the key is function_id and the value is the function
	"""
	src_path = 'functions.json'

	with open(src_path, 'r') as fp:
		src = json.load(fp)

	return src

if __name__ == "__main__":
	dats, come = load()

	for k,v in dats.items():
		print(k, v)