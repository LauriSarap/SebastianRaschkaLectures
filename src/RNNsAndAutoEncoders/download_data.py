import urllib.request

url = "https://github.com/rasbt/python-machine-learning-book-3rd-edition/raw/master/ch08/movie_data.csv.gz"
file_path = "movie_data.csv.gz"

urllib.request.urlretrieve(url, file_path)
print(f"Downloaded {file_path}")
