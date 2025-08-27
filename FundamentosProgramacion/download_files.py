import requests
import multiprocessing
import time

csv_urls = [
    "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/iris.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/tips.csv",
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
]

def download_site(url):
    session = requests.Session()  
    with session.get(url) as response:
        name = multiprocessing.current_process().name
        print(f"{name}: Read {len(response.content)} bytes from {url}")

def download_all_sites(sites):
    with multiprocessing.Pool() as pool:
        pool.map(download_site, sites)

def main():
    start_time = time.time()
    download_all_sites(csv_urls)
    duration = time.time() - start_time
    print(f"Downloaded {len(csv_urls)} files in {duration} seconds")

if __name__ == "__main__":
    main()
