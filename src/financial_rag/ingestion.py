from sec_edgar_downloader import Downloader

# Download the latest 10-K report for each ticker and store into data/raw
def download_fillings(companies: list[str], output_dir: str = 'data/raw'):
    download = Downloader('Mike', 'qu_minh_ng_280@gmail.com', output_dir)

    for ticker in companies:
        print(f"Downloading 10-K report for {ticker}...")
        download.get('10-K', ticker, limit = 1)

    print('Done.')


# Get the data from big 4 companies:
# AAPL: Apple
# MSFT: Microsoft
# AMZN: Amazon
# GOOGL: Google
if __name__ == "__main__":
    companies = ['AAPL', 'MSFT', 'AMZN', 'GOOGL']
    download_fillings(companies)