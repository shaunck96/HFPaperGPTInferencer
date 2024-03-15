import requests
import os

def download_pdf(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"PDF downloaded successfully to: {save_path}")
    else:
        print("Failed to download PDF.")

def main():
    url = "https://arxiv.org/pdf/2403.08551.pdf"
    save_folder = "./downloaded_pdfs"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    filename = os.path.join(save_folder, "2403.08551.pdf")
    download_pdf(url, filename)

if __name__ == "__main__":
    main()
