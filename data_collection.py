import pandas as pd
import os
import requests
import zipfile
import io

def download_bbc_dataset():
    """
    Downloads the BBC News dataset from the public repository
    Contains over 2000 articles across multiple categories
    """
    print("Downloading BBC News dataset...")
    
    # BBC dataset URL from public repository
    url = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # extract zip file
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall("bbc_data")
        
        print("Dataset downloaded successfully")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print("Will use alternative dataset source...")
        return False

def load_bbc_data():
    """
    Loads BBC data from extracted folders
    Focuses on sport and politics categories
    """
    articles = []
    labels = []
    
    categories = {
        'sport': 'Sport',
        'politics': 'Politics'
    }
    
    for folder, label in categories.items():
        folder_path = f"bbc_data/bbc/{folder}"
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(folder_path, filename), 'r', encoding='latin-1') as f:
                        text = f.read()
                        # get just the article body, skip the headline
                        text = ' '.join(text.split('\n')[1:])
                        if len(text) > 100:  # filter out very short texts
                            articles.append(text)
                            labels.append(label)
    
    df = pd.DataFrame({'text': articles, 'category': labels})
    return df


def save_dataset(df, filename='dataset.csv'):
    """Saves the collected dataset to CSV"""
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)
    df.to_csv(filepath, index=False)
    
    print(f"\nDataset saved to {filepath}")
    print(f"Total samples: {len(df)}")
    print(f"Sport samples: {len(df[df['category'] == 'Sport'])}")
    print(f"Politics samples: {len(df[df['category'] == 'Politics'])}")
    
    return filepath

def main():
    """Main data collection workflow"""
    print("=" * 60)
    print("Collecting Sport vs Politics Dataset")
    print("=" * 60)
    
    # try downloading BBC dataset first
    if download_bbc_dataset():
        df = load_bbc_data()
        if len(df) > 0:
            print(f"Loaded {len(df)} articles from BBC dataset")
        else:
            print("Unable to load BBC Dataset, exiting...")
            return
    else:
        print("Unable to load BBC Dataset, exiting...")
        return
    
    # save the dataset
    filepath = save_dataset(df)
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    df = main()
