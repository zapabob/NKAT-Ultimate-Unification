import time; from tqdm import tqdm; print("tqdm test"); [time.sleep(0.1) for i in tqdm(range(10), desc="Test")]
