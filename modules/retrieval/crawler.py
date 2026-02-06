"""
Wikimedia Commons Crawler V2
Module crawl ảnh từ Wikimedia Commons với retry logic, exponential backoff
Dựa trên best practices để xử lý rate limiting
"""

import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm


class WikimediaCommonsCrawler:
    """
    Crawler để lấy ảnh từ Wikimedia Commons
    Xử lý rate limiting với exponential backoff và retry mechanism
    """
    
    def __init__(self, base_dir: str = "data/knowledge_base"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_url = "https://commons.wikimedia.org/w/api.php"
        
        # Session với headers đầy đủ để giả lập browser
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://commons.wikimedia.org/",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "image",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "cross-site"
        })
    
    def get_images_from_category(self, category: str, limit: int = 50) -> List[Dict]:
        """
        Lấy danh sách file từ category
        
        Args:
            category: Tên category (ví dụ: "Angkor Wat", "Eiffel Tower")
            limit: Số file tối đa
        """
        images = []
        cmcontinue = None
        
        while len(images) < limit:
            params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmtype": "file",
                "cmlimit": min(50, limit - len(images)),
                "origin": "*"
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            
            r = self.session.get(self.api_url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            images.extend(data["query"]["categorymembers"])
            
            if "continue" not in data:
                break
            
            cmcontinue = data["continue"]["cmcontinue"]
            time.sleep(0.5)
        
        return images[:limit]
    
    def get_image_metadata(self, titles: List[str], batch_size: int = 20) -> Dict:
        """
        Lấy metadata và URL của ảnh (batch processing)
        
        Args:
            titles: List file titles
            batch_size: Số file mỗi batch
        """
        all_pages = {}
        
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i + batch_size]
            
            params = {
                "action": "query",
                "format": "json",
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|mime|size",
                "iiurlwidth": 1024,  # Thumbnail 1024px (nhanh hơn full resolution)
                "origin": "*"
            }
            
            r = self.session.get(self.api_url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            if "query" in data:
                all_pages.update(data["query"]["pages"])
            
            time.sleep(0.5)  # Rate limit
        
        return all_pages
    
    def download_image(
        self,
        url: str,
        save_path: Path,
        retry: int = 5,
        base_delay: float = 2.0
    ) -> bool:
        """
        Download một ảnh với retry và exponential backoff
        
        Args:
            url: URL ảnh
            save_path: Đường dẫn lưu
            retry: Số lần retry tối đa
            base_delay: Delay cơ bản (sẽ tăng exponential khi retry)
        """
        for attempt in range(retry):
            try:
                # Delay trước mỗi request (trừ lần đầu)
                if attempt > 0:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                
                r = self.session.get(url, stream=True, timeout=30)
                r.raise_for_status()
                
                # Lưu file
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                return True
            
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                
                if status_code == 429:  # Too Many Requests
                    wait_time = base_delay * (2 ** (attempt + 2))
                    print(f"429 Rate Limited! Waiting {wait_time:.1f}s...")
                    if attempt < retry - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Failed after {retry} attempts")
                        return False
                
                elif status_code == 403:
                    print(f"403 Forbidden (attempt {attempt + 1}/{retry})")
                    if attempt < retry - 1:
                        time.sleep(base_delay * 3)
                        continue
                    else:
                        return False
                else:
                    raise
            
            except Exception as e:
                if attempt == retry - 1:
                    print(f"Failed: {e}")
                    return False
                time.sleep(base_delay)
        
        return False
    
    def download_landmark_images(
        self,
        category: str,
        folder_name: str,
        max_images: int = 50,
        delay_between_images: float = 2.5
    ) -> List[Dict]:
        """
        Download tất cả ảnh của một landmark
        
        Args:
            category: Tên category trên Wikimedia
            folder_name: Tên thư mục lưu
            max_images: Số ảnh tối đa
            delay_between_images: Delay giữa các download (giây)
        """
        print(f"\n{'='*60}")
        print(f"Landmark: {category}")
        print(f"{'='*60}")
        
        # Tạo thư mục
        landmark_dir = self.base_dir / "images" / folder_name
        landmark_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Lấy danh sách file
        print("Fetching file list...")
        files = self.get_images_from_category(category, limit=max_images)
        print(f"Found {len(files)} files")
        
        if not files:
            print("No files found")
            return []
        
        # 2. Lấy metadata
        print("Fetching metadata...")
        titles = [f["title"] for f in files]
        meta_pages = self.get_image_metadata(titles)
        
        # 3. Download ảnh
        print(f"Downloading images (delay: {delay_between_images}s)...")
        records = []
        success_count = 0
        
        VALID_MIME = {"image/jpeg", "image/png"}
        
        for idx, page in enumerate(tqdm(meta_pages.values(), desc="Progress")):
            try:
                info = page.get("imageinfo", [{}])[0]
                
                # Kiểm tra MIME type
                if info.get("mime") not in VALID_MIME:
                    continue
                
                # Ưu tiên thumburl (1024px), fallback về url gốc
                img_url = info.get("thumburl") or info.get("url")
                if not img_url:
                    continue
                
                # Tạo tên file
                img_name = f"img_{idx:04d}.jpg"
                img_path = landmark_dir / img_name
                
                # Download với retry
                success = self.download_image(img_url, img_path, retry=5, base_delay=2.0)
                
                if success:
                    success_count += 1
                    
                    # Tạo record
                    meta = info.get("extmetadata", {})
                    record = {
                        "image": str(img_path),
                        "landmark": category,
                        "url": img_url,
                        "description": meta.get("ImageDescription", {}).get("value", ""),
                        "license": meta.get("LicenseShortName", {}).get("value", ""),
                        "author": meta.get("Artist", {}).get("value", "")
                    }
                    records.append(record)
                
                # QUAN TRỌNG: Delay để tránh 429
                time.sleep(delay_between_images)
            
            except Exception as e:
                print(f"Skip image {idx}: {repr(e)}")
                continue
        
        print(f"\nDownloaded {success_count}/{len(meta_pages)} images")
        print(f"Saved to: {landmark_dir}")
        
        # Lưu metadata
        metadata_file = landmark_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        return records


# =============================================================================
# Danh sách landmarks nổi tiếng
# =============================================================================

FAMOUS_LANDMARKS = {
    # --- Asia ---
    "Angkor_Wat": "Angkor Wat",
    "Taj_Mahal": "Taj Mahal",
    "Great_Wall_of_China": "Great Wall of China",
    "Forbidden_City": "Forbidden City",
    "Mount_Fuji": "Mount Fuji",
    "Petronas_Towers": "Petronas Towers",
    
    # --- Europe ---
    "Eiffel_Tower": "Eiffel Tower",
    "Colosseum": "Colosseum",
    "Big_Ben": "Big Ben",
    "Sagrada_Familia": "Sagrada Familia",
    "Notre_Dame_de_Paris": "Notre-Dame de Paris",
    "Neuschwanstein_Castle": "Neuschwanstein Castle",
    
    # --- Americas ---
    "Statue_of_Liberty": "Statue of Liberty",
    "Golden_Gate_Bridge": "Golden Gate Bridge",
    "Machu_Picchu": "Machu Picchu",
    "Christ_the_Redeemer": "Christ the Redeemer",
    
    # --- Middle East & Africa ---
    "Pyramids_of_Giza": "Pyramids of Giza",
    "Petra": "Petra",
    "Burj_Khalifa": "Burj Khalifa",
    
    # --- Oceania ---
    "Sydney_Opera_House": "Sydney Opera House",
}


if __name__ == "__main__":
    # Crawl landmarks nổi tiếng cho dự án DAM-RAG
    landmarks = {
        "Taj_Mahal": "Taj Mahal",
        "Great_Wall_of_China": "Great Wall of China",
        "Eiffel_Tower": "Eiffel Tower",
        "Colosseum": "Colosseum",
        "Statue_of_Liberty": "Statue of Liberty",
        "Machu_Picchu": "Machu Picchu",
        "Angkor_Wat": "Angkor Wat",
        "Burj_Khalifa": "Burj Khaliba",
        "Sydney_Opera_House": "Sydney Opera House",
        "Ha_Long_Bay": "Ha Long Bay",
    }
    
    print(f"Crawling {len(landmarks)} landmarks...")
    print(f"Estimated time: ~{len(landmarks) * 30 * 3 / 60:.0f} minutes\n")
    
    crawler = WikimediaCommonsCrawler()
    all_records = []
    
    for i, (folder, category) in enumerate(landmarks.items(), 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(landmarks)}] {category}")
        print(f"{'='*70}")
        
        try:
            records = crawler.download_landmark_images(
                category=category,
                folder_name=folder,
                max_images=30,
                delay_between_images=3.0
            )
            all_records.extend(records)
            print(f"Downloaded {len(records)} images for {category}")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"CRAWLING COMPLETED")
    print(f"Total images: {len(all_records)}")
    print(f"{'='*70}")
