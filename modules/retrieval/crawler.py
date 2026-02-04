"""
Wikimedia Commons Crawler
Crawl ảnh và metadata về địa danh, di tích, kiến trúc từ Wikimedia Commons
"""

import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
from io import BytesIO
import urllib.parse


class WikimediaCommonsCrawler:
    """
    Crawler để lấy ảnh và metadata từ Wikimedia Commons
    
    Wikimedia Commons chứa hàng triệu ảnh về:
    - Địa danh nổi tiếng (landmarks)
    - Di tích lịch sử (monuments)
    - Kiến trúc (architecture)
    - Cảnh quan (landscapes)
    
    Mỗi ảnh có metadata phong phú: tên, mô tả, địa điểm, năm xây dựng, etc.
    """
    
    def __init__(
        self,
        output_dir: str = "data/knowledge_base",
        max_images_per_category: int = 100,
        sleep_time: float = 0.5
    ):
        """
        Args:
            output_dir: Thư mục lưu ảnh và metadata
            max_images_per_category: Số ảnh tối đa mỗi category
            sleep_time: Thời gian chờ giữa các requests (để tránh rate limit)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.output_dir / "metadata.jsonl"
        
        self.max_images_per_category = max_images_per_category
        self.sleep_time = sleep_time
        
        # Wikimedia Commons API endpoint
        self.api_url = "https://commons.wikimedia.org/w/api.php"
        
        # User agent để tránh bị block
        self.headers = {
            'User-Agent': 'DAM-RAG/1.0 (Educational Research Project)'
        }
        
    def crawl_category(
        self,
        category: str,
        max_images: Optional[int] = None
    ) -> List[Dict]:
        """
        Crawl ảnh từ một category trên Wikimedia Commons
        
        Args:
            category: Tên category (ví dụ: "Angkor_Wat", "Eiffel_Tower")
            max_images: Số ảnh tối đa (None = dùng max_images_per_category)
            
        Returns:
            List các metadata dict
        """
        if max_images is None:
            max_images = self.max_images_per_category
        
        print(f"\nĐang crawl category: {category}")
        print(f"Số ảnh tối đa: {max_images}")
        
        results = []
        continue_param = None
        
        while len(results) < max_images:
            # Query API để lấy list ảnh trong category
            params = {
                'action': 'query',
                'format': 'json',
                'generator': 'categorymembers',
                'gcmtitle': f'Category:{category}',
                'gcmtype': 'file',
                'gcmlimit': min(50, max_images - len(results)),
                'prop': 'imageinfo',
                'iiprop': 'url|extmetadata|size',
                'iiurlwidth': 1024,
            }
            
            if continue_param:
                params['gcmcontinue'] = continue_param
            
            try:
                response = requests.get(
                    self.api_url,
                    params=params,
                    headers=self.headers,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                # Parse results
                if 'query' in data and 'pages' in data['query']:
                    for page_id, page in data['query']['pages'].items():
                        if 'imageinfo' in page:
                            info = page['imageinfo'][0]
                            
                            # Extract metadata
                            metadata = {
                                'page_id': page_id,
                                'title': page.get('title', ''),
                                'url': info.get('url', ''),
                                'thumb_url': info.get('thumburl', ''),
                                'width': info.get('width', 0),
                                'height': info.get('height', 0),
                                'category': category,
                            }
                            
                            # Extract extended metadata
                            extmeta = info.get('extmetadata', {})
                            metadata['description'] = extmeta.get('ImageDescription', {}).get('value', '')
                            metadata['artist'] = extmeta.get('Artist', {}).get('value', '')
                            metadata['credit'] = extmeta.get('Credit', {}).get('value', '')
                            metadata['license'] = extmeta.get('LicenseShortName', {}).get('value', '')
                            metadata['date'] = extmeta.get('DateTimeOriginal', {}).get('value', '')
                            
                            results.append(metadata)
                            
                            if len(results) >= max_images:
                                break
                
                # Check if there are more results
                if 'continue' in data and len(results) < max_images:
                    continue_param = data['continue'].get('gcmcontinue')
                    time.sleep(self.sleep_time)
                else:
                    break
                    
            except Exception as e:
                print(f"Lỗi khi crawl: {e}")
                break
        
        print(f"✓ Đã crawl {len(results)} ảnh từ category {category}")
        return results
    
    def download_image(
        self,
        url: str,
        save_path: Path
    ) -> bool:
        """
        Download ảnh từ URL
        
        Returns:
            True nếu thành công, False nếu lỗi
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Mở và lưu ảnh
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            img.save(save_path, 'JPEG', quality=90)
            
            return True
            
        except Exception as e:
            print(f"Lỗi download {url}: {e}")
            return False
    
    def crawl_and_save(
        self,
        categories: List[str],
        download_images: bool = True
    ) -> int:
        """
        Crawl nhiều categories và lưu metadata + ảnh
        
        Args:
            categories: List các category names
            download_images: True để download ảnh, False để chỉ lưu metadata
            
        Returns:
            Tổng số ảnh đã crawl
        """
        total_count = 0
        
        # Mở file metadata để append
        with open(self.metadata_file, 'a', encoding='utf-8') as f:
            for category in categories:
                results = self.crawl_category(category)
                
                for idx, metadata in enumerate(results):
                    # Download ảnh nếu cần
                    if download_images and metadata.get('thumb_url'):
                        # Tạo filename từ page_id
                        filename = f"{category}_{metadata['page_id']}.jpg"
                        save_path = self.images_dir / filename
                        
                        if not save_path.exists():
                            print(f"Downloading {idx+1}/{len(results)}: {filename}...", end='\r')
                            if self.download_image(metadata['thumb_url'], save_path):
                                metadata['local_path'] = str(save_path)
                            else:
                                continue  # Skip nếu download lỗi
                        else:
                            metadata['local_path'] = str(save_path)
                    
                    # Lưu metadata
                    f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
                    total_count += 1
                
                print(f"\n✓ Hoàn thành category {category}")
                time.sleep(self.sleep_time)
        
        print(f"\n✓ Tổng cộng đã crawl {total_count} ảnh")
        return total_count
    
    @staticmethod
    def get_popular_landmark_categories() -> List[str]:
        """
        Trả về list các category phổ biến về địa danh nổi tiếng
        """
        return [
            "Angkor_Wat",
            "Eiffel_Tower", 
            "Taj_Mahal",
            "Great_Wall_of_China",
            "Colosseum",
            "Machu_Picchu",
            "Petra",
            "Christ_the_Redeemer",
            "Stonehenge",
            "Notre-Dame_de_Paris",
            "Sagrada_Família",
            "Big_Ben",
            "Statue_of_Liberty",
            "Sydney_Opera_House",
            "Burj_Khalifa",
            "Saint_Basil's_Cathedral",
            "Tower_of_Pisa",
            "Alhambra",
            "Hagia_Sophia",
            "Forbidden_City",
        ]


def main():
    """
    Script chính để test crawler
    """
    print("="*80)
    print("WIKIMEDIA COMMONS CRAWLER")
    print("="*80)
    
    crawler = WikimediaCommonsCrawler(
        output_dir="data/knowledge_base",
        max_images_per_category=20,  # Test với số nhỏ trước
        sleep_time=0.5
    )
    
    # Test với một vài category
    test_categories = [
        "Angkor_Wat",
        "Eiffel_Tower",
        "Taj_Mahal",
    ]
    
    total = crawler.crawl_and_save(
        categories=test_categories,
        download_images=True
    )
    
    print(f"\n✓ Crawl hoàn tất: {total} ảnh")
    print(f"✓ Metadata: {crawler.metadata_file}")
    print(f"✓ Images: {crawler.images_dir}")


if __name__ == "__main__":
    main()
