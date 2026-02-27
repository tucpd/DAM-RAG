"""
Wikimedia Commons Crawler V2
Module crawl ảnh từ Wikimedia Commons với retry logic, exponential backoff
Dựa trên best practices để xử lý rate limiting
"""

import requests
import json
import time
import re
import html
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


class WikimediaCommonsCrawler:
    """
    Crawler để lấy ảnh từ Wikimedia Commons
    Xử lý rate limiting với exponential backoff và retry mechanism
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.images_base = self.base_dir / "images"
        self.metadata_base = self.base_dir / "metadata"
        self.images_base.mkdir(parents=True, exist_ok=True)
        self.metadata_base.mkdir(parents=True, exist_ok=True)
        
        self.commons_api = "https://commons.wikimedia.org/w/api.php"
        self.wiki_api = "https://en.wikipedia.org/w/api.php"
        
        # Session với headers đầy đủ để
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })
    
    # =========================================================================
    # Wikipedia API - Thu thap thong tin chi tiet
    # =========================================================================
    
    def get_wikipedia_info(self, title: str) -> Dict:
        """
        Lay thong tin chi tiet tu Wikipedia cho mot landmark
        Bao gom: summary, coordinates, infobox data, full extract
        
        Args:
            title: Ten bai viet tren Wikipedia (vi du: "Eiffel Tower")
            
        Returns:
            Dict chua thong tin chi tiet
        """
        info = {
            "name": title,
            "summary": "",
            "description": "",
            "coordinates": None,
            "location": "",
            "country": "",
            "year_built": "",
            "architect": "",
            "style": "",
            "height": "",
            "area": "",
            "visitors_per_year": "",
            "unesco_status": "",
            "significance": "",
            "wikipedia_url": "",
            "categories": [],
        }
        
        try:
            # --- Buoc 1: Lay summary + extract ---
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts|coordinates|pageimages|categories|info",
                "exintro": True,
                "explaintext": True,
                "exsectionformat": "plain",
                "colimit": 1,
                "piprop": "original",
                "cllimit": 20,
                "inprop": "url",
                "redirects": 1,
                "origin": "*",
            }
            
            r = self.session.get(self.wiki_api, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            pages = data.get("query", {}).get("pages", {})
            page = list(pages.values())[0] if pages else {}
            
            if "missing" in page:
                print(f"  Wikipedia page not found: {title}")
                return info
            
            # Summary (intro extract)
            extract = page.get("extract", "")
            info["summary"] = extract[:2000] if extract else ""
            info["description"] = extract[:500] if extract else ""
            
            # Coordinates
            coords = page.get("coordinates", [])
            if coords:
                info["coordinates"] = {
                    "lat": coords[0].get("lat"),
                    "lon": coords[0].get("lon")
                }
            
            # Wikipedia URL
            info["wikipedia_url"] = page.get("fullurl", "")
            
            # Categories
            cats = page.get("categories", [])
            info["categories"] = [c.get("title", "").replace("Category:", "") 
                                  for c in cats]
            
            time.sleep(0.3)
            
            # --- Buoc 2: Lay full text de extract them thong tin ---
            params_full = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
                "exsectionformat": "plain",
                "redirects": 1,
                "origin": "*",
            }
            
            r2 = self.session.get(self.wiki_api, params=params_full, timeout=30)
            r2.raise_for_status()
            data2 = r2.json()
            
            pages2 = data2.get("query", {}).get("pages", {})
            page2 = list(pages2.values())[0] if pages2 else {}
            full_text = page2.get("extract", "")
            
            # Extract thong tin tu full text
            extracted = self._extract_info_from_text(full_text, title)
            info.update({k: v for k, v in extracted.items() if v})
            
            time.sleep(0.3)
            
            # --- Buoc 3: Lay Wikidata cho thong tin co cau truc ---
            wikidata_info = self._get_wikidata_info(title)
            # Wikidata co do chinh xac cao hon regex extraction,
            # uu tien Wikidata cho cac truong co cau truc
            wikidata_priority_fields = ["location", "country", "architect", 
                                         "style", "year_built", "height"]
            for key, value in wikidata_info.items():
                if value and (not info.get(key) or key in wikidata_priority_fields):
                    info[key] = value
            
        except Exception as e:
            print(f"  Error fetching Wikipedia info for {title}: {e}")
        
        return info
    
    def _extract_info_from_text(self, text: str, title: str) -> Dict:
        """
        Extract thong tin co cau truc tu full text Wikipedia
        Su dung regex patterns de tim cac thong tin quan trong
        """
        result = {}
        text_lower = text.lower()
        
        # --- Location / Country ---
        location_patterns = [
            r"(?:located|situated|found|stands|lies)\s+(?:in|at|on|near)\s+([A-Z][^,.]{5,60})",
            r"(?:is a|is an|is the)\s+[^.]*?(?:in|at)\s+([A-Z][^,.]{5,60})",
        ]
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                result["location"] = match.group(1).strip()[:100]
                break
        
        # --- Year built ---
        year_patterns = [
            r"(?:built|constructed|erected|completed|established|founded|opened)\s+(?:in|between|from|circa|around|c\.?)\s*(\d{3,4}(?:\s*[-\u2013]\s*\d{3,4})?(?:\s*(?:BC|AD|CE|BCE))?)",
            r"(?:built|constructed|erected|completed)\s+(?:in the\s+)?(\d{1,2}(?:st|nd|rd|th)\s+century(?:\s*(?:BC|AD|CE|BCE))?)",
            r"(\d{3,4}(?:\s*[-\u2013]\s*\d{3,4})?)\s*(?:AD|CE|BCE|BC)?\s*(?:by|under|during)",
        ]
        for pattern in year_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["year_built"] = match.group(1).strip()
                break
        
        # --- Architect / Designer ---
        architect_patterns = [
            r"(?:designed|architect(?:ed)?|planned|created)\s+by\s+([A-Z][^,.]{3,60})",
            r"architect\s+([A-Z][^,.]{3,60})",
        ]
        for pattern in architect_patterns:
            match = re.search(pattern, text)
            if match:
                result["architect"] = match.group(1).strip()[:80]
                break
        
        # --- Architectural style ---
        style_patterns = [
            r"(?:style|architecture|architectural style)[:\s]+([A-Z][^,.]{3,80})",
            r"(?:in the|in a|example of)\s+([A-Z][a-z]+(?:\s+[A-Za-z]+){0,3})\s+(?:style|architecture|tradition)",
            r"([A-Z][a-z]+(?:\s+[A-Za-z]+){0,2})\s+(?:architectural|architecture)\s+(?:style|tradition)",
        ]
        for pattern in style_patterns:
            match = re.search(pattern, text)
            if match:
                result["style"] = match.group(1).strip()[:80]
                break
        
        # --- Height ---
        height_patterns = [
            r"(?:height|tall|rises|stands)\s+(?:of\s+)?(?:approximately\s+)?(\d+(?:\.\d+)?\s*(?:m(?:et(?:er|re)s?)?|ft|feet)(?:\s*\(\s*\d+(?:\.\d+)?\s*(?:m|ft|feet)\s*\))?)",
            r"(\d+(?:\.\d+)?\s*(?:m(?:et(?:er|re)s?)?|ft|feet))\s+(?:tall|high|in height)",
        ]
        for pattern in height_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["height"] = match.group(1).strip()
                break
        
        # --- UNESCO ---
        if "unesco" in text_lower or "world heritage" in text_lower:
            unesco_match = re.search(
                r"(?:UNESCO\s+)?[Ww]orld\s+[Hh]eritage\s+[Ss]ite(?:\s+(?:in|since)\s+(\d{4}))?",
                text
            )
            if unesco_match:
                year = unesco_match.group(1) or ""
                result["unesco_status"] = f"World Heritage Site{' (since ' + year + ')' if year else ''}"
        
        # --- Visitors ---
        visitor_patterns = [
            r"(\d+(?:\.\d+)?\s*(?:million|billion))\s+(?:visitors|tourists|people)\s+(?:per|a|each|every)\s+year",
            r"(?:attracts?|receives?|welcomes?)\s+(?:approximately\s+)?(\d+(?:\.\d+)?\s*(?:million|billion))\s+(?:visitors|tourists)",
            r"(?:annual|yearly)\s+(?:visitors?|attendance|tourism)\s+(?:of\s+)?(?:approximately\s+)?(\d+(?:\.\d+)?\s*(?:million|billion))",
        ]
        for pattern in visitor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["visitors_per_year"] = match.group(1).strip()
                break
        
        # --- Significance (lay 1-2 cau dac biet) ---
        significance_patterns = [
            r"((?:is|was|remains)\s+(?:one of the|the|a|an)\s+(?:most|largest|tallest|oldest|greatest|finest|best|world's)[^.]{10,200}\.)",
            r"((?:considered|regarded|recognized|known|famous|renowned)\s+(?:as|for|to be)[^.]{10,200}\.)",
        ]
        significance_parts = []
        for pattern in significance_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches[:2]:
                significance_parts.append(m.strip())
        if significance_parts:
            result["significance"] = " ".join(significance_parts)[:400]
        
        return result
    
    def _get_wikidata_info(self, title: str) -> Dict:
        """
        Lay thong tin co cau truc tu Wikidata thong qua Wikipedia API
        """
        result = {}
        
        try:
            # Tim Wikidata entity ID
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "pageprops",
                "ppprop": "wikibase_item",
                "redirects": 1,
                "origin": "*",
            }
            
            r = self.session.get(self.wiki_api, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            
            pages = data.get("query", {}).get("pages", {})
            page = list(pages.values())[0] if pages else {}
            wikidata_id = page.get("pageprops", {}).get("wikibase_item")
            
            if not wikidata_id:
                return result
            
            # Lay data tu Wikidata
            wd_url = "https://www.wikidata.org/w/api.php"
            params_wd = {
                "action": "wbgetentities",
                "format": "json",
                "ids": wikidata_id,
                "props": "claims|descriptions",
                "languages": "en",
                "origin": "*",
            }
            
            r2 = self.session.get(wd_url, params=params_wd, timeout=20)
            r2.raise_for_status()
            wd_data = r2.json()
            
            entity = wd_data.get("entities", {}).get(wikidata_id, {})
            claims = entity.get("claims", {})
            
            # P17 = country
            country_claim = claims.get("P17", [])
            if country_claim:
                country_id = (country_claim[0].get("mainsnak", {})
                              .get("datavalue", {}).get("value", {}).get("id"))
                if country_id:
                    result["country"] = self._get_wikidata_label(country_id)
            
            # P131 = located in administrative territory
            if not result.get("location"):
                loc_claim = claims.get("P131", [])
                if loc_claim:
                    loc_id = (loc_claim[0].get("mainsnak", {})
                              .get("datavalue", {}).get("value", {}).get("id"))
                    if loc_id:
                        loc_name = self._get_wikidata_label(loc_id)
                        country = result.get("country", "")
                        if loc_name and country:
                            result["location"] = f"{loc_name}, {country}"
                        elif loc_name:
                            result["location"] = loc_name
            
            # P84 = architect
            arch_claim = claims.get("P84", [])
            if arch_claim:
                arch_id = (arch_claim[0].get("mainsnak", {})
                           .get("datavalue", {}).get("value", {}).get("id"))
                if arch_id:
                    result["architect"] = self._get_wikidata_label(arch_id)
            
            # P149 = architectural style
            style_claim = claims.get("P149", [])
            if style_claim:
                style_id = (style_claim[0].get("mainsnak", {})
                            .get("datavalue", {}).get("value", {}).get("id"))
                if style_id:
                    result["style"] = self._get_wikidata_label(style_id)
            
            # P571 = inception (year built)
            inception_claim = claims.get("P571", [])
            if inception_claim:
                time_val = (inception_claim[0].get("mainsnak", {})
                            .get("datavalue", {}).get("value", {}).get("time", ""))
                if time_val:
                    year_match = re.search(r"[+-]?(\d{4})", time_val)
                    if year_match:
                        result["year_built"] = year_match.group(1)
            
            # P2048 = height
            height_claim = claims.get("P2048", [])
            if height_claim:
                height_val = (height_claim[0].get("mainsnak", {})
                              .get("datavalue", {}).get("value", {}).get("amount", ""))
                unit = (height_claim[0].get("mainsnak", {})
                        .get("datavalue", {}).get("value", {}).get("unit", ""))
                if height_val:
                    h = height_val.lstrip("+")
                    if "metre" in unit or "meter" in unit:
                        result["height"] = f"{h} m"
                    elif "foot" in unit or "feet" in unit:
                        result["height"] = f"{h} ft"
                    else:
                        result["height"] = f"{h} m"
        
            # P1174 = visitors per year
            visitors_claim = claims.get("P1174", [])
            if visitors_claim:
                v_val = (visitors_claim[0].get("mainsnak", {})
                         .get("datavalue", {}).get("value", {}).get("amount", ""))
                if v_val:
                    v_num = float(v_val.lstrip("+"))
                    if v_num >= 1_000_000:
                        result["visitors_per_year"] = f"{v_num/1_000_000:.1f} million"
                    else:
                        result["visitors_per_year"] = f"{v_num:,.0f}"
            
            time.sleep(0.3)
            
        except Exception:
            # Wikidata la optional, khong can fail
            pass
        
        return result
    
    def _get_wikidata_label(self, entity_id: str) -> str:
        """Lay label (ten) cua mot Wikidata entity"""
        try:
            wd_url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbgetentities",
                "format": "json",
                "ids": entity_id,
                "props": "labels",
                "languages": "en",
                "origin": "*",
            }
            r = self.session.get(wd_url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            return (data.get("entities", {}).get(entity_id, {})
                    .get("labels", {}).get("en", {}).get("value", ""))
        except Exception:
            return ""
    
    # =========================================================================
    # Wikimedia Commons API - Thu thap anh
    # =========================================================================
    
    def get_images_from_category(self, category: str, limit: int = 100) -> List[Dict]:
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
            
            r = self.session.get(self.commons_api, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            members = data.get("query", {}).get("categorymembers", [])
            images.extend(members)
            
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
                "iiurlwidth": 1280,
                "origin": "*"
            }
            
            r = self.session.get(self.commons_api, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            if "query" in data:
                all_pages.update(data["query"]["pages"])
            
            time.sleep(0.5)
        
        return all_pages
    
    def _clean_html(self, text: str) -> str:
        """Xoa HTML tags va decode entities"""
        if not text:
            return ""
        clean = re.sub(r'<[^>]+>', '', text)
        clean = html.unescape(clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean
    
    def _is_valid_image(self, info: Dict, min_width: int = 400, min_height: int = 300) -> bool:
        """
        Kiem tra anh co hop le khong
        Loc bo anh qua nho, audio files, SVG, etc.
        """
        mime = info.get("mime", "")
        if mime not in {"image/jpeg", "image/png"}:
            return False
        
        width = info.get("width", 0)
        height = info.get("height", 0)
        if width < min_width or height < min_height:
            return False
        
        return True
    
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
                status_code = e.response.status_code if e.response else 0
                
                if status_code == 429:  # Too Many Requests
                    wait_time = base_delay * (2 ** (attempt + 2))
                    print(f"429 Rate Limited! Waiting {wait_time:.1f}s...")
                    if attempt < retry - 1:
                        time.sleep(wait_time)
                        continue
                    return False
                
                elif status_code == 403:
                    print(f"403 Forbidden (attempt {attempt + 1}/{retry})")
                    if attempt < retry - 1:
                        time.sleep(base_delay * 3)
                        continue
                    return False
                else:
                    if attempt == retry - 1:
                        return False
            
            except Exception:
                if attempt == retry - 1:
                    return False
                time.sleep(base_delay)
        
        return False
    
    def crawl_landmark(
        self,
        wiki_title: str,
        commons_category: str,
        folder_name: str,
        max_images: int = 100,
        delay_between_images: float = 2.0
    ) -> Dict:
        """
        Crawl toàn bộ dữ liệu cho một landmark:
        1. Thông tin chi tiết từ Wikipedia + Wikidata
        2. Ảnh từ Wikimedia Commons
        
        Args:
            category: Tên category trên Wikimedia
            folder_name: Tên thư mục lưu
            max_images: Số ảnh tối đa
            delay_between_images: Delay giữa các download (giây)
        """
        print(f"\n{'='*60}")
        print(f"  {wiki_title}")
        print(f"{'='*60}")
        
        # Tao thu muc
        metadata_dir = self.metadata_base / folder_name
        images_dir = self.images_base / folder_name
        metadata_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # ------------------------------------------------------------------
        # Buoc 1: Lay thong tin tu Wikipedia + Wikidata
        # ------------------------------------------------------------------
        print("  [1/3] Fetching Wikipedia + Wikidata info...")
        landmark_info = self.get_wikipedia_info(wiki_title)
        
        # In ra thong tin da thu thap
        non_empty = {k: v for k, v in landmark_info.items() 
                     if v and k not in ["summary", "categories"]}
        for key, val in non_empty.items():
            val_str = str(val)[:80]
            print(f"    {key}: {val_str}")
        
        # ------------------------------------------------------------------
        # Buoc 2: Lay danh sach anh tu Wikimedia Commons
        # ------------------------------------------------------------------
        print(f"  [2/3] Fetching image list from Commons (category: {commons_category})...")
        files = self.get_images_from_category(commons_category, limit=max_images * 2)
        print(f"    Found {len(files)} files in category")
        
        if not files:
            print("    No files found, trying alternative categories...")
            alt_categories = [
                f"Images of {commons_category}",
                f"Photographs of {commons_category}",
                f"Views of {commons_category}",
            ]
            for alt_cat in alt_categories:
                files = self.get_images_from_category(alt_cat, limit=max_images * 2)
                if files:
                    print(f"    Found {len(files)} files in '{alt_cat}'")
                    break
        
        if not files:
            print("    No images found!")
            self._save_landmark_info(metadata_dir, landmark_info, [])
            return {"info": landmark_info, "images": [], "image_count": 0}
        
        # Lay metadata cho cac files
        print("    Fetching metadata...")
        titles = [f["title"] for f in files]
        meta_pages = self.get_image_metadata(titles)
        
        # ------------------------------------------------------------------
        # Buoc 3: Download anh
        # ------------------------------------------------------------------
        print(f"  [3/3] Downloading images (max {max_images}, delay: {delay_between_images}s)...")
        records = []
        success_count = 0
        
        for idx, page in enumerate(tqdm(meta_pages.values(), desc=f"    {folder_name}")):
            if success_count >= max_images:
                break
            
            try:
                info = page.get("imageinfo", [{}])[0]
                
                if not self._is_valid_image(info):
                    continue
                
                img_url = info.get("thumburl") or info.get("url")
                if not img_url:
                    continue
                
                page_id = str(page.get("pageid", idx))
                img_name = f"{folder_name}_{page_id}.jpg"
                img_path = images_dir / img_name
                
                # Neu da download roi thi bo qua
                if img_path.exists():
                    success_count += 1
                    records.append(self._build_record(
                        img_path, folder_name, wiki_title, 
                        img_url, info, landmark_info, page_id
                    ))
                    continue
                
                success = self.download_image(img_url, img_path, retry=5, base_delay=2.0)
                
                if success:
                    success_count += 1
                    records.append(self._build_record(
                        img_path, folder_name, wiki_title,
                        img_url, info, landmark_info, page_id
                    ))
                
                time.sleep(delay_between_images)
            
            except Exception:
                continue
        
        print(f"    Downloaded {success_count} images")
        
        # Luu tat ca du lieu
        self._save_landmark_info(metadata_dir, landmark_info, records)
        
        return {
            "info": landmark_info,
            "images": records,
            "image_count": success_count
        }
    
    def _build_record(
        self, img_path: Path, folder_name: str, wiki_title: str,
        img_url: str, commons_info: Dict, landmark_info: Dict, page_id: str
    ) -> Dict:
        """Tao record metadata cho mot anh"""
        meta = commons_info.get("extmetadata", {})
        
        desc_raw = meta.get("ImageDescription", {}).get("value", "")
        description = self._clean_html(desc_raw)
        
        record = {
            "page_id": page_id,
            "image_path": str(img_path),
            "landmark": wiki_title,
            "name": wiki_title,
            "folder": folder_name,
            "url": img_url,
            
            # Thong tin tu Commons
            "image_description": description,
            "license": meta.get("LicenseShortName", {}).get("value", ""),
            "author": self._clean_html(meta.get("Artist", {}).get("value", "")),
            "date": meta.get("DateTimeOriginal", {}).get("value", 
                     meta.get("DateTime", {}).get("value", "")),
            
            # Thong tin tu Wikipedia/Wikidata (gop vao moi record de retrieval dung duoc)
            "location": landmark_info.get("location", ""),
            "country": landmark_info.get("country", ""),
            "year_built": landmark_info.get("year_built", ""),
            "architect": landmark_info.get("architect", ""),
            "style": landmark_info.get("style", ""),
            "height": landmark_info.get("height", ""),
            "unesco_status": landmark_info.get("unesco_status", ""),
            "visitors_per_year": landmark_info.get("visitors_per_year", ""),
            "significance": landmark_info.get("significance", ""),
            "description": landmark_info.get("description", ""),
            "coordinates": landmark_info.get("coordinates"),
        }
        
        return record
    
    def _save_landmark_info(self, landmark_dir: Path, info: Dict, records: List[Dict]):
        """Luu thong tin landmark va metadata anh"""
        # Luu landmark_info.json (thong tin chi tiet tu Wikipedia)
        info_file = landmark_dir / "landmark_info.json"
        info_save = info.copy()
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info_save, f, ensure_ascii=False, indent=2)
        
        # Luu metadata.jsonl (thong tin moi anh, format JSONL)
        metadata_file = landmark_dir / "metadata.jsonl"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"    Saved: {info_file}")
        print(f"    Saved: {metadata_file} ({len(records)} records)")
    
    def crawl_all_landmarks(
        self,
        landmarks: Dict[str, Dict],
        max_images_per_landmark: int = 100,
        delay_between_images: float = 2.0,
        delay_between_landmarks: float = 5.0,
        skip_existing: bool = True
    ) -> Dict[str, Dict]:
        """
        Crawl toan bo danh sach landmarks
        
        Args:
            landmarks: Dict {folder_name: {"wiki": ..., "commons": ...}}
            max_images_per_landmark: So anh toi da moi landmark
            delay_between_images: Delay giua cac downloads
            delay_between_landmarks: Delay giua cac landmarks
            skip_existing: Bo qua landmark da crawl roi (co landmark_info.json)
            
        Returns:
            Dict ket qua crawl
        """
        total = len(landmarks)
        print(f"\n{'='*70}")
        print(f"CRAWLING {total} LANDMARKS")
        print(f"Max images per landmark: {max_images_per_landmark}")
        est_minutes = total * max_images_per_landmark * delay_between_images / 60
        print(f"Estimated time: ~{est_minutes:.0f} minutes")
        print(f"{'='*70}")
        
        all_results = {}
        
        for idx, (folder_name, info) in enumerate(landmarks.items(), 1):
            wiki_title = info["wiki"]
            commons_category = info["commons"]
            
            print(f"\n[{idx}/{total}] {wiki_title}")
            
            # Kiem tra da crawl chua
            if skip_existing:
                metadata_dir = self.metadata_base / folder_name
                info_file = metadata_dir / "landmark_info.json"
                metadata_file = metadata_dir / "metadata.jsonl"
                
                if info_file.exists() and metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        existing_count = sum(1 for _ in f)
                    
                    if existing_count >= max_images_per_landmark * 0.5:
                        print(f"  SKIPPED (already has {existing_count} images)")
                        continue
            
            try:
                result = self.crawl_landmark(
                    wiki_title=wiki_title,
                    commons_category=commons_category,
                    folder_name=folder_name,
                    max_images=max_images_per_landmark,
                    delay_between_images=delay_between_images
                )
                all_results[folder_name] = result
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            
            if idx < total:
                print(f"\n  Waiting {delay_between_landmarks}s before next landmark...")
                time.sleep(delay_between_landmarks)
        
        # Tong ket
        total_images = sum(r.get("image_count", 0) for r in all_results.values())
        print(f"\n{'='*70}")
        print(f"CRAWLING COMPLETED")
        print(f"Landmarks crawled: {len(all_results)}")
        print(f"Total new images: {total_images}")
        print(f"{'='*70}")
        
        return all_results


# =============================================================================
# Danh sach landmarks mo rong (50+ dia diem)
# Moi entry co: wiki (ten Wikipedia), commons (category Wikimedia Commons)
# =============================================================================

LANDMARKS = {
    # =====================================================================
    # Asia
    # =====================================================================
    "Angkor_Wat": {
        "wiki": "Angkor Wat",
        "commons": "Angkor Wat",
    },
    "Taj_Mahal": {
        "wiki": "Taj Mahal",
        "commons": "Taj Mahal",
    },
    "Great_Wall_of_China": {
        "wiki": "Great Wall of China",
        "commons": "Great Wall of China",
    },
    "Forbidden_City": {
        "wiki": "Forbidden City",
        "commons": "Forbidden City",
    },
    "Mount_Fuji": {
        "wiki": "Mount Fuji",
        "commons": "Mount Fuji",
    },
    "Petronas_Towers": {
        "wiki": "Petronas Towers",
        "commons": "Petronas Towers",
    },
    "Ha_Long_Bay": {
        "wiki": "H\u1ea1 Long Bay",
        "commons": "Ha Long Bay",
    },
    "Hoi_An": {
        "wiki": "Hoi An",
        "commons": "Hoi An",
    },
    "Fushimi_Inari_Taisha": {
        "wiki": "Fushimi Inari-taisha",
        "commons": "Fushimi Inari-taisha",
    },
    "Borobudur": {
        "wiki": "Borobudur",
        "commons": "Borobudur",
    },
    "Sigiriya": {
        "wiki": "Sigiriya",
        "commons": "Sigiriya",
    },
    "Bagan": {
        "wiki": "Bagan",
        "commons": "Bagan",
    },
    "Terracotta_Army": {
        "wiki": "Terracotta Army",
        "commons": "Terracotta Army",
    },
    "Gardens_by_the_Bay": {
        "wiki": "Gardens by the Bay",
        "commons": "Gardens by the Bay",
    },
    
    # =====================================================================
    # Europe
    # =====================================================================
    "Eiffel_Tower": {
        "wiki": "Eiffel Tower",
        "commons": "Eiffel Tower",
    },
    "Colosseum": {
        "wiki": "Colosseum",
        "commons": "Colosseum",
    },
    "Big_Ben": {
        "wiki": "Big Ben",
        "commons": "Big Ben",
    },
    "Sagrada_Familia": {
        "wiki": "Sagrada Fam\u00edlia",
        "commons": "Sagrada Familia",
    },
    "Notre_Dame_de_Paris": {
        "wiki": "Notre-Dame de Paris",
        "commons": "Notre-Dame de Paris",
    },
    "Neuschwanstein_Castle": {
        "wiki": "Neuschwanstein Castle",
        "commons": "Neuschwanstein Castle",
    },
    "Acropolis_of_Athens": {
        "wiki": "Acropolis of Athens",
        "commons": "Acropolis of Athens",
    },
    "Stonehenge": {
        "wiki": "Stonehenge",
        "commons": "Stonehenge",
    },
    "Leaning_Tower_of_Pisa": {
        "wiki": "Leaning Tower of Pisa",
        "commons": "Leaning Tower of Pisa",
    },
    "Santorini": {
        "wiki": "Santorini",
        "commons": "Santorini",
    },
    "Prague_Castle": {
        "wiki": "Prague Castle",
        "commons": "Prague Castle",
    },
    "Tower_of_London": {
        "wiki": "Tower of London",
        "commons": "Tower of London",
    },
    "Alhambra": {
        "wiki": "Alhambra",
        "commons": "Alhambra",
    },
    "Mont_Saint_Michel": {
        "wiki": "Mont-Saint-Michel",
        "commons": "Mont Saint-Michel",
    },
    "Hallstatt": {
        "wiki": "Hallstatt",
        "commons": "Hallstatt",
    },
    "Blue_Mosque": {
        "wiki": "Sultan Ahmed Mosque",
        "commons": "Sultan Ahmed Mosque",
    },
    "Hagia_Sophia": {
        "wiki": "Hagia Sophia",
        "commons": "Hagia Sophia",
    },
    
    # =====================================================================
    # Americas
    # =====================================================================
    "Statue_of_Liberty": {
        "wiki": "Statue of Liberty",
        "commons": "Statue of Liberty",
    },
    "Golden_Gate_Bridge": {
        "wiki": "Golden Gate Bridge",
        "commons": "Golden Gate Bridge",
    },
    "Machu_Picchu": {
        "wiki": "Machu Picchu",
        "commons": "Machu Picchu",
    },
    "Christ_the_Redeemer": {
        "wiki": "Christ the Redeemer (statue)",
        "commons": "Christ the Redeemer (statue)",
    },
    "Chichen_Itza": {
        "wiki": "Chichen Itza",
        "commons": "Chichen Itza",
    },
    "Grand_Canyon": {
        "wiki": "Grand Canyon",
        "commons": "Grand Canyon",
    },
    "Niagara_Falls": {
        "wiki": "Niagara Falls",
        "commons": "Niagara Falls",
    },
    "Central_Park": {
        "wiki": "Central Park",
        "commons": "Central Park",
    },
    "Times_Square": {
        "wiki": "Times Square",
        "commons": "Times Square",
    },
    
    # =====================================================================
    # Middle East & Africa
    # =====================================================================
    "Pyramids_of_Giza": {
        "wiki": "Great Pyramid of Giza",
        "commons": "Great Pyramid of Giza",
    },
    "Petra": {
        "wiki": "Petra",
        "commons": "Petra",
    },
    "Burj_Khalifa": {
        "wiki": "Burj Khalifa",
        "commons": "Burj Khalifa",
    },
    "Abu_Simbel": {
        "wiki": "Abu Simbel temples",
        "commons": "Abu Simbel",
    },
    "Luxor_Temple": {
        "wiki": "Luxor Temple",
        "commons": "Luxor Temple",
    },
    "Sheikh_Zayed_Mosque": {
        "wiki": "Sheikh Zayed Grand Mosque",
        "commons": "Sheikh Zayed Mosque (Abu Dhabi)",
    },
    
    # =====================================================================
    # Oceania
    # =====================================================================
    "Sydney_Opera_House": {
        "wiki": "Sydney Opera House",
        "commons": "Sydney Opera House",
    },
    "Uluru": {
        "wiki": "Uluru",
        "commons": "Uluru",
    },
    "Great_Barrier_Reef": {
        "wiki": "Great Barrier Reef",
        "commons": "Great Barrier Reef",
    },
    
    # =====================================================================
    # Natural Wonders
    # =====================================================================
    "Victoria_Falls": {
        "wiki": "Victoria Falls",
        "commons": "Victoria Falls",
    },
    "Iguazu_Falls": {
        "wiki": "Iguazu Falls",
        "commons": "Iguazu Falls",
    },
    "Aurora_Borealis": {
        "wiki": "Aurora (astronomy)",
        "commons": "Aurora borealis",
    },
    "Matterhorn": {
        "wiki": "Matterhorn",
        "commons": "Matterhorn",
    },
    "Cappadocia": {
        "wiki": "Cappadocia",
        "commons": "Cappadocia",
    },
}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Crawl landmark data from Wikipedia + Wikimedia Commons"
    )
    parser.add_argument("--max-images", type=int, default=30,
                        help="Max images per landmark (default: 100)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay between image downloads in seconds (default: 2.0)")
    parser.add_argument("--landmark-delay", type=float, default=5.0,
                        help="Delay between landmarks in seconds (default: 5.0)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-crawl even if data already exists")
    parser.add_argument("--landmarks", nargs="*", default=None,
                        help="Specific landmarks to crawl (folder names). If none, crawl all.")
    args = parser.parse_args()
    
    # Chon landmarks de crawl
    if args.landmarks:
        selected = {k: v for k, v in LANDMARKS.items() if k in args.landmarks}
        if not selected:
            print(f"No matching landmarks found. Available: {list(LANDMARKS.keys())}")
            exit(1)
    else:
        selected = LANDMARKS
    
    print(f"Selected {len(selected)} landmarks")
    print(f"Max images per landmark: {args.max_images}")
    print(f"Delay between images: {args.delay}s")
    print(f"Skip existing: {not args.no_skip}")
    
    crawler = WikimediaCommonsCrawler()
    
    results = crawler.crawl_all_landmarks(
        landmarks=selected,
        max_images_per_landmark=args.max_images,
        delay_between_images=args.delay,
        delay_between_landmarks=args.landmark_delay,
        skip_existing=not args.no_skip
    )
    
    # In ket qua
    print("\nResults per landmark:")
    for folder, result in results.items():
        info = result.get("info", {})
        count = result.get("image_count", 0)
        loc = info.get("location", "N/A")
        year = info.get("year_built", "N/A")
        print(f"  {folder}: {count} images | Location: {loc} | Year: {year}")
