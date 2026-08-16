from typing import List, Dict, Set, Any
import re

class CrossModalQueryExpander:
    """
    SOTA Cross-Modal Query Expander.
    Expands natural language user queries into multi-modal semantic cues
    (visual object synonyms, acoustic cues, and bilingual translations).
    """

    def __init__(self):
        # Semantic mapping dictionaries for Thai and English domain concepts
        self.synonym_dict: Dict[str, Dict[str, List[str]]] = {
            "ดื่ม": {
                "visual": ["แก้วน้ำ", "ขวดน้ำ", "หลอดดูด", "drinking", "cup", "bottle", "sip", "water"],
                "audio": ["กลืน", "ดื่ม", "drink", "water"]
            },
            "กิน": {
                "visual": ["อาหาร", "จาน", "ช้อน", "ส้อม", "eating", "food", "plate", "spoon", "fork"],
                "audio": ["กิน", "อร่อย", "eat", "food", "delicious"]
            },
            "พูด": {
                "visual": ["ไมโครโฟน", "คนยืนพูด", "ผู้บรรยาย", "speaking", "microphone", "speaker", "presenter"],
                "audio": ["สวัสดี", "พูด", "กล่าว", "explain", "discuss", "presentation"]
            },
            "สไลด์": {
                "visual": ["กราฟ", "หน้าจอ", "ตัวหนังสือ", "ตาราง", "slide", "chart", "graph", "diagram", "table", "presentation"],
                "audio": ["สไลด์", "กราฟ", "ภาพนี้", "ตามตาราง", "slide", "chart", "figure"]
            },
            "กราฟ": {
                "visual": ["กราฟแท่ง", "กราฟเส้น", "แผนภูมิ", "chart", "bar chart", "line plot", "figure"],
                "audio": ["กราฟ", "ร้อยละ", "เปอร์เซ็นต์", "แกน", "axis", "percentage", "trend"]
            },
            "รถ": {
                "visual": ["รถยนต์", "ถนน", "ล้อรถ", "ไฟแดง", "car", "vehicle", "automobile", "road", "street", "traffic"],
                "audio": ["เสียงรถ", "เครื่องยนต์", "car", "engine", "drive"]
            },
            "เดิน": {
                "visual": ["ทางเดิน", "ก้าวขา", "รองเท้า", "walking", "walk", "feet", "pedestrian", "path"],
                "audio": ["เดิน", "ก้าว", "walk", "step"]
            },
            "เขียน": {
                "visual": ["ปากกา", "กระดาษ", "ไวท์บอร์ด", "คีย์บอร์ด", "writing", "pen", "paper", "whiteboard", "typing"],
                "audio": ["เขียน", "จด", "write", "note", "type"]
            }
        }

    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        Expands user query into visual and acoustic keywords and semantic query variations.
        """
        q_clean = query.strip()
        tokens = re.findall(r'\w+', q_clean.lower())
        
        visual_cues: Set[str] = set(tokens)
        audio_cues: Set[str] = set(tokens)
        
        for word in tokens:
            for key, mapped in self.synonym_dict.items():
                if key in word or word in key:
                    visual_cues.update(mapped.get("visual", []))
                    audio_cues.update(mapped.get("audio", []))

        return {
            "original_query": q_clean,
            "visual_keywords": list(visual_cues),
            "audio_keywords": list(audio_cues),
            "expanded_search_str": f"{q_clean} {' '.join(list(visual_cues)[:5])}"
        }

query_expander = CrossModalQueryExpander()
