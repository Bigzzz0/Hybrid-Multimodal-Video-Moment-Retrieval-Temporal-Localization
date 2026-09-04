from typing import List, Dict, Set, Any, Tuple
import re

class VisualQueryDecomposer:
    """
    SOTA Visual Query Decomposer and Semantic Expander (Visual-Centric SOTA 2024-2026).
    Decomposes natural language user queries into fine-grained visual facets:
    - Primary Entities & Objects (สิ่งที่มองเห็นในฉาก)
    - Physical Movements & Action Verbs (กริยาและการเคลื่อนไหวทางกายภาพ)
    - Spatial Scene & Context Attributes (บริบทแวดล้อมและเสื้อผ้า)
    - Cross-lingual Thai-to-English visual alignments for SigLIP 2.
    """

    def __init__(self):
        # Purely Visual & Action concept dictionary
        self.concept_dict: Dict[str, Dict[str, List[str]]] = {
            # Colors & Visual Appearance
            "สีเขียว": {"visual": ["green", "emerald", "green clothing", "green shirt"]},
            "เขียว": {"visual": ["green", "emerald", "green clothing"]},
            "สีแดง": {"visual": ["red", "crimson", "red clothing", "red shirt"]},
            "แดง": {"visual": ["red", "crimson", "red clothing"]},
            "สีน้ำตาล": {"visual": ["brown", "tan", "khaki", "brown shirt"]},
            "น้ำตาล": {"visual": ["brown", "tan", "khaki"]},
            "สีดำ": {"visual": ["black", "dark", "black clothing", "black shirt"]},
            "ดำ": {"visual": ["black", "dark", "black clothing"]},
            "สีขาว": {"visual": ["white", "light", "white clothing", "white shirt"]},
            "ขาว": {"visual": ["white", "light", "white clothing"]},
            "สีน้ำเงิน": {"visual": ["blue", "navy", "dark blue", "blue shirt"]},
            "น้ำเงิน": {"visual": ["blue", "navy", "dark blue"]},
            "สีฟ้า": {"visual": ["blue", "cyan", "sky blue", "blue shirt"]},
            "ฟ้า": {"visual": ["blue", "cyan", "sky blue"]},
            "สีเหลือง": {"visual": ["yellow", "gold", "yellow shirt"]},
            "เหลือง": {"visual": ["yellow", "gold"]},
            "สีส้ม": {"visual": ["orange", "orange shirt"]},
            "ส้ม": {"visual": ["orange", "orange shirt"]},
            "สีชมพู": {"visual": ["pink", "rose", "pink shirt"]},
            "ชมพู": {"visual": ["pink", "rose"]},
            "สีม่วง": {"visual": ["purple", "violet", "purple shirt"]},
            "ม่วง": {"visual": ["purple", "violet"]},
            "สีเทา": {"visual": ["gray", "grey", "gray shirt"]},
            "เทา": {"visual": ["gray", "grey"]},

            # Clothing & Wearables
            "เสื้อ": {"visual": ["shirt", "t-shirt", "polo", "clothing", "apparel", "jacket", "top"]},
            "กางเกง": {"visual": ["pants", "trousers", "jeans", "shorts"]},
            "แว่น": {"visual": ["glasses", "eyeglasses", "spectacles", "eyewear"]},
            "แว่นตา": {"visual": ["glasses", "eyeglasses", "spectacles"]},
            "หมวก": {"visual": ["hat", "cap", "helmet", "headwear"]},
            "รองเท้า": {"visual": ["shoes", "sneakers", "boots", "footwear"]},
            "กระเป๋า": {"visual": ["bag", "backpack", "handbag", "suitcase"]},

            # People & Subjects
            "คน": {"visual": ["person", "people", "individual", "human", "someone"]},
            "ผู้ชาย": {"visual": ["man", "guy", "male", "gentleman"]},
            "ชาย": {"visual": ["man", "guy", "male"]},
            "หนุ่ม": {"visual": ["young man", "guy", "man"]},
            "ผู้หญิง": {"visual": ["woman", "female", "lady", "girl"]},
            "หญิง": {"visual": ["woman", "female", "lady"]},
            "สาว": {"visual": ["young woman", "girl", "woman"]},
            "เด็ก": {"visual": ["child", "kid", "toddler", "boy", "girl"]},
            "อาจารย์": {"visual": ["teacher", "professor", "lecturer", "presenter", "speaker"]},
            "ครู": {"visual": ["teacher", "instructor", "lecturer"]},
            "ผู้บรรยาย": {"visual": ["presenter", "speaker", "person standing in front"]},
            "นักเรียน": {"visual": ["student", "pupil", "uniform"]},
            "นักศึกษา": {"visual": ["student", "college student", "university student"]},

            # Physical Actions & Dynamic Body Movements
            "ใส่": {"visual": ["wearing", "dressed in", "putting on clothing"]},
            "สวม": {"visual": ["wearing", "dressed in", "putting on"]},
            "ถอด": {"visual": ["taking off", "removing clothing", "unbuttoning"]},
            "นั่ง": {"visual": ["sitting", "seated", "sits", "chair", "sofa", "desk"]},
            "ยืน": {"visual": ["standing", "stands upright", "standing up"]},
            "ก้ม": {"visual": ["bending down", "leaning forward", "stooping", "looking down"]},
            "เงย": {"visual": ["looking up", "raising head", "facing upward"]},
            "ยิ้ม": {"visual": ["smiling", "smiles", "happy facial expression", "grin"]},
            "หัวเราะ": {"visual": ["laughing", "laughs", "chuckle", "joyful"]},
            "ยกมือ": {"visual": ["raising hand", "hand raised up", "gesturing with arm", "reaching up"]},
            "โบกมือ": {"visual": ["waving hand", "waving goodbye", "greeting gesture"]},
            "ชี้": {"visual": ["pointing with finger", "indicating", "hand pointing"]},
            "หัน": {"visual": ["turning head", "glancing", "looking towards", "rotating"]},
            "เดิน": {"visual": ["walking", "walk", "stepping", "pedestrian movement", "moving forward"]},
            "ก้าว": {"visual": ["stepping", "taking steps", "walking"]},
            "วิ่ง": {"visual": ["running", "jogging", "sprint", "fast movement"]},
            "กระโดด": {"visual": ["jumping", "leaping", "hop", "in mid-air"]},
            "ล้ม": {"visual": ["falling down", "tripping", "collapsing to ground"]},
            "หกล้ม": {"visual": ["falling down", "tripping over", "sprawling on floor"]},

            # Object Manipulation & Tool Interactions
            "หยิบ": {"visual": ["picking up", "grabbing", "taking", "reaching for object", "hand holding"]},
            "จับ": {"visual": ["holding", "grasping", "gripping", "clutching"]},
            "วาง": {"visual": ["placing down", "putting down", "setting on table"]},
            "เปิด": {"visual": ["opening", "unlocking", "turning on"]},
            "ปิด": {"visual": ["closing", "shutting", "turning off"]},
            "ดื่ม": {"visual": ["drinking", "cup", "bottle", "sip", "water glass", "bringing to mouth"]},
            "กิน": {"visual": ["eating", "food plate", "spoon", "fork", "chewing", "meal"]},
            "ทาน": {"visual": ["eating food", "dining", "meal", "holding utensils"]},
            "เท": {"visual": ["pouring liquid", "pouring into cup", "spilling"]},
            "คน": {"visual": ["stirring", "mixing with spoon", "swirling"]},
            "หั่น": {"visual": ["cutting with knife", "slicing", "chopping food"]},
            "เขียน": {"visual": ["writing", "holding pen", "notebook", "paper", "whiteboard"]},
            "จด": {"visual": ["taking notes", "writing on notepad", "pen and paper"]},
            "พิมพ์": {"visual": ["typing on keyboard", "fingers on laptop", "computer desk"]},
            "ถือ": {"visual": ["carrying", "holding in hands", "bearing"]},
            "ส่ง": {"visual": ["passing object", "handing over", "giving to someone"]},
            "รับ": {"visual": ["receiving object", "accepting item", "reaching out hands"]},
            "ซ่อม": {"visual": ["repairing", "fixing with tools", "wrench", "screwdriver", "inspecting engine"]},
            "เช็ด": {"visual": ["wiping with cloth", "cleaning surface", "polishing"]},
            "กวาด": {"visual": ["sweeping with broom", "cleaning floor"]},

            # Transport & Vehicles
            "รถ": {"visual": ["car", "automobile", "vehicle", "street", "road traffic"]},
            "รถยนต์": {"visual": ["car", "automobile", "sedan", "vehicle"]},
            "มอเตอร์ไซค์": {"visual": ["motorcycle", "motorbike", "scooter", "two-wheeler"]},
            "จักรยานยนต์": {"visual": ["motorcycle", "motorbike", "scooter"]},
            "จักรยาน": {"visual": ["bicycle", "bike", "cycling", "cyclist"]},
            "ขับรถ": {"visual": ["driving car", "steering wheel", "driver behind wheel"]},
            "เลี้ยว": {"visual": ["turning", "vehicle turning corner", "curving road"]},
            "เลี้ยวซ้าย": {"visual": ["turning left", "vehicle turning left at corner"]},
            "เลี้ยวขวา": {"visual": ["turning right", "vehicle turning right at intersection"]},
            "ตัดหน้า": {"visual": ["cutting in front", "sudden turn in front of vehicle", "near miss"]},
            "เบรก": {"visual": ["braking", "sudden stop", "vehicle stopping"]},
            "จอด": {"visual": ["parking", "parked car", "stopping at roadside"]},

            # Presentation, Screens & Devices
            "คอมพิวเตอร์": {"visual": ["computer", "laptop", "monitor", "pc desktop", "screen"]},
            "คอม": {"visual": ["computer", "laptop", "monitor", "display"]},
            "โน้ตบุ๊ก": {"visual": ["laptop", "notebook computer", "open laptop"]},
            "โทรศัพท์": {"visual": ["smartphone", "mobile phone", "holding phone", "screen"]},
            "มือถือ": {"visual": ["smartphone", "mobile phone", "cellphone"]},
            "สไลด์": {"visual": ["presentation display", "large screen", "projector screen", "monitor"]},
            "กราฟ": {"visual": ["chart", "bar chart", "diagram", "data plot", "visual graph"]},
            "รูปภาพ": {"visual": ["picture", "photograph", "image display"]}
        }

    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        Decomposes query into fine-grained visual terms and constructs
        bilingual visual search cues for SigLIP 2 and LanceDB FTS.
        """
        q_clean = query.strip()
        q_low = q_clean.lower()
        
        # 1. Regex tokenization
        raw_tokens = re.findall(r'\w+', q_low)
        
        visual_cues: Set[str] = set(raw_tokens)
        english_translations: List[str] = []
        action_cues: List[str] = []

        # 2. Longest-Match Substring Matching
        sorted_keys = sorted(self.concept_dict.keys(), key=lambda x: len(x), reverse=True)
        matched_keys = []

        for key in sorted_keys:
            if key in q_low:
                matched_keys.append(key)
                mapped = self.concept_dict[key]
                v_terms = mapped.get("visual", [])
                
                visual_cues.update(v_terms)
                if v_terms:
                    english_translations.append(v_terms[0])
                    # If it has action verbs, append to action cues
                    if any(act in v_terms[0] for act in ["ing", "turn", "walk", "run", "drink", "eat", "sit", "stand", "hold"]):
                        action_cues.append(v_terms[0])

        # 3. Construct Bilingual Visual Representation
        unique_en = []
        for en in english_translations:
            if en not in unique_en:
                unique_en.append(en)

        if unique_en:
            bilingual_suffix = " ".join(unique_en[:5])
            expanded_str = f"{q_clean} {bilingual_suffix}"
        else:
            expanded_str = q_clean

        return {
            "original_query": q_clean,
            "matched_concepts": matched_keys,
            "visual_keywords": list(visual_cues),
            "action_keywords": action_cues,
            "audio_keywords": [],  # Kept as empty list for backward compatibility
            "expanded_search_str": expanded_str
        }

    def get_tta_queries(self, query: str) -> List[Tuple[str, float]]:
        """
        Module 6: Multi-Query Test-Time Augmentation (TTA Ensemble).
        Generates 3 parallel query variations with consensus weights:
        1. Original Query (weight: 0.50)
        2. Bilingual Literal Query (weight: 0.30)
        3. Physical Action & Movement Query (weight: 0.20)
        """
        expanded = self.expand_query(query)
        q_clean = expanded["original_query"]
        expanded_str = expanded["expanded_search_str"]
        action_cues = expanded.get("action_keywords", [])
        
        tta_list = [(q_clean, 0.50)]
        
        if expanded_str != q_clean:
            tta_list.append((expanded_str, 0.30))
        else:
            tta_list.append((q_clean, 0.30))
            
        if action_cues:
            action_str = f"person performing {' '.join(action_cues[:4])}"
            tta_list.append((action_str, 0.20))
        else:
            tta_list.append((expanded_str, 0.20))
            
        return tta_list

# Global singleton
query_expander = VisualQueryDecomposer()
CrossModalQueryExpander = VisualQueryDecomposer
