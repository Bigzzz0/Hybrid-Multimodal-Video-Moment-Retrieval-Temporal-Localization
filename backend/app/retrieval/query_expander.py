from typing import List, Dict, Set, Any
import re

class CrossModalQueryExpander:
    """
    SOTA Cross-Modal Query Expander.
    Expands natural language user queries into multi-modal semantic cues
    (visual object synonyms, colors, clothing, acoustic cues, and bilingual translations).
    """

    def __init__(self):
        # Semantic mapping dictionary for Thai and English domain concepts
        self.concept_dict: Dict[str, Dict[str, List[str]]] = {
            # Colors
            "สีเขียว": {"visual": ["green", "emerald", "green shirt", "green clothing"], "audio": ["เขียว", "green"]},
            "เขียว": {"visual": ["green", "emerald", "green shirt", "green clothing"], "audio": ["เขียว", "green"]},
            "สีแดง": {"visual": ["red", "crimson", "red shirt", "red clothing"], "audio": ["แดง", "red"]},
            "แดง": {"visual": ["red", "crimson", "red shirt", "red clothing"], "audio": ["แดง", "red"]},
            "สีน้ำตาล": {"visual": ["brown", "tan", "brown shirt", "brown polo", "khaki"], "audio": ["น้ำตาล", "brown"]},
            "น้ำตาล": {"visual": ["brown", "tan", "brown shirt", "brown polo", "khaki"], "audio": ["น้ำตาล", "brown"]},
            "สีดำ": {"visual": ["black", "dark", "black shirt", "black clothing"], "audio": ["ดำ", "black"]},
            "ดำ": {"visual": ["black", "dark", "black shirt", "black clothing"], "audio": ["ดำ", "black"]},
            "สีขาว": {"visual": ["white", "light", "white shirt", "white clothing"], "audio": ["ขาว", "white"]},
            "ขาว": {"visual": ["white", "light", "white shirt", "white clothing"], "audio": ["ขาว", "white"]},
            "สีน้ำเงิน": {"visual": ["blue", "navy", "dark blue", "blue shirt"], "audio": ["น้ำเงิน", "blue"]},
            "น้ำเงิน": {"visual": ["blue", "navy", "dark blue", "blue shirt"], "audio": ["น้ำเงิน", "blue"]},
            "สีฟ้า": {"visual": ["blue", "cyan", "sky blue", "blue shirt"], "audio": ["ฟ้า", "blue"]},
            "ฟ้า": {"visual": ["blue", "cyan", "sky blue", "blue shirt"], "audio": ["ฟ้า", "blue"]},
            "สีเหลือง": {"visual": ["yellow", "gold", "yellow shirt"], "audio": ["เหลือง", "yellow"]},
            "เหลือง": {"visual": ["yellow", "gold", "yellow shirt"], "audio": ["เหลือง", "yellow"]},
            "สีส้ม": {"visual": ["orange", "orange shirt"], "audio": ["ส้ม", "orange"]},
            "ส้ม": {"visual": ["orange", "orange shirt"], "audio": ["ส้ม", "orange"]},
            "สีชมพู": {"visual": ["pink", "rose", "pink shirt"], "audio": ["ชมพู", "pink"]},
            "ชมพู": {"visual": ["pink", "rose", "pink shirt"], "audio": ["ชมพู", "pink"]},
            "สีม่วง": {"visual": ["purple", "violet", "purple shirt"], "audio": ["ม่วง", "purple"]},
            "ม่วง": {"visual": ["purple", "violet", "purple shirt"], "audio": ["ม่วง", "purple"]},
            "สีเทา": {"visual": ["gray", "grey", "gray shirt"], "audio": ["เทา", "gray"]},
            "เทา": {"visual": ["gray", "grey", "gray shirt"], "audio": ["เทา", "gray"]},

            # Clothing & Appearance
            "เสื้อ": {"visual": ["shirt", "t-shirt", "polo", "clothing", "apparel", "top", "jacket"], "audio": ["เสื้อ", "shirt"]},
            "กางเกง": {"visual": ["pants", "trousers", "jeans", "shorts"], "audio": ["กางเกง", "pants"]},
            "แว่น": {"visual": ["glasses", "eyeglasses", "spectacles"], "audio": ["แว่น", "glasses"]},
            "แว่นตา": {"visual": ["glasses", "eyeglasses", "spectacles"], "audio": ["แว่น", "แว่นตา", "glasses"]},
            "หมวก": {"visual": ["hat", "cap", "helmet"], "audio": ["หมวก", "hat"]},
            "รองเท้า": {"visual": ["shoes", "sneakers", "boots"], "audio": ["รองเท้า", "shoes"]},

            # People & Subjects
            "คน": {"visual": ["person", "people", "man", "individual", "someone"], "audio": ["คน", "person"]},
            "ผู้ชาย": {"visual": ["man", "guy", "male", "gentleman", "boy"], "audio": ["ผู้ชาย", "man"]},
            "ชาย": {"visual": ["man", "guy", "male", "boy"], "audio": ["ชาย", "man"]},
            "หนุ่ม": {"visual": ["young man", "guy", "man"], "audio": ["หนุ่ม", "man"]},
            "ผู้หญิง": {"visual": ["woman", "female", "lady", "girl"], "audio": ["ผู้หญิง", "woman"]},
            "หญิง": {"visual": ["woman", "female", "lady", "girl"], "audio": ["หญิง", "woman"]},
            "สาว": {"visual": ["young woman", "girl", "woman"], "audio": ["สาว", "woman"]},
            "เด็ก": {"visual": ["child", "kid", "baby"], "audio": ["เด็ก", "kid"]},
            "อาจารย์": {"visual": ["teacher", "professor", "lecturer", "instructor", "presenter", "speaker"], "audio": ["อาจารย์", "ครู", "professor"]},
            "ครู": {"visual": ["teacher", "instructor", "lecturer"], "audio": ["ครู", "อาจารย์", "teacher"]},
            "ผู้บรรยาย": {"visual": ["speaker", "presenter", "lecturer"], "audio": ["ผู้บรรยาย", "speaker"]},
            "นักเรียน": {"visual": ["student", "pupil"], "audio": ["นักเรียน", "student"]},
            "นักศึกษา": {"visual": ["student", "university student"], "audio": ["นักศึกษา", "student"]},

            # Actions & Interactions
            "ใส่": {"visual": ["wearing", "dressed in", "wears"], "audio": ["ใส่", "wear"]},
            "สวม": {"visual": ["wearing", "dressed in", "putting on"], "audio": ["สวม", "wear"]},
            "นั่ง": {"visual": ["sitting", "seated", "sits", "chair", "desk"], "audio": ["นั่ง", "sit"]},
            "ยืน": {"visual": ["standing", "stands", "upright"], "audio": ["ยืน", "stand"]},
            "ยิ้ม": {"visual": ["smiling", "smiles", "grin", "happy"], "audio": ["ยิ้ม", "smile"]},
            "หัวเราะ": {"visual": ["laughing", "laughs", "chuckle"], "audio": ["หัวเราะ", "laugh"]},
            "ยกมือ": {"visual": ["raising hand", "hand raised", "reaches hand", "gesturing"], "audio": ["ยกมือ", "ถาม", "hand"]},
            "หัน": {"visual": ["turning", "looking", "glancing"], "audio": ["หัน", "มอง", "look"]},
            "ดื่ม": {"visual": ["drinking", "cup", "bottle", "sip", "water", "glass"], "audio": ["กลืน", "ดื่ม", "น้ำ", "drink", "water"]},
            "กิน": {"visual": ["eating", "food", "plate", "spoon", "fork", "meal"], "audio": ["กิน", "อร่อย", "ทาน", "eat", "food"]},
            "ทาน": {"visual": ["eating", "food", "plate", "meal"], "audio": ["ทาน", "กิน", "eat"]},
            "พูด": {"visual": ["speaking", "talking", "microphone", "speaker", "presenter"], "audio": ["สวัสดี", "พูด", "กล่าว", "explain", "discuss", "presentation"]},
            "บรรยาย": {"visual": ["presenting", "speaker", "presentation", "lecture"], "audio": ["บรรยาย", "อธิบาย", "lecture", "present"]},
            "อธิบาย": {"visual": ["explaining", "pointing", "presentation"], "audio": ["อธิบาย", "กล่าว", "explain"]},
            "สไลด์": {"visual": ["slide", "presentation", "screen", "chart", "diagram", "table", "monitor"], "audio": ["สไลด์", "กราฟ", "ภาพนี้", "ตามตาราง", "slide", "chart", "figure"]},
            "กราฟ": {"visual": ["chart", "bar chart", "line plot", "figure", "diagram"], "audio": ["กราฟ", "ร้อยละ", "เปอร์เซ็นต์", "แกน", "axis", "percentage", "trend"]},
            "รถ": {"visual": ["car", "vehicle", "automobile", "road", "street", "traffic"], "audio": ["เสียงรถ", "เครื่องยนต์", "car", "engine", "drive"]},
            "รถยนต์": {"visual": ["car", "vehicle", "automobile", "road"], "audio": ["รถ", "รถยนต์", "car"]},
            "มอเตอร์ไซค์": {"visual": ["motorcycle", "motorbike", "scooter", "bike"], "audio": ["มอเตอร์ไซค์", "จักรยานยนต์", "motorcycle"]},
            "จักรยานยนต์": {"visual": ["motorcycle", "motorbike", "scooter", "bike"], "audio": ["จักรยานยนต์", "มอเตอร์ไซค์", "motorcycle"]},
            "เลี้ยว": {"visual": ["turning", "turns", "swerve", "corner"], "audio": ["เลี้ยว", "turn"]},
            "ตัดหน้า": {"visual": ["cut in front", "cutting off", "sudden turn"], "audio": ["ตัดหน้า", "cut in front"]},
            "เดิน": {"visual": ["walking", "walk", "feet", "pedestrian", "path"], "audio": ["เดิน", "ก้าว", "walk", "step"]},
            "วิ่ง": {"visual": ["running", "jogging", "sprint"], "audio": ["วิ่ง", "run"]},
            "เขียน": {"visual": ["writing", "pen", "paper", "whiteboard", "typing"], "audio": ["เขียน", "จด", "write", "note", "type"]},
            "จด": {"visual": ["writing", "taking notes", "pen", "paper"], "audio": ["จด", "เขียน", "note"]},
            "คอมพิวเตอร์": {"visual": ["computer", "laptop", "monitor", "screen", "pc", "desktop"], "audio": ["คอม", "คอมพิวเตอร์", "computer"]},
            "คอม": {"visual": ["computer", "laptop", "monitor", "screen"], "audio": ["คอม", "computer"]}
        }

    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        Expands user query with Thai sub-word segmentation and English cross-lingual translations.
        """
        q_clean = query.strip()
        q_low = q_clean.lower()
        
        # 1. Regex tokenization for spaced/English tokens
        raw_tokens = re.findall(r'\w+', q_low)
        
        visual_cues: Set[str] = set(raw_tokens)
        audio_cues: Set[str] = set(raw_tokens)
        english_translations: List[str] = []

        # 2. Thai sub-string dictionary matching (Longest Match First)
        sorted_keys = sorted(self.concept_dict.keys(), key=lambda x: len(x), reverse=True)
        matched_keys = []

        for key in sorted_keys:
            if key in q_low:
                matched_keys.append(key)
                mapped = self.concept_dict[key]
                v_terms = mapped.get("visual", [])
                a_terms = mapped.get("audio", [])
                
                visual_cues.update(v_terms)
                audio_cues.update(a_terms)
                
                # Add primary English translation
                if v_terms:
                    english_translations.append(v_terms[0])

        # 3. Construct clean Bilingual Search String for SigLIP 2
        # Example: "คนใส่เสื้อสีเขียว" -> "คนใส่เสื้อสีเขียว person wearing green shirt"
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
            "audio_keywords": list(audio_cues),
            "expanded_search_str": expanded_str
        }

query_expander = CrossModalQueryExpander()
