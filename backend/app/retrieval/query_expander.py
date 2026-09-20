from typing import List, Dict, Set, Any, Tuple
import re

class VisualQueryDecomposer:
    """
    Deterministic visual query decomposer and semantic expander.
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

            # Common standalone object queries. These must reach the English
            # visual encoder instead of being left as Thai-only text.
            "แมว": {"visual": ["cat", "kitten", "domestic cat"]},
            "สุนัข": {"visual": ["dog", "puppy", "domestic dog"]},
            "หมา": {"visual": ["dog", "puppy", "domestic dog"]},
            "cat": {"visual": ["cat", "kitten", "domestic cat"]},
            "dog": {"visual": ["dog", "puppy", "domestic dog"]},

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
            # Vehicle motion is distinct from a person running. Keep these
            # phrases explicit so visual retrieval and the Qwen verifier do
            # not interpret รถยนต์วิ่งผ่าน as human running.
            "วิ่งผ่าน": {"visual": ["car driving past", "vehicle passing", "car moving along road"]},
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
            "คนส่วนผสม": {"visual": ["stirring", "mixing with spoon", "swirling"]},
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
            "ถนน": {"visual": ["road", "street", "highway", "roadway"]},
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
        
        # Keep expansion deterministic.  Set iteration made otherwise
        # identical queries produce different vectors and rankings between
        # processes.
        visual_cues: List[str] = []
        seen_cues: Set[str] = set()

        def add_cue(value: str) -> None:
            value = value.strip()
            if value and value not in seen_cues:
                seen_cues.add(value)
                visual_cues.append(value)

        for token in raw_tokens:
            add_cue(token)
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
                
                for term in v_terms:
                    add_cue(term)
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
            "visual_keywords": visual_cues,
            "action_keywords": action_cues,
            "expanded_search_str": expanded_str
        }

    def get_query_variants(self, query: str) -> List[Tuple[str, float]]:
        """Return deterministic original/bilingual/action variants.

        The original query is always retained so multilingual SigLIP receives
        the user's exact wording. Missing variants are omitted and remaining
        weights are normalized in their stable order.
        """
        expanded = self.expand_query(query)
        original = expanded["original_query"]
        bilingual = expanded["expanded_search_str"]
        actions = expanded.get("action_keywords", [])
        action_variant = "visual action: " + " ".join(actions[:6]) if actions else ""
        candidates = [(original, 0.60), (bilingual, 0.25), (action_variant, 0.15)]
        seen: Set[str] = set()
        available: List[Tuple[str, float]] = []
        for text, weight in candidates:
            text = text.strip()
            if text and text not in seen:
                seen.add(text)
                available.append((text, weight))
        total = sum(weight for _, weight in available) or 1.0
        return [(text, weight / total) for text, weight in available]

    def get_tta_queries(self, query: str) -> List[Tuple[str, float]]:
        """
        Backward-compatible alias for deterministic visual query variants.
        """
        return self.get_query_variants(query)

    def disentangle_query(self, query: str) -> Dict[str, str]:
        """
        Decomposes natural language query into 3 explicit semantic facets:
        1. Subject Entity (e.g. "person, bicyclist, teacher")
        2. Physical Motion Verb (e.g. "riding bicycle, turning, writing on board")
        3. Spatial Scene Context (e.g. "sidewalk, classroom, street")
        
        Returns:
            Dict with keys: 'subject', 'verb', and 'context'
        """
        q_clean = query.strip()
        q_low = q_clean.lower()

        # Bilingual Entity Mappings
        subject_map = {
            "คน": ["person", "people", "human", "pedestrian"],
            "ผู้ชาย": ["man", "male"],
            "ผู้หญิง": ["woman", "female"],
            "เด็ก": ["child", "kid"],
            "อาจารย์": ["teacher", "instructor", "lecturer"],
            "ครู": ["teacher", "instructor"],
            "ผู้บรรยาย": ["presenter", "speaker"],
            "นักเรียน": ["students", "pupils"],
            "นักศึกษา": ["college students", "students"],
            "รถ": ["car", "automobile", "vehicle"],
            "รถยนต์": ["car", "automobile", "vehicle"],
            "จักรยาน": ["bicycle", "bike", "cyclist"],
            "มอเตอร์ไซค์": ["motorcycle", "motorbike"],
            "person": ["person", "human", "pedestrian"],
            "people": ["people", "pedestrians"],
            "pedestrian": ["pedestrian", "person walking"],
            "cyclist": ["cyclist", "bicyclist", "bike rider"],
            "teacher": ["teacher", "instructor", "lecturer"],
            "lecturer": ["lecturer", "presenter"],
            "instructor": ["instructor", "teacher"],
            "students": ["students", "pupils"],
            "pupils": ["pupils", "students"],
            "car": ["car", "automobile", "vehicle"],
            "automobile": ["automobile", "car"],
            "vehicle": ["vehicle", "car"],
            "bicycle": ["bicycle", "bike"],
            "bike": ["bike", "bicycle"]
        }

        # Bilingual Action & Motion Verb Mappings
        verb_map = {
            "รถยนต์วิ่ง": ["car driving", "vehicle moving on road", "automobile driving"],
            "รถวิ่ง": ["car moving", "vehicle driving on road", "automobile moving"],
            "วิ่งผ่าน": ["vehicle driving past", "car moving past", "driving on road"],
            "ขับผ่าน": ["driving past", "car driving on road"],
            "แล่นผ่าน": ["moving past", "car driving by"],
            "แล่น": ["cruising", "car moving on road"],
            "กำลังเดิน": ["walking", "pedestrian walking"],
            "เดินผ่าน": ["walking past", "pedestrian walking"],
            "ขี่จักรยาน": ["cyclist riding bike", "person riding bicycle"],
            "ปั่นจักรยาน": ["cyclist riding bicycle", "bicyclist pedaling"],
            "ปั่น": ["riding bicycle", "cycling", "pedaling"],
            "ขี่": ["riding", "cycling"],
            "ขับ": ["driving", "moving on roadway"],
            "เลี้ยว": ["turning", "turning corner"],
            "เดิน": ["walking", "stepping forward"],
            "วิ่ง": ["running", "jogging", "moving fast"],
            "กระโดด": ["jumping", "in mid air"],
            "ล้ม": ["falling down"],
            "หกล้ม": ["falling down", "tripping"],
            "ยกมือ": ["raising hand", "hand raised"],
            "โบกมือ": ["waving hand"],
            "ชี้": ["pointing with finger"],
            "เขียน": ["writing", "writing on board"],
            "จด": ["taking notes"],
            "ดื่ม": ["drinking", "holding cup"],
            "กิน": ["eating food"],
            "หยิบ": ["picking up", "reaching for"],
            "จับ": ["holding", "grasping"],
            "วาง": ["placing down"],
            "นั่ง": ["sitting", "seated at desks"],
            "ยืน": ["standing", "standing upright in front"],
            "เปิด": ["opening"],
            "ปิด": ["closing"],
            "ตัดหน้า": ["cutting in front"],
            "เบรก": ["braking", "sudden stop"],
            "walking": ["walking", "stepping forward", "pedestrian moving"],
            "walk": ["walking", "moving on foot"],
            "running": ["running", "jogging", "moving fast"],
            "run": ["running"],
            "riding": ["riding bicycle", "cycling", "pedaling"],
            "ride": ["riding", "cycling"],
            "cycling": ["cycling", "riding bicycle"],
            "driving": ["driving", "moving vehicle", "cruising on road"],
            "drive": ["driving", "moving on road"],
            "moving": ["moving along roadway", "in motion"],
            "standing": ["standing upright", "standing in front of room"],
            "stand": ["standing", "lecturing"],
            "sitting": ["sitting down", "seated at classroom desks"],
            "seated": ["seated at desks", "sitting attending class"],
            "sit": ["sitting", "seated"]
        }

        # Bilingual Spatial Scene & Context Mappings
        context_map = {
            "distance": ["in distance", "far away", "distant view"],
            "ทางไกล": ["in distance", "far background"],
            "ทางเท้า": ["sidewalk", "pedestrian walkway", "pavement"],
            "ริมถนน": ["roadside", "sidewalk along road", "street curb"],
            "ถนน": ["street roadway", "road asphalt", "traffic street"],
            "ห้องเรียน": ["classroom", "school lecture room", "indoor classroom"],
            "หน้าห้อง": ["in front of classroom", "near blackboard", "lecturing area"],
            "หน้าชั้น": ["front of classroom", "chalkboard podium", "teacher area"],
            "สี่แยก": ["intersection", "crossroads"],
            "โต๊ะ": ["classroom desks", "school tables"],
            "เก้าอี้": ["chairs", "classroom seats"],
            "กระดาน": ["blackboard", "whiteboard", "chalkboard"],
            "คอมพิวเตอร์": ["computer screen", "desktop monitor"],
            "street": ["street roadway", "road asphalt", "outdoors"],
            "road": ["road street", "traffic lane", "roadway"],
            "sidewalk": ["sidewalk", "pedestrian walkway", "pavement"],
            "pavement": ["pavement", "sidewalk pathway"],
            "classroom": ["classroom", "school lecture hall", "indoor desks"],
            "front": ["front of classroom", "blackboard podium"],
            "desks": ["classroom desks", "school study desks", "tables"],
            "school": ["school classroom", "lecture hall"]
        }

        subject_tokens = []
        for sk in sorted(subject_map.keys(), key=lambda x: len(x), reverse=True):
            if sk in q_low:
                subject_tokens.extend(subject_map[sk])
                break

        verb_tokens = []
        for vk in sorted(verb_map.keys(), key=lambda x: len(x), reverse=True):
            if vk in q_low:
                verb_tokens.extend(verb_map[vk])
                break

        context_tokens = []
        for ck in sorted(context_map.keys(), key=lambda x: len(x), reverse=True):
            if ck in q_low:
                context_tokens.extend(context_map[ck])
                break

        expanded = self.expand_query(query)
        action_keywords = expanded.get("action_keywords", [])
        visual_keywords = expanded.get("visual_keywords", [])

        sub_str = " ".join(subject_tokens[:3]) if subject_tokens else (visual_keywords[0] if visual_keywords else "subject")
        verb_str = " ".join(verb_tokens[:3]) if verb_tokens else (action_keywords[0] if action_keywords else "action activity")
        ctx_str = " ".join(context_tokens[:3]) if context_tokens else "scene environment"

        return {
            "subject": f"{q_clean} {sub_str}".strip(),
            "verb": f"person or object {verb_str}".strip(),
            "context": f"{ctx_str} background setting".strip(),
        }

# Global singleton
query_expander = VisualQueryDecomposer()
