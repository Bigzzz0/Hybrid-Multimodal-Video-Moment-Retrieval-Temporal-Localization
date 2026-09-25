# -*- coding: utf-8 -*-
"""
Script to generate the Senior Project Proposal (.docx) matching
'เอกสารเค้าโครงโครงงาน.pdf' 100% in typography, margins, structure, and academic content,
with natural Thai left-alignment (no awkward stretched spaces) and cleaned parenthetical explanations.
"""
import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

# Ensure UTF-8 console output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FONT_NAME = "TH Sarabun New"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=50, bottom=50, left=70, right=70):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def set_table_borders(table, color="94A3B8", sz="4", inside_h="single", inside_v="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:right w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideH w:val="{inside_h}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideV w:val="{inside_v}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def make_table_robust(table):
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    if len(table.rows) > 0:
        trPr0 = table.rows[0]._tr.get_or_add_trPr()
        trPr0.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

def format_run(run, font_name=FONT_NAME, size_pt=16, bold=False, italic=False, color_rgb=(0, 0, 0)):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)
    rPr = run._r.get_or_add_rPr()
    rFonts = parse_xml(f'<w:rFonts {nsdecls("w")} w:ascii="{font_name}" w:hAnsi="{font_name}" w:cs="{font_name}" w:eastAsia="{font_name}"/>')
    rPr.append(rFonts)

def add_p(doc, text="", size_pt=16, bold=False, italic=False, color_rgb=(0, 0, 0),
          align=WD_ALIGN_PARAGRAPH.LEFT, first_line_indent=0, left_indent=0,
          before=0, after=2, line_spacing=1.15, keep_with_next=False):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = line_spacing
    if keep_with_next:
        p.paragraph_format.keep_with_next = True
    if first_line_indent > 0:
        p.paragraph_format.first_line_indent = Inches(first_line_indent)
    if left_indent > 0:
        p.paragraph_format.left_indent = Inches(left_indent)
    if text:
        run = p.add_run(text)
        format_run(run, size_pt=size_pt, bold=bold, italic=italic, color_rgb=color_rgb)
    return p

def create_proposal_document(output_paths):
    doc = docx.Document()

    # Exact Thesis / Proposal Margins (matching PDF):
    # A4: 8.27 x 11.69 inches. Top 1.5 in (108 pt), Left 1.5 in (108 pt), Right 1.0 in (72 pt), Bottom 1.0 in (72 pt)
    # Available content width: 8.27 - 1.5 - 1.0 = 5.77 inches
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(1.5)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.5)
        section.right_margin = Inches(1.0)
        section.different_first_page_header_footer = True

    # =========================================================================
    # PAGE 1: COVER PAGE (หน้าปกเค้าโครงโครงงาน)
    # Calibrated carefully so all content fits on Page 1 strictly!
    # =========================================================================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_dir, "kku_logo.png")
    if os.path.exists(logo_path):
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_before = Pt(0)
        p_logo.paragraph_format.space_after = Pt(8)
        run_logo = p_logo.add_run()
        run_logo.add_picture(logo_path, width=Inches(1.15))

    add_p(doc, "CS2568/หมายเลขกลุ่ม", size_pt=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    add_p(doc, "เค้าโครงโครงงานคอมพิวเตอร์", size_pt=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=12)

    add_p(doc, "ระบบสืบค้นและระบุช่วงเวลาในวิดีโอด้วยภาษาธรรมชาติแบบไฮบริดหลายมิติ", 
          size_pt=20, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    add_p(doc, "(Pure-Visual Video Moment Retrieval and Temporal Localization System)", 
          size_pt=20, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=16)

    add_p(doc, "โดย", size_pt=18, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
    add_p(doc, "รหัสนักศึกษา .................... นาย/นางสาว ....................................", 
          size_pt=18, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
    add_p(doc, "รหัสนักศึกษา .................... นาย/นางสาว ....................................", 
          size_pt=18, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=18)

    p_adv = add_p(doc, "อาจารย์ที่ปรึกษา  ผู้ช่วยศาสตราจารย์ ดร. สิลดา อินทรโสธรฉันท์", 
                  size_pt=18, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, left_indent=0.5, after=24)

    add_p(doc, "รายงานนี้เป็นส่วนหนึ่งของการศึกษาวิชา CP353761 สัมมนาทางวิทยาการคอมพิวเตอร์", 
          size_pt=16, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=1)
    add_p(doc, "ภาคเรียนที่ 1 ปีการศึกษา 2568", 
          size_pt=16, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=1)
    add_p(doc, "สาขาวิชาวิทยาการคอมพิวเตอร์ วิทยาลัยการคอมพิวเตอร์", 
          size_pt=16, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=1)
    add_p(doc, "มหาวิทยาลัยขอนแก่น", 
          size_pt=16, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=1)
    add_p(doc, "(เดือน กันยายน พ.ศ. 2568)", 
          size_pt=16, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)

    doc.add_page_break()

    # =========================================================================
    # PAGE 2: CONTENT PAGES (การเสนอเค้าโครงโครงงานคอมพิวเตอร์)
    # =========================================================================
    add_p(doc, "การเสนอเค้าโครงโครงงานคอมพิวเตอร์", size_pt=18, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2, keep_with_next=True)
    add_p(doc, "สาขาวิชาวิทยาการคอมพิวเตอร์ วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น", 
          size_pt=16, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=10, keep_with_next=True)

    # Student & Advisor Metadata block
    p_s1 = doc.add_paragraph()
    p_s1.paragraph_format.line_spacing = 1.15
    p_s1.paragraph_format.space_after = Pt(2)
    format_run(p_s1.add_run("ชื่อ  "), bold=True)
    format_run(p_s1.add_run("นาย / นางสาว .......................................... รหัสประจำตัว ............................"))

    p_s1e = doc.add_paragraph()
    p_s1e.paragraph_format.line_spacing = 1.15
    p_s1e.paragraph_format.space_after = Pt(2)
    p_s1e.paragraph_format.left_indent = Inches(0.35)
    format_run(p_s1e.add_run("Mr. / Miss .........................................................................................................."))

    p_s2 = doc.add_paragraph()
    p_s2.paragraph_format.line_spacing = 1.15
    p_s2.paragraph_format.space_after = Pt(2)
    p_s2.paragraph_format.left_indent = Inches(0.35)
    format_run(p_s2.add_run("นาย / นางสาว .......................................... รหัสประจำตัว ............................"))

    p_s2e = doc.add_paragraph()
    p_s2e.paragraph_format.line_spacing = 1.15
    p_s2e.paragraph_format.space_after = Pt(2)
    p_s2e.paragraph_format.left_indent = Inches(0.35)
    format_run(p_s2e.add_run("Mr. / Miss .........................................................................................................."))

    p_deg = doc.add_paragraph()
    p_deg.paragraph_format.line_spacing = 1.15
    p_deg.paragraph_format.space_after = Pt(2)
    format_run(p_deg.add_run("นักศึกษาระดับปริญญาตรี  "), bold=True)
    format_run(p_deg.add_run("หลักสูตรวิทยาการคอมพิวเตอร์"))

    p_adv1 = doc.add_paragraph()
    p_adv1.paragraph_format.line_spacing = 1.15
    p_adv1.paragraph_format.space_after = Pt(2)
    format_run(p_adv1.add_run("อาจารย์ที่ปรึกษาโครงงาน  "), bold=True)
    format_run(p_adv1.add_run("ผู้ช่วยศาสตราจารย์ ดร. สิลดา อินทรโสธรฉันท์"))

    p_adv2 = doc.add_paragraph()
    p_adv2.paragraph_format.line_spacing = 1.15
    p_adv2.paragraph_format.space_after = Pt(8)
    format_run(p_adv2.add_run("Project Advisor  "), bold=True)
    format_run(p_adv2.add_run("Asst. Prof. Dr. Silada Indrasothornsunt"))

    # -------------------------------------------------------------------------
    # 1. ชื่อหัวข้อโครงงาน
    # -------------------------------------------------------------------------
    add_p(doc, "1.  ชื่อหัวข้อโครงงาน", size_pt=16, bold=True, before=4, after=2, keep_with_next=True)
    add_p(doc, "ภาษาไทย: ระบบสืบค้นและระบุช่วงเวลาในวิดีโอด้วยภาษาธรรมชาติแบบไฮบริดหลายมิติ", 
          size_pt=16, left_indent=0.35, after=2)
    add_p(doc, "ภาษาอังกฤษ: Pure-Visual Video Moment Retrieval and Temporal Localization System", 
          size_pt=16, left_indent=0.35, after=6)

    # -------------------------------------------------------------------------
    # 2. หลักการและเหตุผล
    # -------------------------------------------------------------------------
    add_p(doc, "2.  หลักการและเหตุผล", size_pt=16, bold=True, before=4, after=2, keep_with_next=True)
    add_p(doc, "ในยุคสารสนเทศปัจจุบัน ปริมาณข้อมูลสื่อวิดีโอความยาวสูงเติบโตขึ้นอย่างก้าวกระโดดในทุกภาคส่วน ทั้งในด้านการศึกษา เช่น วิดีโอบันทึกการเรียนการสอนและการนำเสนอทางวิชาการ ด้านการทำงาน เช่น วิดีโอบันทึกการประชุม ตลอดจนด้านความปลอดภัยและนิติวิทยาศาสตร์ เช่น ฟุตเทจจากกล้องวงจรปิดและกล้องติดหน้ารถยนต์ อย่างไรก็ตาม ปัญหาคอขวดสำคัญที่ผู้ใช้งานต้องเผชิญคือภาระในการค้นหาและระบุตำแหน่งช่วงเวลาที่เกิดเหตุการณ์เฉพาะเจาะจง ซึ่งผู้ใช้งานจำเป็นต้องเสียเวลาเปิดรับชมหรือเลื่อนแถบเวลายาวนานหลายสิบนาทีถึงหลายชั่วโมงเพื่อค้นหาเหตุการณ์สั้น ๆ เพียงไม่กี่วินาที", 
          size_pt=16, first_line_indent=0.5, after=3)
    add_p(doc, "ระบบสืบค้นวิดีโอแบบดั้งเดิมส่วนใหญ่ยังคงพึ่งพาเฉพาะข้อมูลกำกับภายนอก เช่น ชื่อไฟล์ แท็ก หรือคำอธิบายภาพรวม ซึ่งไม่สามารถเข้าถึงเนื้อหาเชิงลึกในระดับช่วงเวลา และไม่สามารถเข้าใจคำค้นหาภาษาธรรมชาติที่ซับซ้อนได้ เช่น \"ช่วงที่อาจารย์เริ่มอธิบายสไลด์กราฟแท่งเปรียบเทียบผลลัพธ์\" หรือ \"ตอนที่มีคนสวมเสื้อสีแดงเดินเข้ามาหยิบกระเป๋าบนโต๊ะ\" หรือ \"ฉากที่รถจักรยานยนต์เลี้ยวตัดหน้ากะทันหันก่อนถึงทางแยก\" การสืบค้นเหตุการณ์เหล่านี้จำเป็นต้องอาศัยความเข้าใจร่วมกันระหว่างภาพนิ่ง การกระทำต่อเนื่อง และความสัมพันธ์เชิงเวลาอย่างลึกซึ้ง นอกจากนี้บริการสืบค้นวิดีโอบนคลาวด์ในปัจจุบันส่วนใหญ่มีค่าใช้จ่ายตามปริมาณการใช้งาน และต้องส่งข้อมูลวิดีโอออกไปประมวลผลภายนอกองค์กร ซึ่งอาจก่อให้เกิดความเสี่ยงด้านความเป็นส่วนตัวของข้อมูลที่มีความอ่อนไหว เช่น วิดีโอการประชุมภายในหรือฟุตเทจจากกล้องรักษาความปลอดภัย", 
          size_pt=16, first_line_indent=0.5, after=3)
    add_p(doc, "ในช่วงปี ค.ศ. 2024–2026 วงการปัญญาประดิษฐ์ได้พัฒนาแบบจำลองพื้นฐานด้านภาษาและภาพที่ก้าวหน้าอย่างมาก ได้แก่ SigLIP 2 พัฒนาโดย Google DeepMind ซึ่งรองรับสถาปัตยกรรม NaFlex ช่วยรักษารายละเอียดภาพและตำแหน่งเชิงพื้นที่ในวิดีโออัตราส่วน 16:9 ได้อย่างสมบูรณ์โดยไม่ต้องตัดขอบหรือบิดเบือนภาพ ควบคู่กับแบบจำลองภาษาภาพขนาดใหญ่ เช่น Qwen2.5-VL และ CapRL-Qwen3VL-4B ที่สามารถสร้างคำบรรยายฉากภาพแบบละเอียดและตรวจสอบความถูกต้องของเหตุการณ์ได้ ตลอดจนฐานข้อมูลเวกเตอร์ประสิทธิภาพสูงอย่าง LanceDB ที่ทำงานบนสถาปัตยกรรม Apache Arrow และรองรับการค้นหาแบบไฮบริดทั้งเวกเตอร์ภาพและการค้นหาข้อความเต็มรูปแบบด้วย BM25", 
          size_pt=16, first_line_indent=0.5, after=3)
    add_p(doc, "โครงงานนี้จึงนำเสนอการวิจัยและพัฒนาระบบสืบค้นและระบุตำแหน่งช่วงเวลาในวิดีโอด้วยภาษาธรรมชาติแบบไฮบริดหลายมิติ ที่ทำงานบนเครื่องประมวลผลส่วนบุคคลทั้งหมดโดยไม่ต้องส่งข้อมูลออกสู่คลาวด์ภายนอก ช่วยรักษาความเป็นส่วนตัวของข้อมูลและประหยัดต้นทุนได้อย่างเด็ดขาด โดยผสานการดึงข้อมูลเวกเตอร์ภาพและคำบรรยายฉากด้วย Reciprocal Rank Fusion (RRF) ร่วมกับการเกลี่ยสัญญาณเวลา 1D Gaussian Convolution และการสกัดขอบเขตเวลาอย่างแม่นยำด้วย Soft-NMS และ Platt Calibration", 
          size_pt=16, first_line_indent=0.5, after=6)

    # -------------------------------------------------------------------------
    # 3. วัตถุประสงค์ของโครงงาน
    # -------------------------------------------------------------------------
    add_p(doc, "3.  วัตถุประสงค์ของโครงงาน", size_pt=16, bold=True, before=4, after=2, keep_with_next=True)
    add_p(doc, "3.1  เพื่อออกแบบและพัฒนาระบบสกัดข้อมูลวิดีโอเร่งความเร็วด้วยฮาร์ดแวร์ โดยใช้ Decord สำหรับถอดรหัสวิดีโอบน GPU ควบคู่กับการตัดแบ่งฉากด้วย PySceneDetect เพื่อลดปริมาณเฟรมซ้ำซ้อน", 
          size_pt=16, left_indent=0.35, after=2)
    add_p(doc, "3.2  เพื่อประยุกต์ใช้แบบจำลองภาษาภาพบนเครื่องประมวลผลส่วนบุคคล ได้แก่ SigLIP 2 NaFlex สำหรับการสกัดเวกเตอร์ตัวแทนของเฟรมภาพและคำค้นหา และ CapRL-Qwen3VL-4B สำหรับสร้างคำบรรยายฉากและตรวจสอบความถูกต้องเชิงลึก", 
          size_pt=16, left_indent=0.35, after=2)
    add_p(doc, "3.3  เพื่อพัฒนาฐานข้อมูลเวกเตอร์และระบบสืบค้นประสิทธิภาพสูง โดยใช้ LanceDB บนโครงสร้าง Apache Arrow ร่วมกับดัชนี Disk-based IVF-PQ และการค้นหาข้อความเต็มรูปแบบด้วย Tantivy BM25", 
          size_pt=16, left_indent=0.35, after=2)
    add_p(doc, "3.4  เพื่อพัฒนาระบบคำนวณและระบุขอบเขตช่วงเวลาเหตุการณ์ ที่ผสานคะแนนหลายมิติด้วย Reciprocal Rank Fusion, การเกลี่ยสัญญาณเวลา 1D Gaussian Convolution, การสกัดขอบเขตช่วงเวลาด้วย Soft-NMS, และการปรับเทียบความเชื่อมั่นด้วย Platt Calibration", 
          size_pt=16, left_indent=0.35, after=2)
    add_p(doc, "3.5  เพื่อพัฒนาเว็บแอปพลิเคชันส่วนต่อประสานผู้ใช้แบบโต้ตอบ ด้วย Next.js และ FastAPI ที่มีแถบแสดงความหนาแน่นของความเกี่ยวข้องแบบ 1-Hz Relevance Heatmap และระบบเล่นวิดีโอแบบเลื่อนไปยังช่วงเวลาเป้าหมายอัตโนมัติ", 
          size_pt=16, left_indent=0.35, after=2)
    add_p(doc, "3.6  เพื่อประเมินประสิทธิภาพของระบบอย่างครอบคลุม ทั้งบนชุดข้อมูลมาตรฐานสากล QVHighlights และ Charades-STA, ชุดข้อมูลวิดีโอสถานการณ์จริง 30 ชั่วโมง, และการทดสอบกับกลุ่มผู้ใช้งานจริง", 
          size_pt=16, left_indent=0.35, after=6)

    # -------------------------------------------------------------------------
    # 4. ทฤษฎีและผลงานวิจัยที่เกี่ยวข้อง
    # -------------------------------------------------------------------------
    add_p(doc, "4.  ทฤษฎีและผลงานวิจัยที่เกี่ยวข้อง", size_pt=16, bold=True, before=4, after=2, keep_with_next=True)
    add_p(doc, "การศึกษาทฤษฎีและงานวิจัยที่เกี่ยวข้องเพื่อการพัฒนาโครงงานฉบับนี้ ได้ทำการรวบรวม วิเคราะห์ และสังเคราะห์องค์ความรู้อย่างเป็นระบบ โดยครอบคลุมทั้งวรรณกรรมวิจัยสำคัญตามกรอบการคิดวิเคราะห์ 5W1H (Who, What, Where, When, Why, How), ทฤษฎีและหลักการทางคณิตศาสตร์ที่เกี่ยวข้อง, และตารางสังเคราะห์วรรณกรรม (Synthesis Matrix) ดังมีรายละเอียดต่อไปนี้:", 
          size_pt=16, first_line_indent=0.5, after=3)

    add_p(doc, "4.1  งานวิจัยที่เกี่ยวข้อง (การสังเคราะห์วรรณกรรมตามกรอบ 5W1H)", size_pt=16, bold=True, left_indent=0.2, after=2, keep_with_next=True)
    add_p(doc, "จากการทบทวนวรรณกรรมวิจัยสำคัญจำนวน 12 งานวิจัย สามารถจำแนกออกเป็น 4 กลุ่มหลักได้ดังนี้:", 
          size_pt=16, first_line_indent=0.5, after=2)

    # 4.1.1
    add_p(doc, "4.1.1  งานวิจัยด้านการโมเดลความสัมพันธ์เชิงเวลาและการตรวจจับด้วย Transformer และ DETR", size_pt=16, bold=True, left_indent=0.35, after=2, keep_with_next=True)
    add_p(doc, "1) โครงข่าย 2D-TAN (Zhang et al., 2020): นำเสนอโครงข่าย 2D Temporal Adjacent Networks เพื่อแก้ปัญหา 1D Sliding Window ที่ขาดการมองเห็นความสัมพันธ์ของช่วงเวลาเหตุการณ์ซ้อนทับ โดยแปลงวิดีโอเป็น 2D Temporal Feature Map ตามแกนเริ่มต้นและแกนสิ้นสุด แล้วใช้ 2D Convolution ดึงบริบทเชิงเวลาก่อนจับคู่กับคำค้นหา", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "2) ชุดข้อมูล QVHighlights และแบบจำลอง Moment-DETR (Lei et al., 2021): เสนอเกณฑ์การประเมินช่วงเวลาแบบ Fine-grained Saliency Score รายวินาที โดยใช้โครงสร้าง DEtection TRansformer ทำนายพิกัดช่วงเวลาระหว่างจุดเริ่มต้นถึงจุดสิ้นสุดและคะแนนความโดดเด่นตลอดคลิปไปพร้อมกัน", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "3) แบบจำลอง QD-DETR (Moon et al., 2023): พัฒนา Query-Dependent DETR เพื่อแก้ปัญหาการเข้ารหัสวิดีโอแบบไม่ขึ้นกับคำค้นหา โดยใช้ Cross-Attention ปรับแต่งการเข้ารหัสวิดีโอตั้งแต่ชั้นแรก และใช้ Negative Pairs กดคะแนนฉากที่ไม่เกี่ยวข้องลง", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "4) กรอบการทำงาน UniVTG (Lin et al., 2023): รวมศูนย์ภารกิจการสืบค้นช่วงเวลา การตรวจจับจุดเด่น และการสรุปวิดีโอเข้าด้วยกันบนแกนหลัก Transformer เดียวกัน รองรับการถ่ายโอนความรู้ข้ามภารกิจได้อย่างมีประสิทธิภาพ", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "5) สถาปัตยกรรม ActionFormer (Zhang et al., 2022): ออกแบบการตรวจจับเหตุการณ์แบบไม่ใช้ Anchor โดยใช้ Multi-scale 1D Temporal Convolution ผสานกับ Self-Attention เพื่อทำนายขอบเขตเวลาทั้งกิจกรรมสั้นและยาวได้อย่างยืดหยุ่น", 
          size_pt=16, left_indent=0.5, after=3)

    # 4.1.2
    add_p(doc, "4.1.2  งานวิจัยด้านแบบจำลองภาษาภาพและวิดีโอ", size_pt=16, bold=True, left_indent=0.35, after=2, keep_with_next=True)
    add_p(doc, "6) แบบจำลอง SigLIP 2 (Google DeepMind, 2025): แก้ปัญหาภาพวิดีโอ 16:9 ถูกบิดเบือนในแบบจำลองยุคแรกด้วยนวัตกรรม NaFlex ที่ประมวลผลภาพตามอัตราส่วนจริง พร้อมใช้ Pairwise Sigmoid Loss, Masked Prediction, และ Self-Distillation ทำให้เข้าใจตำแหน่งเชิงพื้นที่ได้เหนือกว่าเดิมอย่างมาก", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "7) แบบจำลอง Qwen2.5-VL และ CapRL-Qwen3VL-4B (2024–2025): แบบจำลองภาษาภาพขนาดใหญ่สำหรับสร้างคำบรรยายฉากภาพแบบละเอียด และทำหน้าที่ตรวจสอบเหตุการณ์เชิงลึก โดยส่งเฟรมภาพจริงพร้อมพิกัดเวลาเข้าไปให้แบบจำลองประเมินความสอดคล้อง", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "8) แบบจำลอง TimeChat (Ren et al., 2024): แก้ปัญหาการขาดความเข้าใจมิติเวลาของโมเดลภาษาขนาดใหญ่ โดยใช้ตัวเข้ารหัสเฟรมที่คำนึงถึงเวลาควบคู่กับ Time Binding Tokens เพื่อผูกมิติเวลาเข้ากับคำตอบของแบบจำลอง", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "9) แบบจำลอง Video-LLaVA (Lin et al., 2024): ผสานปริภูมิการแทนข้อมูลของภาพนิ่งและวิดีโอให้อยู่ในระนาบเดียวกัน ช่วยให้แบบจำลองเข้าใจความสัมพันธ์เชิงบริบทได้สมบูรณ์ยิ่งขึ้น", 
          size_pt=16, left_indent=0.5, after=3)

    # 4.1.3
    add_p(doc, "4.1.3  งานวิจัยด้านการตรวจสอบลำดับการกระทำเชิงภาพและการตรวจจับรอยต่อฉาก", size_pt=16, bold=True, left_indent=0.35, after=2, keep_with_next=True)
    add_p(doc, "10) สถาปัตยกรรม Two-Stage Retrieval: แยกการค้นหาผู้สมัครรอบแรกความเร็วสูงด้วยเวกเตอร์และ BM25 ออกจากการตรวจสอบเชิงลึกด้วยแบบจำลองภาษาภาพ เพื่อควบคุมเวลาและทรัพยากรให้อยู่ในเกณฑ์ที่เหมาะสม", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "11) แบบจำลอง ImageBind (Girdhar et al., 2023): พิสูจน์แนวคิดการใช้ภาพเป็นแกนกลางในการเชื่อมโยงข้อมูลหลายรูปแบบในปริภูมิร่วมเดียวกัน ซึ่งเป็นรากฐานสำคัญของการสืบค้นแบบไฮบริด", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "12) สถาปัตยกรรม TransNet V2 (Souček & Lokoč, 2020): โครงข่าย 3D CNN ตรวจจับรอยต่อฉากด้วยความเร็วสูงกว่า 500 เฟรมต่อวินาที เพื่อตัดแบ่งวิดีโอและลดปริมาณเฟรมซ้ำซ้อนลงกว่าร้อยละ 75", 
          size_pt=16, left_indent=0.5, after=4)

    # Table 4.1: 5W1H Summary Table
    add_p(doc, "ตารางที่ 4.1  สรุปการวิเคราะห์งานวิจัยที่เกี่ยวข้องด้วยกรอบ 5W1H", size_pt=15, bold=True, after=2, keep_with_next=True)
    t_5w1h = doc.add_table(rows=13, cols=6)
    t_5w1h.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_5w1h, color="94A3B8", sz="4")
    make_table_robust(t_5w1h)

    headers_5w1h = ["งานวิจัย", "ผู้พัฒนา", "ปี", "สาระสำคัญที่พัฒนา", "ปัญหาที่แก้ไข", "วิธีการหลัก"]
    col_widths_5w1h = [Inches(0.85), Inches(0.80), Inches(0.42), Inches(1.30), Inches(1.20), Inches(1.20)]

    hdr_cells = t_5w1h.rows[0].cells
    for i, title in enumerate(headers_5w1h):
        hdr_cells[i].width = col_widths_5w1h[i]
        set_cell_background(hdr_cells[i], "F1F5F9")
        set_cell_margins(hdr_cells[i], top=45, bottom=45, left=45, right=45)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_run(p.add_run(title), size_pt=13, bold=True)

    rows_data_5w1h = [
        ("2D-TAN", "Zhang et al.", "2020", "2D Temporal Adjacent Networks", "1D Window ขาดความสัมพันธ์ข้ามเวลา", "2D Convolution บน Feature Map"),
        ("Moment-DETR", "Lei et al.", "2021", "Joint Moment & Saliency Grounding", "ขาดเกณฑ์ประเมินคะแนนระดับวินาที", "Transformer Direct Set Prediction"),
        ("QD-DETR", "Moon et al.", "2023", "Query-Dependent Video DETR", "ฉากที่ไม่เกี่ยวข้องได้คะแนนสูงเกินจริง", "Query Cross-Attention & Negative Loss"),
        ("UniVTG", "Lin et al.", "2023", "Unified Temporal Grounding", "การแยกพัฒนาแต่ละภารกิจย่อยในอดีต", "Unified Transformer Backbone"),
        ("ActionFormer", "Zhang et al.", "2022", "Anchor-free 1D Temporal Grounding", "กรอบเวลาแบบ Anchor ขาดความยืดหยุ่น", "Multi-scale 1D Temporal Pyramid"),
        ("SigLIP 2", "DeepMind", "2025", "NaFlex Vision-Language Model", "ภาพ 16:9 ถูกยืดและสูญเสียมิติพื้นที่", "NaFlex Dynamic Res & Sigmoid Loss"),
        ("Qwen2.5-VL", "Qwen Team", "2024", "Dense Caption + Verification", "VLM ทั่วไปประมวลผลช้าเกินไป", "Sequential Top-3 Candidates Verify"),
        ("TimeChat", "Ren et al.", "2024", "Time-Sensitive Video-LLM", "LLM ขาดความเข้าใจลำดับมิติเวลา", "Time Binding Tokens & Frame Encoder"),
        ("Video-LLaVA", "Lin et al.", "2024", "Unified Image-Video Representation", "การแยก Representation ภาพและวิดีโอ", "Joint Representation Alignment Layer"),
        ("Pure-Visual", "โครงงานนี้", "2026", "Dual-Channel Hybrid Retrieval", "การพึ่งพาเสียงหรือ Cloud API ราคาแพง", "Frame SigLIP 2 + Scene BM25 + RRF"),
        ("ImageBind", "Girdhar et al.", "2023", "One Embedding for 6 Modalities", "การผสานหลายมิติขาดคู่ข้อมูลจับคู่", "Image-Centric Joint Binding Space"),
        ("TransNet V2", "Souček et al.", "2020", "Fast Shot Boundary Detection", "การประมวลผลทุกเฟรมทำให้ระบบช้า", "Dilated 3D Convolutional Network")
    ]

    for r_idx, r_data in enumerate(rows_data_5w1h):
        row_cells = t_5w1h.rows[r_idx + 1].cells
        for c_idx, val in enumerate(r_data):
            row_cells[c_idx].width = col_widths_5w1h[c_idx]
            set_cell_margins(row_cells[c_idx], top=30, bottom=30, left=40, right=40)
            p = row_cells[c_idx].paragraphs[0]
            if c_idx in [0, 1, 2]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            format_run(p.add_run(val), size_pt=12, bold=(c_idx == 0))

    add_p(doc, after=5)

    # 4.2 ทฤษฎีและหลักการทางคณิตศาสตร์
    add_p(doc, "4.2  ทฤษฎีและหลักการทางคณิตศาสตร์ที่เกี่ยวข้อง", size_pt=16, bold=True, left_indent=0.2, after=2, keep_with_next=True)
    add_p(doc, "4.2.1  ทฤษฎีปริภูมิแฝงร่วมระหว่างภาพและภาษาและสถาปัตยกรรม NaFlex: แบบจำลองแปลงเฟรมภาพ f_t และคำค้นหาข้อความ Q ให้อยู่ในปริภูมิเวกเตอร์เดียวกันขนาด d = 768 มิติ โดยคำนวณความคล้ายคลึงด้วย Cosine Similarity:", 
          size_pt=16, first_line_indent=0.5, after=2)
    add_p(doc, "Cosine_Similarity(f_t, Q) = (v_t · u_q) / (||v_t|| × ||u_q||)", 
          size_pt=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    add_p(doc, "โดยที่ v_t คือเวกเตอร์แทนภาพจาก Vision Encoder และ u_q คือเวกเตอร์แทนข้อความจาก Text Encoder กลไก NaFlex ช่วยให้โมเดลรองรับภาพอัตราส่วน 16:9 ได้โดยตรงโดยไม่สูญเสียความละเอียดเชิงพื้นที่", 
          size_pt=16, first_line_indent=0.5, after=3)

    add_p(doc, "4.2.2  ทฤษฎีฟังก์ชันการสูญเสียแบบจับคู่ Pairwise Sigmoid Loss: SigLIP 2 ใช้ฟังก์ชันการสูญเสียแบบ Sigmoid แทน Softmax ดั้งเดิม ช่วยลดการพึ่งพา Global Batch Size และทำให้เวกเตอร์มีความเสถียรสูงขึ้น:", 
          size_pt=16, first_line_indent=0.5, after=2)
    add_p(doc, "L_SigLIP = - ∑_{i=1}^N ∑_{j=1}^N [ y_ij · log σ(t · u_i · v_j + b) + (1 - y_ij) · log(1 - σ(t · u_i · v_j + b)) ]", 
          size_pt=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    add_p(doc, "โดย y_ij = 1 สำหรับคู่ที่ตรงกัน และ y_ij = 0 สำหรับคู่ตรงข้าม, σ คือ Sigmoid Activation, t คือ Temperature, และ b คือ Bias ที่เรียนรู้ได้", 
          size_pt=16, first_line_indent=0.5, after=3)

    add_p(doc, "4.2.3  ทฤษฎีการรวมผลลัพธ์ด้วย Reciprocal Rank Fusion (RRF): ผสานผลลัพธ์จากสายเวกเตอร์ภาพและสายคำบรรยายฉากที่มีมาตรวัดต่างกันโดยไม่ต้องปรับสเกลคะแนน:", 
          size_pt=16, first_line_indent=0.5, after=2)
    add_p(doc, "RRF(d) = ∑_{m ∈ {Visual, Caption}} [ w_m / (k + r_m(d)) ]", 
          size_pt=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    add_p(doc, "โดย r_m(d) คืออันดับของข้อมูลในแต่ละช่องทาง กำหนด k = 60 เพื่อป้องกันอันดับแรกมีน้ำหนักสูงเกินไป และ w_m คือค่าน้ำหนัก (Visual = 0.80, Caption = 0.20)", 
          size_pt=16, first_line_indent=0.5, after=3)

    add_p(doc, "4.2.4  ทฤษฎีการสร้างช่วงเวลาผู้สมัครและการเกลี่ยสัญญาณเวลา 1D Gaussian Convolution: สร้างช่วงเวลาผู้สมัครที่ความยาวต่าง ๆ ตั้งแต่ 2 ถึง 32 วินาที โดยนำคะแนน S(t) มาผ่านตัวกรอง Gaussian Filter: S_smooth(t) = S(t) * G_σ เพื่อลดสัญญาณรบกวน จากนั้นขยายขอบเขตจากจุดสูงสุดเฉพาะที่จนคะแนนลดต่ำกว่าร้อยละ 60 ของจุดสูงสุด และใช้ Soft-NMS เพื่อตัดช่วงซ้ำซ้อน", 
          size_pt=16, first_line_indent=0.5, after=3)

    add_p(doc, "4.2.5  ทฤษฎีการปรับเทียบความเชื่อมั่นด้วย Platt Calibration: ใช้ฟังก์ชัน Logistic Sigmoid แปลงคะแนนดิบเป็นค่าความน่าจะเป็นเชิงสถิติ P(Y=1|S) = 1 / (1 + exp(A · S + B)) เพื่อตัดช่วงเวลาที่ไม่เกี่ยวข้องออกอย่างแม่นยำ", 
          size_pt=16, first_line_indent=0.5, after=3)

    add_p(doc, "4.2.6  ทฤษฎีการจัดเก็บและสร้างดัชนีเวกเตอร์แบบ Disk-based IVF-PQ: LanceDB ใช้สถาปัตยกรรม Columnar บน Apache Arrow ร่วมกับดัชนี Inverted File with Product Quantization (IVF-PQ) แบ่งเวกเตอร์ 768 มิติออกเป็นคลัสเตอร์และบีบอัดแต่ละส่วน ช่วยประหยัดหน่วยความจำได้ 8–16 เท่า และสืบค้นได้เร็วต่ำกว่า 5 มิลลิวินาที", 
          size_pt=16, first_line_indent=0.5, after=4)

    # Table 4.2: Synthesis Matrix Table
    add_p(doc, "ตารางที่ 4.2  ตารางสังเคราะห์วรรณกรรม (Synthesis Matrix) เปรียบเทียบสถาปัตยกรรมและมิติข้อมูล", size_pt=15, bold=True, after=2, keep_with_next=True)
    t_sm = doc.add_table(rows=7, cols=5)
    t_sm.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_sm, color="94A3B8", sz="4")
    make_table_robust(t_sm)

    headers_sm = ["แนวทาง / งานวิจัย", "มิติข้อมูล", "สถาปัตยกรรมหลัก", "จุดเด่นสำคัญ", "ข้อจำกัดและช่องว่างวิจัย"]
    col_widths_sm = [Inches(1.10), Inches(0.80), Inches(1.28), Inches(1.27), Inches(1.32)]

    hdr_cells_sm = t_sm.rows[0].cells
    for i, title in enumerate(headers_sm):
        hdr_cells_sm[i].width = col_widths_sm[i]
        set_cell_background(hdr_cells_sm[i], "F1F5F9")
        set_cell_margins(hdr_cells_sm[i], top=35, bottom=35, left=35, right=35)
        p = hdr_cells_sm[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_run(p.add_run(title), size_pt=12, bold=True)

    rows_data_sm = [
        ("Proposal-based (2D-TAN)", "Visual Only (2D CNN)", "Sliding Window & 2D Temporal Feature Map", "เข้าใจความสัมพันธ์ของช่วงเวลาซ้อนทับได้ดี", "ไม่รองรับ Open-Vocabulary คำนวณช้าบนวิดีโอยาว"),
        ("DETR-based (Moment-DETR)", "Visual + Text", "Transformer Encoder-Decoder Set Prediction", "ทำนายช่วงเวลาและ Highlight Score พร้อมกัน", "ต้อง Fine-tune บนชุดข้อมูลปิด กิน VRAM สูง"),
        ("Unified Grounding (UniVTG)", "Visual + Text", "Multi-scale 1D Temporal Convolution", "สกัดเหตุการณ์สั้นและยาวได้อย่างยืดหยุ่น", "ยังขาดระบบดัชนีเวกเตอร์ที่รองรับคลังวิดีโอใหญ่"),
        ("TimeChat (Video-LLM)", "Visual + Text", "Large Language Model & Time Tokens", "เข้าใจการกระทำซับซ้อนและตอบคำถามเหตุผลได้", "กินทรัพยากรสูง ประมวลผลช้า เหมาะกับ Re-rank"),
        ("Two-Stage (SeViLA)", "Multi-modal Binding", "Localizer-Filter + Answerer VLM", "ประหยัด Token ในงาน Video QA", "ไม่มี Vector DB ฝังตัวสำหรับค้นหาทันที"),
        ("Proposed Pure-Visual (โครงงานนี้)", "Visual Vector + Dense Caption", "SigLIP 2 NaFlex + Qwen + LanceDB", "รักษา Aspect Ratio, รองรับหลาย Moments, Local VRAM ≤ 12GB", "ต้องทำดัชนีล่วงหน้า และขึ้นกับความคมชัดของภาพ")
    ]

    for r_idx, r_data in enumerate(rows_data_sm):
        row_cells = t_sm.rows[r_idx + 1].cells
        for c_idx, val in enumerate(r_data):
            row_cells[c_idx].width = col_widths_sm[c_idx]
            set_cell_margins(row_cells[c_idx], top=30, bottom=30, left=35, right=35)
            p = row_cells[c_idx].paragraphs[0]
            if c_idx in [0, 1]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            format_run(p.add_run(val), size_pt=12, bold=(c_idx == 0))

    add_p(doc, after=4)
    add_p(doc, "4.3  บทวิเคราะห์สังเคราะห์วรรณกรรม: จากการทบทวนวรรณกรรมพบประเด็นสำคัญ 3 ประการ ได้แก่ (1) การผสานสัญญาณสองช่องทาง โดยเวกเตอร์ภาพระดับเฟรมช่วยครอบคลุมเส้นเวลา ส่วนคำบรรยายฉากช่วยจับบริบทการกระทำต่อเนื่อง (2) ความสำคัญของอัตราส่วนภาพ ซึ่งสถาปัตยกรรม NaFlex ช่วยแก้ปัญหาภาพบิดเบือนในวิดีโอ 16:9 ได้อย่างสมบูรณ์ และ (3) ความสมดุลด้านความเร็ว โดยการออกแบบระบบสองระยะที่แยกการค้นหารอบแรกระดับมิลลิวินาทีออกจากการตรวจทานเชิงลึก ช่วยให้ระบบตอบสนองผู้ใช้ได้ทันทีโดยไม่ติดขัด", 
          size_pt=16, first_line_indent=0.5, after=6)

    # -------------------------------------------------------------------------
    # 5. วิธีดำเนินการวิจัย
    # -------------------------------------------------------------------------
    add_p(doc, "5.  วิธีดำเนินการวิจัย", size_pt=16, bold=True, before=4, after=2, keep_with_next=True)
    add_p(doc, "โครงงานนี้ได้กำหนดขั้นตอนและระเบียบวิธีดำเนินการวิจัยออกเป็น 8 ขั้นตอนหลัก เพื่อให้การพัฒนาระบบเป็นไปอย่างมีประสิทธิภาพและบรรลุวัตถุประสงค์ที่ตั้งไว้ ดังนี้:", 
          size_pt=16, first_line_indent=0.5, after=3)

    steps = [
        ("5.1  การศึกษาค้นคว้าทฤษฎีและเทคโนโลยีที่เกี่ยวข้อง", 
         "ศึกษาทฤษฎีด้านแบบจำลองภาษาภาพพื้นฐาน, การสืบค้นช่วงเวลาในวิดีโอด้วยภาษาธรรมชาติ, ฐานข้อมูลเวกเตอร์, และการระบุขอบเขตช่วงเวลาเชิงเวลา ตลอดจนศึกษาสถาปัตยกรรม SigLIP 2, Qwen2.5-VL, Decord, LanceDB, และ Next.js"),
        ("5.2  การออกแบบสถาปัตยกรรมระบบโดยรวม", 
         "ออกแบบสถาปัตยกรรมแบบสองช่องทางคู่ขนาน ประกอบด้วยระบบนำเข้าและสกัดข้อมูลวิดีโอ, ชั้นการจัดเก็บข้อมูลบน LanceDB, กลไกสืบค้นที่ผสาน RRF ร่วมกับ Gaussian Smoothing, และส่วนต่อประสานผู้ใช้"),
        ("5.3  การออกแบบโครงสร้างฐานข้อมูลเวกเตอร์ LanceDB", 
         "ออกแบบโครงสร้างตารางข้อมูลบน Apache Arrow ประกอบด้วยตาราง videos สำหรับจัดเก็บข้อมูลไฟล์, ตาราง video_frames_v2 สำหรับจัดเก็บเวกเตอร์ 768 มิติของเฟรมภาพ, ตาราง scenes_v2 สำหรับจัดเก็บคำบรรยายฉาก, และตาราง index_metadata สำหรับตรวจสอบความเข้ากันได้ของแบบจำลอง"),
        ("5.4  การพัฒนาระบบนำเข้าและประมวลผลวิดีโอแบบก้าวหน้า", 
         "พัฒนาโมดูลถอดรหัสวิดีโอด้วย Decord ผ่านการเร่งความเร็วบน GPU, โมดูลตรวจจับรอยต่อฉากด้วย PySceneDetect, และโมดูลสกัดเวกเตอร์ SigLIP 2 ในระยะแรกเพื่อให้เริ่มค้นหาได้ทันที ก่อนส่งฉากไปสร้างคำบรรยายด้วยแบบจำลองภาษาภาพในเบื้องหลัง"),
        ("5.5  การพัฒนาระบบสืบค้นและระบุตำแหน่งเวลาแบบไฮบริด", 
         "พัฒนาระบบค้นหาคู่ขนานระหว่างการค้นหาเวกเตอร์ภาพและการค้นหาข้อความเต็มรูปแบบด้วย BM25 บนคำบรรยายฉาก ผสานคะแนนด้วย RRF, เกลี่ยสัญญาณด้วย 1D Gaussian Convolution, สกัดช่วงเวลาระหว่างจุดเริ่มต้นถึงจุดสิ้นสุดด้วย Soft-NMS, และปรับเทียบความเชื่อมั่นด้วย Platt Calibration"),
        ("5.6  การพัฒนาเว็บแอปพลิเคชันส่วนต่อประสานผู้ใช้", 
         "พัฒนาเว็บแอปพลิเคชันด้วย Next.js และ FastAPI รองรับการอัปโหลดวิดีโอ, แถบแสดงความหนาแน่นของความเกี่ยวข้องแบบ 1-Hz Relevance Heatmap, ระบบเล่นวิดีโอแบบเลื่อนไปยังช่วงเวลาเป้าหมายอัตโนมัติ, และแผงตรวจสอบการใช้งานหน่วยความจำ VRAM บน GPU"),
        ("5.7  การทดลองและประเมินประสิทธิภาพของระบบ", 
         "ทดสอบประสิทธิภาพบนชุดข้อมูลมาตรฐาน QVHighlights และ Charades-STA, ทดสอบบนวิดีโอสถานการณ์จริง 30 ชั่วโมง, ทำการทดลองแยกส่วนประกอบ 5 รูปแบบเพื่อพิสูจน์การทำงานของแต่ละองค์ประกอบ, และทดสอบความพึงพอใจกับกลุ่มผู้ใช้งานจริง 30 คน"),
        ("5.8  การสรุปผล การจัดทำรายงานวิจัย และการนำเสนอผลงาน", 
         "รวบรวมและวิเคราะห์ผลการทดลอง สรุปข้อค้นพบและข้อจำกัดของโครงงาน จัดทำเล่มรายงานวิจัยฉบับสมบูรณ์ จัดทำคู่มือการใช้งานระบบ และเตรียมเอกสารเพื่อนำเสนอผลงานและส่งบทความวิจัยตีพิมพ์")
    ]

    for title, desc in steps:
        add_p(doc, title, size_pt=16, bold=True, left_indent=0.35, after=2, keep_with_next=True)
        add_p(doc, desc, size_pt=16, left_indent=0.5, after=3)

    add_p(doc, after=4)

    # -------------------------------------------------------------------------
    # 6. ขอบเขตและข้อจำกัดของการวิจัย
    # -------------------------------------------------------------------------
    add_p(doc, "6.  ขอบเขตและข้อจำกัดของการวิจัย", size_pt=16, bold=True, before=4, after=2, keep_with_next=True)
    add_p(doc, "การกำหนดขอบเขตและข้อจำกัดของโครงงานวิจัย มีรายละเอียดดังนี้:", 
          size_pt=16, first_line_indent=0.5, after=2)

    add_p(doc, "6.1  ขอบเขตของการวิจัย:", size_pt=16, bold=True, left_indent=0.35, after=2, keep_with_next=True)
    add_p(doc, "1) ด้านรูปแบบไฟล์ข้อมูล: รองรับไฟล์วิดีโอดิจิทัลสกุล .mp4, .mov, .mkv, .webm ความละเอียดตั้งแต่ 720p ถึง 4K UHD ที่อัตราเฟรม 24–60 เฟรมต่อวินาที", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "2) ด้านประเภทเนื้อหาวิดีโอ: ครอบคลุม 3 กลุ่มหลัก ได้แก่ วิดีโอบรรยายการเรียนการสอนและการนำเสนอทางวิชาการ, วิดีโอบันทึกการประชุม, และฟุตเทจจากกล้องวงจรปิดหรือกล้องติดหน้ารถยนต์", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "3) ด้านแบบจำลองปัญญาประดิษฐ์: ประยุกต์ใช้ SigLIP 2 NaFlex สำหรับสกัดเวกเตอร์ภาพและข้อความ และ CapRL-Qwen3VL-4B สำหรับสร้างคำบรรยายฉากและตรวจสอบความถูกต้อง", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "4) ด้านฐานข้อมูลและการสืบค้น: ใช้ LanceDB บนโครงสร้าง Apache Arrow จัดเก็บดัชนีเวกเตอร์แบบ Disk-based IVF-PQ และดัชนีข้อความ Tantivy BM25 รองรับการสืบค้นแบบไฮบริดผสานด้วย RRF", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "5) ด้านสถาปัตยกรรมซอฟต์แวร์: พัฒนาส่วนบริการหลังบ้านด้วย FastAPI ร่วมกับ Decord และ PyTorch และพัฒนาส่วนหน้าบ้านด้วย Next.js พร้อมแถบ Canvas Heatmap แสดงผลการสืบค้น", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "6) ด้านการประมวลผล: ทำงานบนเครื่องประมวลผลส่วนบุคคลทั้งหมดที่มีหน่วยความจำ VRAM 12GB โดยไม่มีการพึ่งพาบริการคลาวด์ภายนอก", 
          size_pt=16, left_indent=0.5, after=3)

    add_p(doc, "6.2  ข้อจำกัดของการวิจัย:", size_pt=16, bold=True, left_indent=0.35, after=2, keep_with_next=True)
    add_p(doc, "1) ระบบทำงานบนพื้นฐานข้อมูลภาพแท้เท่านั้น จึงไม่ใช้สัญญาณเสียงพูดหรือข้อความตัวอักษรจากการอ่านภาพเป็นหลักฐานในการสืบค้น", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "2) ความแม่นยำในการค้นหาขึ้นอยู่กับความคมชัดของภาพ หากวิดีโอมีสภาพมืดจัด ภาพเบลอจากการเคลื่อนไหวรุนแรง หรือเหตุการณ์อยู่นอกมุมกล้อง ประสิทธิภาพอาจลดลง", 
          size_pt=16, left_indent=0.5, after=2)
    add_p(doc, "3) ระบบใช้การประมวลผลแบบก้าวหน้า ในช่วงแรกหลังนำเข้าวิดีโอทันที การค้นหาจะทำงานด้วยเวกเตอร์ภาพ SigLIP 2 เป็นหลัก จนกว่ากระบวนการสร้างคำบรรยายฉากในเบื้องหลังจะเสร็จสมบูรณ์", 
          size_pt=16, left_indent=0.5, after=6)

    # -------------------------------------------------------------------------
    # 7. สถานที่ทำวิจัย
    # -------------------------------------------------------------------------
    add_p(doc, "7.  สถานที่ทำวิจัย", size_pt=16, bold=True, before=4, after=2, keep_with_next=True)
    add_p(doc, "สาขาวิชาวิทยาการคอมพิวเตอร์ วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น", 
          size_pt=16, first_line_indent=0.5, after=6)

    # -------------------------------------------------------------------------
    # 8. ประโยชน์ที่คาดว่าจะได้รับ
    # -------------------------------------------------------------------------
    add_p(doc, "8.  ประโยชน์ที่คาดว่าจะได้รับ", size_pt=16, bold=True, before=4, after=2, keep_with_next=True)
    add_p(doc, "8.1  ได้ระบบซอฟต์แวร์ต้นแบบการสืบค้นและระบุช่วงเวลาในวิดีโอด้วยภาษาธรรมชาติแบบไฮบริด ที่ทำงานได้อย่างสมบูรณ์บนเครื่องประมวลผลส่วนบุคคล", 
          size_pt=16, left_indent=0.35, after=2)
    add_p(doc, "8.2  ช่วยลดระยะเวลาและภาระในการค้นหาเหตุการณ์สำคัญในวิดีโอความยาวสูงของผู้ใช้งานได้อย่างมีนัยสำคัญ โดยเปลี่ยนจากการเลื่อนแถบเวลาด้วยมือมาเป็นการค้นหาด้วยข้อความภาษาธรรมชาติและไปยังช่วงเวลาเป้าหมายได้ทันที", 
          size_pt=16, left_indent=0.35, after=2)
    add_p(doc, "8.3  ช่วยปกป้องความลับและความเป็นส่วนตัวของข้อมูลวิดีโอในองค์กร เนื่องจากไม่มีการส่งข้อมูลภาพออกสู่คลาวด์ภายนอก และช่วยประหยัดต้นทุนค่าบริการคลาวด์ได้อย่างเด็ดขาด", 
          size_pt=16, left_indent=0.35, after=2)
    add_p(doc, "8.4  ได้พิมพ์เขียวเชิงสถาปัตยกรรมและแนวทางการประยุกต์ใช้แบบจำลองภาษาภาพรุ่นใหม่บนฮาร์ดแวร์ระดับทั่วไป ซึ่งเป็นประโยชน์ต่อการศึกษาและการวิจัยทางด้านวิทยาการคอมพิวเตอร์ต่อไป", 
          size_pt=16, left_indent=0.35, after=2)
    add_p(doc, "8.5  ได้ผลงานวิจัยที่มีความพร้อมสำหรับการจัดทำบทความวิจัยเพื่อส่งตีพิมพ์ในการประชุมวิชาการระดับชาติด้านคอมพิวเตอร์", 
          size_pt=16, left_indent=0.35, after=6)

    # -------------------------------------------------------------------------
    # 9. แผนและระยะเวลาดำเนินการ (Start on a clean page to keep Table 1 intact!)
    # -------------------------------------------------------------------------
    doc.add_page_break()
    add_p(doc, "9.  แผนและระยะเวลาดำเนินการ", size_pt=16, bold=True, before=4, after=2, keep_with_next=True)
    add_p(doc, "ตารางการดำเนินงานและระยะเวลา จากขั้นตอนในวิธีการดำเนินการวิจัยมีระยะเวลาในการดำเนินการประมาณ 12 เดือน (ครอบคลุมปีการศึกษา 2568 ถึง 2569) ดังแสดงในตารางที่ 1:", 
          size_pt=16, first_line_indent=0.5, after=3, keep_with_next=True)

    add_p(doc, "ตารางที่ 1  แผนและระยะเวลาดำเนินการ", size_pt=15, bold=True, after=2, keep_with_next=True)

    t_plan = doc.add_table(rows=10, cols=13)
    t_plan.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_plan, color="94A3B8", sz="4")
    make_table_robust(t_plan)

    task_w = Inches(2.05)
    m_w = Inches(0.31)

    r0 = t_plan.rows[0].cells
    r0[0].width = task_w
    set_cell_background(r0[0], "F1F5F9")
    set_cell_margins(r0[0], top=35, bottom=35, left=35, right=35)
    p = r0[0].paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_run(p.add_run("การดำเนินงาน"), size_pt=13, bold=True)

    for c in range(1, 8):
        r0[c].width = m_w
        set_cell_background(r0[c], "F1F5F9")
        set_cell_margins(r0[c], top=30, bottom=30, left=15, right=15)
    r0[1].merge(r0[7])
    p = r0[1].paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_run(p.add_run("ปี 2568"), size_pt=13, bold=True)

    for c in range(8, 13):
        r0[c].width = m_w
        set_cell_background(r0[c], "F1F5F9")
        set_cell_margins(r0[c], top=30, bottom=30, left=15, right=15)
    r0[8].merge(r0[12])
    p = r0[8].paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_run(p.add_run("ปี 2569"), size_pt=13, bold=True)

    r1 = t_plan.rows[1].cells
    r1[0].width = task_w
    set_cell_background(r1[0], "F1F5F9")
    r0[0].merge(r1[0])

    month_labels = ["6", "7", "8", "9", "10", "11", "12", "1", "2", "3", "4", "5"]
    for idx, ml in enumerate(month_labels):
        cell = r1[idx + 1]
        cell.width = m_w
        set_cell_background(cell, "F8FAFC")
        set_cell_margins(cell, top=25, bottom=25, left=10, right=10)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_run(p.add_run(ml), size_pt=12, bold=True)

    plan_tasks = [
        ("1. ศึกษาค้นคว้าทฤษฎีและงานวิจัยที่เกี่ยวข้อง", [0, 1]),
        ("2. ออกแบบสถาปัตยกรรมและฐานข้อมูล LanceDB", [1, 2]),
        ("3. พัฒนาระบบนำเข้าวิดีโอแบบก้าวหน้า", [2, 3]),
        ("4. นำเสนอเค้าโครงโครงงาน (Proposal Defense) ★", [3]),
        ("5. พัฒนาระบบสืบค้นไฮบริดและปรับเทียบช่วงเวลา", [4, 5, 6]),
        ("6. พัฒนาส่วนต่อประสานผู้ใช้ Web Application", [6, 7, 8]),
        ("7. การทดลอง ประเมินผล และการทำ User Study", [8, 9, 10]),
        ("8. จัดทำรายงานฉบับสมบูรณ์และการสอบจบโครงงาน", [10, 11])
    ]

    for t_idx, (t_name, active_months) in enumerate(plan_tasks):
        row_cells = t_plan.rows[t_idx + 2].cells
        row_cells[0].width = task_w
        set_cell_margins(row_cells[0], top=30, bottom=30, left=35, right=35)
        p = row_cells[0].paragraphs[0]
        format_run(p.add_run(t_name), size_pt=12, bold=(t_idx == 3))

        for m_idx in range(12):
            cell = row_cells[m_idx + 1]
            cell.width = m_w
            set_cell_margins(cell, top=30, bottom=30, left=10, right=10)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if m_idx in active_months:
                if t_idx == 3:
                    set_cell_background(cell, "E0E7FF")
                    format_run(p.add_run("█"), size_pt=11, bold=True, color_rgb=(67, 56, 202))
                else:
                    set_cell_background(cell, "F1F5F9")
                    format_run(p.add_run("█"), size_pt=11, color_rgb=(71, 85, 105))

    add_p(doc, "หมายเหตุ  ปรับตามความเหมาะสม ทั้งนี้ต้องระบุช่วงที่เสนอเค้าโครงโครงงาน (★ สัญลักษณ์ระบุช่วงนำเสนอเค้าโครงโครงงานในเดือนกันยายน 2568)", 
          size_pt=14, italic=True, before=2, after=6)

    # -------------------------------------------------------------------------
    # 10. งบประมาณ (Clean page break to keep Table 1 unified and budget neat)
    # -------------------------------------------------------------------------
    doc.add_page_break()
    add_p(doc, "10.  งบประมาณ", size_pt=16, bold=True, before=4, after=4, keep_with_next=True)
    
    t_bg = doc.add_table(rows=8, cols=2)
    t_bg.alignment = WD_TABLE_ALIGNMENT.CENTER
    make_table_robust(t_bg)
    bg_widths = [Inches(4.5), Inches(1.27)]
    
    bg_data = [
        ("หมวดวัสดุอุปกรณ์", "", True),
        ("  -  ค่าวัสดุสำนักงาน (กระดาษ ปากกา แฟ้มเอกสาร และอื่นๆ)", "1,500 บาท", False),
        ("  -  ค่าวัสดุคอมพิวเตอร์ (แฟลชไดรฟ์ สายเชื่อมต่อ สื่อบันทึกข้อมูลสำรอง)", "2,500 บาท", False),
        ("หมวดค่าใช้สอย", "", True),
        ("  -  ค่าถ่ายเอกสารและพิมพ์เอกสารทางวิชาการ", "1,000 บาท", False),
        ("  -  ค่าจัดรูปเล่มเค้าโครงโครงงานและเล่มรายงานฉบับสมบูรณ์", "1,500 บาท", False),
        ("หมวดค่าใช้จ่ายอื่นๆ", "", True),
        ("  -  ค่าลงทะเบียนและเตรียมส่งบทความวิจัยในการประชุมวิชาการ", "3,500 บาท", False)
    ]
    
    for r_idx, (item_txt, cost_txt, is_hdr) in enumerate(bg_data):
        row_cells = t_bg.rows[r_idx].cells
        row_cells[0].width = bg_widths[0]
        row_cells[1].width = bg_widths[1]
        set_cell_margins(row_cells[0], top=20, bottom=20, left=20, right=20)
        set_cell_margins(row_cells[1], top=20, bottom=20, left=20, right=20)
        
        p0 = row_cells[0].paragraphs[0]
        p0.paragraph_format.line_spacing = 1.15
        p0.paragraph_format.space_after = Pt(1)
        format_run(p0.add_run(item_txt), size_pt=16, bold=is_hdr)
        
        p1 = row_cells[1].paragraphs[0]
        p1.paragraph_format.line_spacing = 1.15
        p1.paragraph_format.space_after = Pt(1)
        p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        format_run(p1.add_run(cost_txt), size_pt=16, bold=is_hdr)

    add_p(doc, after=2)
    add_p(doc, "รวมงบประมาณทั้งสิ้น (หนึ่งหมื่นบาทถ้วน)                                               10,000 บาท", 
          size_pt=16, bold=True, left_indent=0.35, after=6)

    # -------------------------------------------------------------------------
    # 11. เอกสารอ้างอิง
    # -------------------------------------------------------------------------
    add_p(doc, "11.  เอกสารอ้างอิง", size_pt=16, bold=True, before=4, after=2, keep_with_next=True)
    add_p(doc, "รายชื่อเอกสารอ้างอิง การอ้างอิงเอกสารให้อ้างอิงตามแบบสากล (รูปแบบ APA 7th Edition / IEEE):", 
          size_pt=16, first_line_indent=0.5, after=3, keep_with_next=True)

    references = [
        "Bai, S., et al. (2024). Qwen2.5-VL: Enhancing Vision-Language Models for Fine-Grained Multimodal Understanding and Localization. arXiv preprint.",
        "Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual machine learning methods for search result fusion. In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (pp. 758–759).",
        "Gao, J., Sun, C., Yang, Z., & Nevatia, R. (2017). TALL: Temporal activity localization via language query. In Proceedings of the IEEE International Conference on Computer Vision (ICCV) (pp. 5267–5275).",
        "Girdhar, R., El-Nouby, A., Liu, Z., Mann, M., Rabbat, M., Singh, M., ... & Misra, I. (2023). ImageBind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 15180–15190).",
        "Google DeepMind. (2025). SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic and Spatial Awareness. arXiv preprint arXiv:2502.14786.",
        "LanceDB Authors. (2024). LanceDB: Serverless, Developer-friendly Vector Database for Multimodal AI. LanceDB Whitepaper & Documentation.",
        "Lei, J., Berg, T. L., & Bansal, M. (2021). QVHighlights: Detecting moments and highlights in videos via natural language queries. In Advances in Neural Information Processing Systems (NeurIPS 2021) (Vol. 34, pp. 11846–11858).",
        "Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., & Yuan, L. (2024). Video-LLaVA: Learning united visual representation by alignment before projection. In Findings of the Association for Computational Linguistics: EMNLP 2024 (pp. 6784–6798).",
        "Lin, K. Q., Zhang, P., Chen, J., Pramanick, S., Gao, X., Dai, P., & Yan, S. (2023). UniVTG: Towards unified video-language temporal grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (pp. 2794–2804).",
        "Moon, W., Hyun, S., Park, S., Park, D., & Heo, J. P. (2023). Query-dependent video representation for moment retrieval and highlight detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 23023–23033).",
        "Ren, S., Yao, L., Li, Z., Sun, Y., & Zhu, X. (2024). TimeChat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 14313–14323).",
        "Souček, T., & Lokoč, J. (2020). TransNet V2: An effective deep network for fast video shot boundary detection. arXiv preprint arXiv:2008.05238.",
        "Zhang, C. L., Wu, J., & Li, Y. (2022). ActionFormer: Localizing moments of actions with transformers. In European Conference on Computer Vision (ECCV) (pp. 492–510). Springer.",
        "Zhang, S., Peng, H., Fu, J., & Luo, J. (2020). Learning 2D temporal adjacent networks for moment localization with natural language. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 07, pp. 12870–12877)."
    ]

    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.paragraph_format.space_after = Pt(1.5)
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)
        format_run(p_ref.add_run(ref), size_pt=14)

    # -------------------------------------------------------------------------
    # SIGNATURES SECTION (Kept together on a single clean page)
    # -------------------------------------------------------------------------
    doc.add_page_break()
    add_p(doc, "การรับรองและตรวจสอบเค้าโครงโครงงาน", size_pt=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=18, keep_with_next=True)

    p_sig1 = doc.add_paragraph()
    p_sig1.paragraph_format.left_indent = Inches(2.6)
    p_sig1.paragraph_format.space_after = Pt(2)
    p_sig1.paragraph_format.keep_with_next = True
    format_run(p_sig1.add_run("ลงชื่อผู้ทำโครงงาน ...................................................."))

    p_sig1_name = doc.add_paragraph()
    p_sig1_name.paragraph_format.left_indent = Inches(2.6)
    p_sig1_name.paragraph_format.space_after = Pt(2)
    p_sig1_name.paragraph_format.keep_with_next = True
    format_run(p_sig1_name.add_run("(นาย/นางสาว ............................................)"))

    p_sig1_date = doc.add_paragraph()
    p_sig1_date.paragraph_format.left_indent = Inches(2.6)
    p_sig1_date.paragraph_format.space_after = Pt(16)
    p_sig1_date.paragraph_format.keep_with_next = True
    format_run(p_sig1_date.add_run("วันที่ ...................................................."))

    p_sig2 = doc.add_paragraph()
    p_sig2.paragraph_format.left_indent = Inches(2.6)
    p_sig2.paragraph_format.space_after = Pt(2)
    p_sig2.paragraph_format.keep_with_next = True
    format_run(p_sig2.add_run("ลงชื่อผู้ทำโครงงาน ...................................................."))

    p_sig2_name = doc.add_paragraph()
    p_sig2_name.paragraph_format.left_indent = Inches(2.6)
    p_sig2_name.paragraph_format.space_after = Pt(2)
    p_sig2_name.paragraph_format.keep_with_next = True
    format_run(p_sig2_name.add_run("(นาย/นางสาว ............................................)"))

    p_sig2_date = doc.add_paragraph()
    p_sig2_date.paragraph_format.left_indent = Inches(2.6)
    p_sig2_date.paragraph_format.space_after = Pt(24)
    p_sig2_date.paragraph_format.keep_with_next = True
    format_run(p_sig2_date.add_run("วันที่ ...................................................."))

    # Advisor Inspection block matching template
    add_p(doc, "การตรวจสอบจากอาจารย์ที่ปรึกษาโครงงาน", size_pt=16, bold=True, left_indent=0.35, after=3, keep_with_next=True)
    add_p(doc, "(   ) ตรวจสอบแล้ว เห็นชอบตามเสนอ", size_pt=16, left_indent=0.35, after=3, keep_with_next=True)
    add_p(doc, "(   ) อื่น ๆ ……………………………………………………………………………………………………………………………………….", size_pt=16, left_indent=0.35, after=18, keep_with_next=True)

    p_adv_sig = doc.add_paragraph()
    p_adv_sig.paragraph_format.left_indent = Inches(2.6)
    p_adv_sig.paragraph_format.space_after = Pt(2)
    p_adv_sig.paragraph_format.keep_with_next = True
    format_run(p_adv_sig.add_run("(ลงชื่อ) .........................................................."))

    p_adv_sig_name = doc.add_paragraph()
    p_adv_sig_name.paragraph_format.left_indent = Inches(2.6)
    p_adv_sig_name.paragraph_format.space_after = Pt(2)
    p_adv_sig_name.paragraph_format.keep_with_next = True
    format_run(p_adv_sig_name.add_run("(ผู้ช่วยศาสตราจารย์ ดร. สิลดา อินทรโสธรฉันท์)"))

    p_adv_sig_date = doc.add_paragraph()
    p_adv_sig_date.paragraph_format.left_indent = Inches(2.6)
    p_adv_sig_date.paragraph_format.space_after = Pt(12)
    format_run(p_adv_sig_date.add_run("วันที่ ...................................................."))

    # -------------------------------------------------------------------------
    # APPENDICES (ภาคผนวก)
    # -------------------------------------------------------------------------
    doc.add_page_break()
    add_p(doc, "ภาคผนวก", size_pt=18, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=12, keep_with_next=True)

    # ภาคผนวก ก
    add_p(doc, "ภาคผนวก ก: รายละเอียดชุดข้อมูลทดสอบมาตรฐานสากล", size_pt=16, bold=True, before=4, after=3, keep_with_next=True)
    t_ds = doc.add_table(rows=4, cols=4)
    t_ds.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_ds, color="94A3B8", sz="4")
    make_table_robust(t_ds)
    hdr_ds = ["ชุดข้อมูลมาตรฐาน", "ผู้พัฒนา / ปี", "จำนวนคลิป / ความยาว", "ตัวชี้วัดหลักที่ใช้ประเมิน"]
    widths_ds = [Inches(1.4), Inches(1.2), Inches(1.4), Inches(1.77)]
    for i, title in enumerate(hdr_ds):
        c = t_ds.rows[0].cells[i]
        c.width = widths_ds[i]
        set_cell_background(c, "F1F5F9")
        set_cell_margins(c, top=40, bottom=40, left=50, right=50)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_run(p.add_run(title), size_pt=13, bold=True)

    data_ds = [
        ("QVHighlights", "Lei et al. (2021)", "10,148 วิดีโอ (~150 ชม.)", "R@1@IoU=0.5, 0.7, mIoU, HIT@1"),
        ("Charades-STA", "Gao et al. (2017)", "9,848 ช่วงเวลา (~82 ชม.)", "R@1@IoU=0.3, 0.5, 0.7, mIoU"),
        ("ActivityNet Captions", "Krishna et al. (2017)", "20,000 วิดีโอ (~849 ชม.)", "R@1@IoU=0.3, 0.5, mIoU")
    ]
    for r_idx, r_data in enumerate(data_ds):
        for c_idx, val in enumerate(r_data):
            c = t_ds.rows[r_idx + 1].cells[c_idx]
            c.width = widths_ds[c_idx]
            set_cell_margins(c, top=35, bottom=35, left=50, right=50)
            p = c.paragraphs[0]
            if c_idx in [0, 1]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            format_run(p.add_run(val), size_pt=12, bold=(c_idx == 0))

    add_p(doc, after=8)

    # ภาคผนวก ข
    add_p(doc, "ภาคผนวก ข: ข้อกำหนดฮาร์ดแวร์และการใช้หน่วยความจำ VRAM ของแต่ละแบบจำลอง", size_pt=16, bold=True, before=4, after=3, keep_with_next=True)
    t_hw = doc.add_table(rows=5, cols=4)
    t_hw.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_hw, color="94A3B8", sz="4")
    make_table_robust(t_hw)
    hdr_hw = ["แบบจำลอง AI / ไลบรารี", "ฟังก์ชันการทำงาน", "รูปแบบ Quantization / ความแม่นยำ", "Peak VRAM Budget"]
    widths_hw = [Inches(1.5), Inches(1.8), Inches(1.3), Inches(1.17)]
    for i, title in enumerate(hdr_hw):
        c = t_hw.rows[0].cells[i]
        c.width = widths_hw[i]
        set_cell_background(c, "F1F5F9")
        set_cell_margins(c, top=40, bottom=40, left=50, right=50)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_run(p.add_run(title), size_pt=13, bold=True)

    data_hw = [
        ("Decord (NVDEC)", "Hardware Video Decoding บน GPU", "CUDA Hardware Context", "~0.5 GB VRAM"),
        ("SigLIP 2 NaFlex", "สกัด Frame / Query Embeddings (768-dim)", "FP16 / Tensor Cores", "~2.2 GB VRAM"),
        ("CapRL-Qwen3VL-4B", "Scene Captioning & Accurate Verification", "GGUF Q6_K (llama.cpp CUDA)", "~4.8 GB VRAM"),
        ("LanceDB Serverless", "Vector Storage (IVF-PQ) & Tantivy FTS", "Zero-Copy Disk Memory Mapped", "~0.2 GB VRAM")
    ]
    for r_idx, r_data in enumerate(data_hw):
        for c_idx, val in enumerate(r_data):
            c = t_hw.rows[r_idx + 1].cells[c_idx]
            c.width = widths_hw[c_idx]
            set_cell_margins(c, top=35, bottom=35, left=50, right=50)
            p = c.paragraphs[0]
            if c_idx in [0, 2, 3]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            format_run(p.add_run(val), size_pt=12, bold=(c_idx == 0))

    # Clean Page Break for Appendix C
    doc.add_page_break()
    add_p(doc, "ภาคผนวก ค: สถานะการพัฒนาระบบจริงในปัจจุบัน", size_pt=16, bold=True, before=4, after=3, keep_with_next=True)
    add_p(doc, "ภาคผนวกนี้บันทึกสถานะของระบบที่ได้รับการติดตั้งและทดสอบจริงบนเครื่องคอมพิวเตอร์สำหรับการวิจัย (RTX 5070 12GB VRAM):", 
          size_pt=16, first_line_indent=0.5, after=3)
    add_p(doc, "• โครงสร้างระบบหลัก: ระบบใช้ SigLIP 2 NaFlex เป็นแกนหลักในการสืบค้น และใช้ CapRL-Qwen3VL-4B สำหรับสร้างคำบรรยายฉากและทำหน้าที่ตรวจสอบความถูกต้องผ่าน llama.cpp CUDA Worker", 
          size_pt=15, left_indent=0.35, after=2.5)
    add_p(doc, "• การจัดการ VRAM: จัดสรรงบประมาณหน่วยความจำจริง 11.8 GB VRAM ป้องกันปัญหาหน่วยความจำไม่เพียงพอ โดยแยกโพรเซสการทำงานระหว่าง FastAPI Web Server และ Inference Worker อย่างเด็ดขาด", 
          size_pt=15, left_indent=0.35, after=2.5)
    add_p(doc, "• การประมวลผลข้อมูลแบบก้าวหน้า: ภายหลังจากการสร้างดัชนีภาพ SigLIP 2 ในระยะแรกเสร็จสิ้น ผู้ใช้งานสามารถเริ่มทำการค้นหาวิดีโอได้ทันที จากนั้นระบบจะสร้างคำบรรยายฉากด้วย CapRL ในเบื้องหลังและอัปเดตข้อมูลเมื่อเสร็จสมบูรณ์", 
          size_pt=15, left_indent=0.35, after=4)

    for p in output_paths:
        doc.save(p)
        print(f"Saved: {p}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    targets = [
        os.path.join(base_dir, "เค้าโครงโครงงานคอมพิวเตอร์_Pure_Visual_Video_Moment_Retrieval.docx"),
        os.path.join(base_dir, "Assignment4_เค้าโครงโครงงานคอมพิวเตอร์.docx")
    ]
    create_proposal_document(targets)
