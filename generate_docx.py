# -*- coding: utf-8 -*-
import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
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

def set_callout_box(cell, bg_hex="F8FAFC", border_color="1E3A8A"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_hex}"/>')
    tcPr.append(shd)
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="18" w:space="0" w:color="{border_color}"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)

def set_table_borders(table, color="CBD5E1", sz="4"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
            <w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def format_run(run, font_name="TH Sarabun New", size_pt=14, bold=False, italic=False, color_rgb=(15, 23, 42)):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)
    rPr = run._r.get_or_add_rPr()
    rFonts = parse_xml(f'<w:rFonts {nsdecls("w")} w:ascii="{font_name}" w:hAnsi="{font_name}" w:cs="{font_name}"/>')
    rPr.append(rFonts)

def add_p(doc, text="", size_pt=14, bold=False, italic=False, color_rgb=(15, 23, 42), 
          align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=3, line_spacing=1.15):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = line_spacing
    if text:
        run = p.add_run(text)
        format_run(run, size_pt=size_pt, bold=bold, italic=italic, color_rgb=color_rgb)
    return p

def main():
    doc = docx.Document()

    # Compact margins (0.75 in)
    for section in doc.sections:
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Header
    p_top = add_p(doc, "GE362785 การคิดเชิงสร้างสรรค์และการแก้ปัญหา | โมดูล 3 การคิดเชิงระบบ", 
                  size_pt=12, bold=True, color_rgb=(100, 116, 139), align=WD_ALIGN_PARAGRAPH.RIGHT, after=2)

    p_title = add_p(doc, "กิจกรรมที่ 3.2 การพัฒนาคิดเชิงระบบ (งานกลุ่ม)", 
                    size_pt=16, bold=True, color_rgb=(30, 58, 138), align=WD_ALIGN_PARAGRAPH.CENTER, after=4)

    # Info grid
    tbl_info = doc.add_table(rows=2, cols=2)
    tbl_info.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_info, color="E2E8F0", sz="4")
    
    r0 = tbl_info.rows[0].cells
    r0[0].width = Inches(3.5)
    r0[1].width = Inches(3.5)
    format_run(r0[0].paragraphs[0].add_run("กลุ่มการเรียนที่: ........ กลุ่มย่อยที่: ........"), size_pt=13, color_rgb=(71, 85, 105))
    format_run(r0[1].paragraphs[0].add_run("ลำดับที่: ........ วันที่ส่ง: ...................."), size_pt=13, color_rgb=(71, 85, 105))
    
    r1 = tbl_info.rows[1].cells
    r1[0].width = Inches(3.5)
    r1[1].width = Inches(3.5)
    format_run(r1[0].paragraphs[0].add_run("ชื่อ-นามสกุล: ...................................................."), size_pt=13, color_rgb=(71, 85, 105))
    format_run(r1[1].paragraphs[0].add_run("รหัสนักศึกษา: ...................................."), size_pt=13, color_rgb=(71, 85, 105))

    add_p(doc, after=4)

    # Selected Situation Box
    tbl_sit = doc.add_table(rows=1, cols=1)
    tbl_sit.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_sit = tbl_sit.rows[0].cells[0]
    c_sit.width = Inches(7.0)
    set_callout_box(c_sit, bg_hex="EFF6FF", border_color="2563EB")
    set_cell_margins(c_sit, top=80, bottom=80, left=140, right=140)

    p = c_sit.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    format_run(p.add_run("สถานการณ์ที่กลุ่มเลือก: "), size_pt=14, bold=True, color_rgb=(30, 58, 138))
    format_run(p.add_run("สถานการณ์ที่ 5 การออกแบบภายในอัจฉริยะ (Smart Interior Design)\n"), size_pt=14, bold=True, color_rgb=(15, 23, 42))
    
    format_run(p.add_run("ชื่อผลงาน: "), size_pt=13, bold=True, color_rgb=(30, 58, 138))
    format_run(p.add_run("ระบบพื้นที่พักอาศัยอัจฉริยะเพื่อการดูแลผู้สูงอายุ พร้อมการสืบค้นวิดีโอเหตุการณ์ด้วย AI (Smart Living & Video Retrieval)\n"), size_pt=13, color_rgb=(15, 23, 42))

    format_run(p.add_run("★ กลุ่มผู้ใช้งานหลัก (Primary User): "), size_pt=13, bold=True, color_rgb=(185, 28, 28))
    format_run(p.add_run("บุตรหลานวัยทำงาน (Caregiver) ที่ต้องดูแลผู้สูงอายุที่พักอาศัยอยู่บ้านเพียงลำพังระหว่างวัน"), size_pt=13, bold=True, color_rgb=(15, 23, 42))

    add_p(doc, after=4)

    # -------------------------------------------------------------
    # ข้อ 1. ที่มาของแนวความคิดในการออกแบบ
    # -------------------------------------------------------------
    add_p(doc, "1. ที่มาของแนวความคิดในการออกแบบ", size_pt=15, bold=True, color_rgb=(30, 58, 138), before=4, after=3)
    
    reasons = [
        ("กลุ่มเป้าหมายและปัญหาที่แท้จริง: ", "บุตรหลานวัยทำงานกังวลเรื่องอุบัติเหตุของผู้สูงอายุ (เช่น การหกล้ม, ลืมทานยา) แต่กล้อง CCTV แบบเดิมค้นหาย้อนหลังยากมาก ต้องเสียเวลานั่งกรอฟุตเทจ 24 ชม. ทำให้ช่วยเหลือไม่ทันท่วงที"),
        ("ความต้องการด้านพื้นที่อยู่อาศัย: ", "ผู้สูงอายุรู้สึกอึดอัดที่ถูกกล้องวงจรปิดขนาดใหญ่จับจ้องตลอดเวลา จึงต้องการการออกแบบภายในที่กลมกลืนและไม่รบกวนสายตา"),
        ("แนวคิดการแก้ปัญหาเชิงระบบ: ", "ออกแบบตกแต่งภายในโดยซ่อนกล้อง AI ไว้ในโคมไฟและบัวผนังอย่างแนบเนียน เชื่อมโยงกับระบบ Video Event Retrieval ให้บุตรหลานพิมพ์ค้นหาเหตุการณ์เป็นข้อความ (เช่น 'แม่หกล้ม', 'ลุกจากเตียงตอนไหน') แล้วดึงคลิปช่วงเกิดเหตุ (Moment) พร้อม Heatmap แจ้งเตือนได้ทันทีในไม่กี่วินาที")
    ]
    for title, desc in reasons:
        p = add_p(doc, after=3)
        p.paragraph_format.left_indent = Inches(0.2)
        format_run(p.add_run("• " + title), size_pt=13, bold=True, color_rgb=(15, 23, 42))
        format_run(p.add_run(desc), size_pt=13, color_rgb=(51, 65, 85))

    add_p(doc, after=4)

    # -------------------------------------------------------------
    # ข้อ 2. แบบร่าง/ภาพประกอบพร้อมระบุ รายละเอียด
    # -------------------------------------------------------------
    add_p(doc, "2. แบบร่าง/ภาพประกอบพร้อมระบุ รายละเอียด", size_pt=15, bold=True, color_rgb=(30, 58, 138), before=4, after=3)

    img_path = r"C:\Users\User\.gemini\antigravity-ide\brain\60c5a9ba-45ef-43ae-b251-2de50b069a5d\smart_interior_sketch_1788357370005.jpg"
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(2)
        p_img.paragraph_format.space_after = Pt(2)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(5.6))

        p_cap = add_p(doc, "ภาพ: ผังการออกแบบตกแต่งภายในอัจฉริยะ (Ambient Interior) และหน้าจอระบบสืบค้นวิดีโอ (Video Event Retrieval)", 
                      size_pt=11, italic=True, color_rgb=(100, 116, 139), align=WD_ALIGN_PARAGRAPH.CENTER, after=4)

    # Compact Detail Box
    tbl_d = doc.add_table(rows=1, cols=1)
    tbl_d.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_d = tbl_d.rows[0].cells[0]
    c_d.width = Inches(7.0)
    set_callout_box(c_d, bg_hex="F8FAFC", border_color="0284C7")
    set_cell_margins(c_d, top=60, bottom=60, left=120, right=120)

    specs = [
        ("1. Hidden Ambient Cameras & Sensors: ", "ติดตั้งกล้อง AI มุมกว้างซ่อนในรางโคมไฟเพดานและชั้นวางของ กลมกลืนกับห้อง ไม่รบกวนสายตา"),
        ("2. Safe Interior Layout: ", "เฟอร์นิเจอร์ขอบมน พื้นกันกระแทกไร้ธรณีประตู พร้อมไฟทางเดินกลางคืนเปิดอัตโนมัติเมื่อหย่อนเท้าลงพื้น"),
        ("3. Natural Language Search (Web UI): ", "บุตรหลานพิมพ์ค้นหาวิดีโอด้วยภาษาพูด เช่น 'คนล้มในห้องนั่งเล่น', 'ยายทานข้าวตอนกี่โมง'"),
        ("4. Timeline Heatmap & Moment Playback: ", "แถบสีแสดงช่วงเวลาผิดปกติ กดเล่นคลิปสั้นช่วงเกิดเหตุได้ทันที พร้อมปุ่มส่งสัญญาณฉุกเฉิน")
    ]
    for title, desc in specs:
        p = c_d.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Inches(0.15)
        format_run(p.add_run("✔ " + title), size_pt=12, bold=True, color_rgb=(2, 132, 199))
        format_run(p.add_run(desc), size_pt=12, color_rgb=(51, 65, 85))

    add_p(doc, after=4)

    # -------------------------------------------------------------
    # ข้อ 3. ผลกระทบเชิงบวกและเชิงลบ
    # -------------------------------------------------------------
    add_p(doc, "3. ผลกระทบเชิงบวกและเชิงลบ (Systems Thinking Analysis)", size_pt=15, bold=True, color_rgb=(30, 58, 138), before=4, after=3)

    tbl_imp = doc.add_table(rows=3, cols=2)
    tbl_imp.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_imp, color="CBD5E1", sz="4")

    # Header
    h = tbl_imp.rows[0].cells
    h[0].width = Inches(3.5)
    h[1].width = Inches(3.5)
    set_cell_background(h[0], "1E3A8A")
    set_cell_background(h[1], "1E3A8A")
    set_cell_margins(h[0], 60, 60, 100, 100)
    set_cell_margins(h[1], 60, 60, 100, 100)
    format_run(h[0].paragraphs[0].add_run("ผลกระทบเชิงบวก (+)"), size_pt=13, bold=True, color_rgb=(255, 255, 255))
    format_run(h[1].paragraphs[0].add_run("ผลกระทบเชิงลบ (-) และแนวทางแก้ไข"), size_pt=13, bold=True, color_rgb=(255, 255, 255))

    # Row 1
    r1 = tbl_imp.rows[1].cells
    r1[0].width = Inches(3.5)
    r1[1].width = Inches(3.5)
    set_cell_margins(r1[0], 60, 60, 100, 100)
    set_cell_margins(r1[1], 60, 60, 100, 100)
    format_run(r1[0].paragraphs[0].add_run("• ผู้ใช้หลัก (ลูกหลาน): ตรวจสอบเหตุการณ์ได้ในไม่กี่วินาที ไม่ต้องนั่งเฝ้าจอกล้อง ช่วยลดความเครียดในการทำงาน\n• ผู้สูงอายุ: ได้รับความช่วยเหลือทันท่วงที และไม่รู้สึกอึดอัดเหมือนถูกจับผิด"), size_pt=12, color_rgb=(22, 101, 52))
    format_run(r1[1].paragraphs[0].add_run("• ประเด็นความเป็นส่วนตัว (Privacy):\n→ แก้ไข: ไม่ติดตั้งในห้องน้ำ, ประมวลผลแบบ Edge-AI ในบ้าน ไม่ส่งภาพสดขึ้น Cloud สาธารณะ"), size_pt=12, color_rgb=(153, 27, 27))

    # Row 2
    r2 = tbl_imp.rows[2].cells
    r2[0].width = Inches(3.5)
    r2[1].width = Inches(3.5)
    set_cell_background(r2[0], "F8FAFC")
    set_cell_background(r2[1], "F8FAFC")
    set_cell_margins(r2[0], 60, 60, 100, 100)
    set_cell_margins(r2[1], 60, 60, 100, 100)
    format_run(r2[0].paragraphs[0].add_run("• ความปลอดภัยทางกายภาพ: การจัดวางพื้นที่และไฟนำทางช่วยป้องกันการสะดุดล้มตอนกลางคืนได้อย่างมีประสิทธิภาพ"), size_pt=12, color_rgb=(22, 101, 52))
    format_run(r2[1].paragraphs[0].add_run("• การแจ้งเตือนผิดพลาด (False Alarm) & ต้นทุน:\n→ แก้ไข: มี Timeline Heatmap ให้ยืนยันผล และทำเป็นชุด Modular Kit ติดตั้งเสริมกับเฟอร์นิเจอร์เดิมได้"), size_pt=12, color_rgb=(153, 27, 27))

    # Save compact docx
    out_path = os.path.join(os.getcwd(), "GE362785_Activity_3.2_Smart_Interior_Design.docx")
    doc.save(out_path)
    print(f"Generated compact docx at: {out_path}")

if __name__ == "__main__":
    main()
