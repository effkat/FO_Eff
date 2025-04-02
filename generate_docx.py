from docx import Document
from docx.shared import Pt, Inches, RGBColor
import os
from fpdf import FPDF
from fpdf import HTMLMixin
from docx.oxml import OxmlElement, ns
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
import datetime
import pandas as pd


#class FlightReportDocx(FPDF)

def generate_docx_report(output_path):

    def header(route_name):
        """Setzt den Header mit dem Namen der Route."""
        #add header
        header = section.header
        header_para = header.paragraphs[0]
        
        text_run = header_para.add_run()
        
        text_run.text = 'FO-Document: ' + route_name + '\t'
        
        logo_run = header_para.add_run()
        logo_run.add_picture("beagle.png", width=Inches(1.25))

        #add horizontal line to header
        insertHR(header_para)
    
    #generate page numbers

    def create_element(name):
        return OxmlElement(name)

    def create_attribute(element, name, value):
        element.set(ns.qn(name), value)


    def add_page_number(run):
        run.text = 'Page '
        fldChar1 = create_element('w:fldChar')
        create_attribute(fldChar1, 'w:fldCharType', 'begin')

        instrText = create_element('w:instrText')
        create_attribute(instrText, 'xml:space', 'preserve')
        instrText.text = "PAGE"

        fldChar2 = create_element('w:fldChar')
        create_attribute(fldChar2, 'w:fldCharType', 'end')

        run._r.append(fldChar1)
        run._r.append(instrText)
        run._r.append(fldChar2)

    def insertHR(paragraph):
        p = paragraph._p
        pPr = p.get_or_add_pPr()
        pBdr = OxmlElement('w:pBdr')
        pPr.insert_element_before(pBdr,
            'w:shd', 'w:tabs', 'w:suppressAutoHyphens', 'w:kinsoku', 'w:wordWrap',
            'w:overflowPunct', 'w:topLinePunct', 'w:autoSpaceDE', 'w:autoSpaceDN',
            'w:bidi', 'w:adjustRightInd', 'w:snapToGrid', 'w:spacing', 'w:ind',
            'w:contextualSpacing', 'w:mirrorIndents', 'w:suppressOverlap', 'w:jc',
            'w:textDirection', 'w:textAlignment', 'w:textboxTightWrap',
            'w:outlineLvl', 'w:divId', 'w:cnfStyle', 'w:rPr', 'w:sectPr',
            'w:pPrChange'
        )
        bottom = OxmlElement('w:bottom')
        bottom.set(qn('w:val'), 'single')
        bottom.set(qn('w:sz'), '6')
        bottom.set(qn('w:space'), '1')
        bottom.set(qn('w:color'), 'auto')
        pBdr.append(bottom)

    def generate_distance_table(bundesland, route_length):
        df = pd.read_csv("3DFaktor.csv")  # CSV-Datei laden
        #bundesland = get_bundesland_from_geo_data(geo_data)  # Bundesland bestimmen

        D_Faktor = df[df.iloc[:, 1] == bundesland].iloc[0, 2] if bundesland in df.iloc[:, 1].values else "Nicht gefunden"

        # Berechnungen für Tabelle
        line_length_km = route_length # Beispielwert, muss berechnet werden
        equivalent_3d_distance = round(line_length_km * D_Faktor, 2)
        flight_time_mk1 = round(equivalent_3d_distance*1000 / 24 /60,0)
        flight_time_mk2 = round(equivalent_3d_distance*1000 / 28/60,0)
        flight_time_octo = round(equivalent_3d_distance*1000 / 3/60,0)
        
        table_data = [["From take-off/landing site and back", 
                   f"{line_length_km:.2f} km", 
                   f"{equivalent_3d_distance:.2f} km", 
                   f"\n"
                   f"{int(flight_time_octo)} min\n"
                   f"\n"]]
        col_widths = [60, 35, 35, 45]
        
        
    def footer():

        #add line
        footer = section.footer
        footer_line_para = footer.paragraphs[0]
        insertHR(footer_line_para)

        #add page number
        footer_para = footer.add_paragraph()
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_page_number(footer_para.add_run())
    # docx laden

    document = Document()
    section = document.sections[0]

    # Set the default font for all paragraphs
    style = document.styles['Normal']
    font = style.font
    font.name = 'Montserrat'           
    font.size = Pt(12)            
    font.color.rgb = RGBColor(0,0,0)
    


    # Set the default font for all paragraphs
    style = document.styles['Heading 1']
    font = style.font
    font.name = 'Montserrat'           
    font.size = Pt(14)            
    font.color.rgb = RGBColor(0,0,0)
    font.bold = True

    # demo variables
    
    route_name = "Unknown Route"
    suffix = "1"
    issue = "00"
    image_path_overview = "flight_route.png"
    bundesland = "Berlin"
    route_length = 1000
    
    header(route_name)

    #generate intro paragraph

    intro_para = document.add_paragraph()

    intro_para_text = f"Appendix {suffix} to the Flight Operation Document {route_name[:10]} \n"+ f"{datetime.date.today().strftime('%d.%m.%Y')}\n"+f"ISSUE {issue}\n"+f"Flight Route {suffix}"

    intro_para.add_run(intro_para_text)

    #generate route overview

    document.add_heading('1. Route Overview', 1)

    route_overview_para = document.add_paragraph()
    route_overview_para.add_run(f"Figure A{suffix}.1 gives a general overview of the mission.\n")
    image1_run = route_overview_para.add_run()
    image1_run.add_picture(image_path_overview, width=Inches(19.5))
    

    #Flight distances and Times

    document.add_heading('2. Flight Distances and Times', 1)

    flight_distances_para = document.add_paragraph()

    #todo table fixen

    table = generate_distance_table(bundesland, route_length)
    table_para = document.add_table(rows=1, cols=4)
    table.style = 'Table Grid'

    footer()

    
    document.save(output_path)

output_docx_path = "flight_report.docx"

generate_docx_report(output_docx_path)