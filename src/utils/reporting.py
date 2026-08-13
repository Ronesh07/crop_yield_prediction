import pandas as pd
import streamlit as st
from io import BytesIO

# For PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def get_csv_download(dataframe, filename="report.csv"):
    csv = dataframe.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )

def get_pdf_download(text, filename="report.pdf"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    lines = text.split('\n')
    y = height - 40
    for line in lines:
        c.drawString(40, y, line)
        y -= 15
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
    buffer.seek(0)
    st.download_button(
        label="Download as PDF",
        data=buffer,
        file_name=filename,
        mime='application/pdf'
    ) 