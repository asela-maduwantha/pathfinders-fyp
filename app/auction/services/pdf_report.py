"""
PDF valuation report generator for the Auction Bid Price module, ported from
Module 4's Flask app.py (fpdf2-based), adapted to return bytes for a FastAPI
StreamingResponse instead of Flask's send_file.
"""
from __future__ import annotations

from typing import Any

from fpdf import FPDF


class TimberReportPDF(FPDF):
    def header(self) -> None:
        self.set_fill_color(27, 67, 50)
        self.rect(0, 0, 210, 45, 'F')

        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 22)
        self.cell(0, 15, 'ADVANCED TIMBER VALUATION REPORT', border=False, ln=True, align='C')

        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 5, 'Powered by Conformal Uncertainty Models & XGBoost', border=False, ln=True, align='C')
        self.ln(25)

    def footer(self) -> None:
        self.set_y(-20)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'Timber Value Prediction System - Advanced Valuation Report', 0, 0, 'L')
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'R')


def build_report_pdf(row: dict[str, Any]) -> bytes:
    pdf = TimberReportPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()

    dark_gray = (50, 50, 50)
    light_gray = (240, 240, 240)
    green_text = (27, 67, 50)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(*dark_gray)
    pdf.cell(50, 6, f"Transaction ID: T-{row['id']:06d}", ln=False)
    pdf.cell(0, 6, f"Valuation Date: {row['timestamp']}", ln=True, align="R")
    pdf.ln(5)

    pdf.set_fill_color(*light_gray)
    pdf.rect(10, 60, 190, 30, 'F')
    pdf.set_xy(10, 62)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(*green_text)
    pdf.cell(0, 5, "ESTIMATED VALUATION METRICS SUMMARY:", ln=True, align="C")

    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 8, f"Rs. {row['predicted_value']:,.2f}", ln=True, align="C")

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 5, f"90% Uncertainty Interval: Rs. {row['low_bound']:,.2f} - Rs. {row['high_bound']:,.2f}", ln=True, align="C")
    pdf.ln(12)

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(*dark_gray)
    pdf.multi_cell(
        0, 5,
        f"Based on the provided characteristics, the estimated timber value is Rs. {row['predicted_value']:,.2f}. "
        "This is an automated estimation compiled via an XGBoost Machine Learning model. Quality gradings, "
        "dimensional attributes, and local market competition levels were evaluated.",
        align="J",
    )
    pdf.ln(5)

    def draw_table_header(title: str) -> None:
        pdf.set_fill_color(45, 106, 79)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(190, 8, title, border=1, ln=True, fill=True)
        pdf.set_text_color(*dark_gray)
        pdf.set_font('Helvetica', '', 9)

    def draw_table_row(label: str, val: Any) -> None:
        pdf.set_fill_color(250, 250, 250)
        pdf.cell(95, 7, f" {label}", border=1, fill=True)
        pdf.cell(95, 7, f" {val}", border=1, ln=True)

    draw_table_header("1. Sourcing & Dimensional Properties")
    draw_table_row("Timber Species", row['species'])
    draw_table_row("Source Region / Season", f"{row['region']} / {row['season']} season")
    draw_table_row("Log Dimensions", f"Diameter: {row['diameter_cm']} cm | Length: {row['length_m']} m")
    draw_table_row("Calculated Volume", f"{row['volume_m3']} m³")
    draw_table_row("Density / Moisture", f"{row['density_kg_m3']} kg/m³ / {row['moisture_content']}%")
    pdf.ln(5)

    draw_table_header("2. Quality & Defect Metrics")
    draw_table_row("Overall Quality Grade", f"Grade {row['quality_grade']}/5")
    draw_table_row("Straightness / Taper Score", f"Straightness: {row['straightness_score']}/10 | Taper: {row['taper_score']}/10")
    draw_table_row("Visible Defects Score", f"{row['visible_defects_score']}/10")
    draw_table_row("Internal Defect Risk Index", f"{row['internal_defect_risk'] * 100:.1f}%")
    pdf.ln(5)

    draw_table_header("3. Advanced Bidding & Market Estimations")
    draw_table_row("Recommended Starting Bid", f"Rs. {row['starting_bid']:,.2f}")
    draw_table_row("Expected Final Auction Price", f"Rs. {row['expected_final_price']:,.2f}")
    draw_table_row("Estimated Probability of Sale", f"{row['sale_probability'] * 100:.1f}%")
    draw_table_row("Base Market Species Price", f"Rs. {row['avg_market_price_species']:.2f}/m³ (Volatility: {row['price_volatility'] * 100:.1f}%)")
    draw_table_row("Auction Configuration", f"{row['auction_type']} auction ({row['competition_level']} level / {row['num_expected_bidders']} bidders)")
    pdf.ln(8)

    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(
        0, 4,
        "Disclaimer: This document contains an automated commercial prediction derived using Scikit-Learn "
        "pipelines and XGBoost. It does not constitute a legal valuation and should be used solely as bid "
        "assistance guidelines in public and private auctions.",
    )

    return bytes(pdf.output())
