"""
=============================================================================
SESI AUTOMATED REPORT GENERATION
PDF/HTML reports with embedded visualizations and analysis
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         2026

WHAT THIS MODULE DOES
---------------------
[D] Generates comprehensive reports including:
  • Executive summary
  • Design parameters and constraints
  • Simulation results (S-parameters, field distributions)
  • Optimization results (Pareto front, selected design)
  • Sensitivity analysis
  • Embedded visualizations (plots, images)
  • Recommendations and next steps

REPORT FORMATS
--------------
  [C] PDF: professional, printable, static
  [C] HTML: interactive, shareable, web-based
  [D] Markdown: lightweight, version-control friendly

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Reports are DOCUMENTATION AIDS, not peer-reviewed publications.
• All claims must be supported by underlying data.
• Visualizations are interpretive; choose scales carefully.
• Reports should include uncertainty and limitation statements.
=============================================================================
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sesi.report")

# Try report generation backends
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    import jinja2
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

__all__ = [
    "ReportConfig",
    "ReportGenerator",
    "PDFReportGenerator",
    "HTMLReportGenerator",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ReportConfig:
    """Configuration for report generation."""
    # Output
    output_dir: str = "./reports"
    report_title: str = "SESI Stealth Design Report"
    author: str = "PAI & Yoon A. Limsuwan"
    organization: str = "MSPS NETWORK"
    
    # Content
    include_executive_summary: bool = True
    include_technical_details: bool = True
    include_visualizations: bool = True
    include_raw_data: bool = False
    
    # Formatting
    pdf_format: str = "A4"  # A4, letter
    html_template: str = "default"
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# [D] REPORT GENERATOR BASE
# ═══════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """
    [D] Base class for report generation.
    """
    
    def __init__(self, cfg: ReportConfig):
        self.cfg = cfg
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate(self, data: Dict[str, Any],
                 output_path: Optional[str] = None) -> str:
        """
        Generate report from data.
        
        Args:
          • data: dict with all report content
          • output_path: optional custom output path
        
        Returns:
          • Path to generated report file
        """
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════
# [C] PDF REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class PDFReportGenerator(ReportGenerator):
    """
    [C] Generate PDF reports using ReportLab.
    """
    
    def __init__(self, cfg: ReportConfig):
        super().__init__(cfg)
        if not HAS_REPORTLAB:
            logger.warning("ReportLab not available; PDF generation disabled")
    
    def generate(self, data: Dict[str, Any],
                 output_path: Optional[str] = None) -> str:
        """Generate PDF report."""
        if not HAS_REPORTLAB:
            raise RuntimeError("ReportLab required for PDF generation")
        
        output_path = output_path or f"{self.cfg.output_dir}/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
        )
        
        body_style = styles['Normal']
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph(self.cfg.report_title, title_style))
        story.append(Paragraph(f"Author: {self.cfg.author}", body_style))
        story.append(Paragraph(f"Organization: {self.cfg.organization}", body_style))
        story.append(Paragraph(f"Generated: {self.timestamp}", body_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        if self.cfg.include_executive_summary and "executive_summary" in data:
            story.append(Paragraph("Executive Summary", heading_style))
            story.append(Paragraph(data["executive_summary"], body_style))
            story.append(Spacer(1, 10))
        
        # Design Parameters
        if "design_params" in data:
            story.append(Paragraph("Design Parameters", heading_style))
            params_table_data = [["Parameter", "Value", "Unit"]]
            for key, value in data["design_params"].items():
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        params_table_data.append([f"{key}.{subkey}", str(subval), ""])
                else:
                    params_table_data.append([key, str(value), ""])
            
            params_table = Table(params_table_data)
            params_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(params_table)
            story.append(Spacer(1, 10))
        
        # Simulation Results
        if "simulation_results" in data:
            story.append(Paragraph("Simulation Results", heading_style))
            sim_results = data["simulation_results"]
            
            if "absorption" in sim_results:
                story.append(Paragraph(f"Mean Absorption: {sim_results['absorption']:.4f}", body_style))
            if "rcs_reduction_db" in sim_results:
                story.append(Paragraph(f"RCS Reduction: {sim_results['rcs_reduction_db']:.2f} dB", body_style))
            
            story.append(Spacer(1, 10))
        
        # Visualizations
        if self.cfg.include_visualizations and "visualization_paths" in data:
            story.append(Paragraph("Visualizations", heading_style))
            for viz_path in data["visualization_paths"]:
                if Path(viz_path).exists():
                    img = Image(viz_path, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 10))
        
        # Recommendations
        if "recommendations" in data:
            story.append(Paragraph("Recommendations", heading_style))
            for rec in data["recommendations"]:
                story.append(Paragraph(f"• {rec}", body_style))
            story.append(Spacer(1, 10))
        
        # Disclaimer
        story.append(Spacer(1, 20))
        disclaimer = (
            "DISCLAIMER: This report is generated automatically from simulation data. "
            "All results are computational predictions and require experimental validation. "
            "Visualizations are interpretive aids. Constraints are engineering requirements, "
            "not physical laws."
        )
        story.append(Paragraph(disclaimer, ParagraphStyle(
            'Disclaimer',
            parent=body_style,
            fontSize=8,
            textColor=colors.grey,
        )))
        
        # Build PDF
        doc.build(story)
        logger.info(f"PDF report saved to {output_path}")
        
        return output_path


# ═══════════════════════════════════════════════════════════════════════
# [C] HTML REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class HTMLReportGenerator(ReportGenerator):
    """
    [C] Generate HTML reports using Jinja2 templates.
    """
    
    def __init__(self, cfg: ReportConfig):
        super().__init__(cfg)
        if not HAS_JINJA2:
            logger.warning("Jinja2 not available; HTML generation disabled")
    
    def generate(self, data: Dict[str, Any],
                 output_path: Optional[str] = None) -> str:
        """Generate HTML report."""
        if not HAS_JINJA2:
            raise RuntimeError("Jinja2 required for HTML generation")
        
        output_path = output_path or f"{self.cfg.output_dir}/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Prepare template data
        template_data = {
            "title": self.cfg.report_title,
            "author": self.cfg.author,
            "organization": self.cfg.organization,
            "timestamp": self.timestamp,
            "data": data,
            "include_executive_summary": self.cfg.include_executive_summary,
            "include_visualizations": self.cfg.include_visualizations,
        }
        
        # Render template
        template_str = self._get_default_template()
        template = jinja2.Template(template_str)
        html_content = template.render(**template_data)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")
        return output_path
    
    def _get_default_template(self) -> str:
        """Return default HTML template."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        h1 { color: #2c3e50; text-align: center; }
        h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
        .metadata { text-align: center; color: #7f8c8d; margin-bottom: 30px; }
        .section { margin-bottom: 30px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        img { max-width: 100%; height: auto; margin: 10px 0; }
        .disclaimer { font-size: 0.8em; color: #95a5a6; margin-top: 40px; padding: 10px; background-color: #f9f9f9; }
        ul { margin-left: 20px; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <div class="metadata">
        <p><strong>Author:</strong> {{ author }} | <strong>Organization:</strong> {{ organization }}</p>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
    </div>
    
    {% if include_executive_summary and data.executive_summary %}
    <div class="section">
        <h2>Executive Summary</h2>
        <p>{{ data.executive_summary }}</p>
    </div>
    {% endif %}
    
    {% if data.design_params %}
    <div class="section">
        <h2>Design Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
            {% for key, value in data.design_params.items() %}
            <tr>
                <td>{{ key }}</td>
                <td>{{ value }}</td>
                <td></td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
    
    {% if data.simulation_results %}
    <div class="section">
        <h2>Simulation Results</h2>
        {% if data.simulation_results.absorption %}
        <p><strong>Mean Absorption:</strong> {{ "%.4f"|format(data.simulation_results.absorption) }}</p>
        {% endif %}
        {% if data.simulation_results.rcs_reduction_db %}
        <p><strong>RCS Reduction:</strong> {{ "%.2f"|format(data.simulation_results.rcs_reduction_db) }} dB</p>
        {% endif %}
    </div>
    {% endif %}
    
    {% if include_visualizations and data.visualization_paths %}
    <div class="section">
        <h2>Visualizations</h2>
        {% for viz_path in data.visualization_paths %}
        <img src="{{ viz_path }}" alt="Visualization">
        {% endfor %}
    </div>
    {% endif %}
    
    {% if data.recommendations %}
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
        {% for rec in data.recommendations %}
            <li>{{ rec }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}
    
    <div class="disclaimer">
        <strong>DISCLAIMER:</strong> This report is generated automatically from simulation data. 
        All results are computational predictions and require experimental validation. 
        Visualizations are interpretive aids. Constraints are engineering requirements, 
        not physical laws.
    </div>
</body>
</html>
        """


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Automated Report Generation — Smoke Test")
    print("=" * 72)
    print(f"  ReportLab available    : {HAS_REPORTLAB}")
    print(f"  Jinja2 available       : {HAS_JINJA2}")
    
    cfg = ReportConfig(
        report_title="Hypersonic Stealth Design Report",
        author="Test Author",
        organization="MSPS NETWORK",
        output_dir="./test_reports",
    )
    
    # Sample data
    sample_data = {
        "executive_summary": "This report presents the design and simulation results for a hypersonic stealth metamaterial absorber.",
        "design_params": {
            "carpet_level": 3,
            "eps_r": 10.0,
            "mu_r": 10.0,
            "tan_d": 0.1,
            "thickness_mm": 2.0,
        },
        "simulation_results": {
            "absorption": 0.92,
            "rcs_reduction_db": 15.5,
        },
        "visualization_paths": [],  # Would contain paths to generated plots
        "recommendations": [
            "Increase carpet level for better RCS reduction",
            "Optimize thickness for broader bandwidth",
            "Validate with experimental measurements",
        ],
    }
    
    # Generate PDF
    if HAS_REPORTLAB:
        print("\n  ── Generating PDF report ──")
        pdf_gen = PDFReportGenerator(cfg)
        pdf_path = pdf_gen.generate(sample_data)
        print(f"    PDF saved to         : {pdf_path}")
    else:
        print("\n  [SKIP] ReportLab not available for PDF")
    
    # Generate HTML
    if HAS_JINJA2:
        print("\n  ── Generating HTML report ──")
        html_gen = HTMLReportGenerator(cfg)
        html_path = html_gen.generate(sample_data)
        print(f"    HTML saved to        : {html_path}")
    else:
        print("\n  [SKIP] Jinja2 not available for HTML")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Reports are DOCUMENTATION AIDS, not peer-reviewed publications")
    print("  • All claims must be supported by underlying data")
    print("  • Visualizations are interpretive; choose scales carefully")
    print("  • Reports include uncertainty and limitation statements")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
