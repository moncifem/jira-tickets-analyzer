"""
PDF Generation Utilities for Jira Analysis Reports
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import base64
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether, Flowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import plotly.graph_objects as go
from plotly.io import to_image
import pandas as pd


class PlotlyImageFlowable(Flowable):
    """Custom flowable for Plotly charts"""
    
    def __init__(self, plotly_figure_json, width=6*inch, height=4*inch):
        Flowable.__init__(self)
        self.plotly_json = plotly_figure_json
        self.width = width
        self.height = height
        self._image = None
    
    def draw(self):
        """Draw the Plotly figure"""
        if self._image is None:
            # Convert Plotly JSON to figure
            fig = go.Figure(json.loads(self.plotly_json))
            
            # Convert to image bytes
            img_bytes = fig.to_image(format="png", width=800, height=600)
            
            # Create image from bytes
            img_buffer = BytesIO(img_bytes)
            self._image = Image(img_buffer, width=self.width, height=self.height)
        
        # Draw the image
        self._image.drawOn(self.canv, 0, 0)


class JiraAnalysisPDFGenerator:
    """Generate professional PDF reports for Jira analysis"""
    
    def __init__(self, filename: str = "jira_analysis_report.pdf"):
        self.filename = filename
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Heading 1
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#2ca02c'),
            spaceBefore=20,
            spaceAfter=15
        ))
        
        # Heading 2
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#ff7f0e'),
            spaceBefore=15,
            spaceAfter=10
        ))
        
        # Insight style
        self.styles.add(ParagraphStyle(
            name='Insight',
            parent=self.styles['Normal'],
            fontSize=11,
            leftIndent=20,
            rightIndent=20,
            spaceBefore=10,
            spaceAfter=10,
            borderColor=colors.HexColor('#d62728'),
            borderWidth=1,
            borderPadding=10,
            backColor=colors.HexColor('#f0f0f0')
        ))
        
        # Metric style
        self.styles.add(ParagraphStyle(
            name='Metric',
            parent=self.styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#9467bd')
        ))
    
    def generate_report(self, analysis_results: Dict[str, Any]):
        """Generate a comprehensive PDF report from analysis results"""
        doc = SimpleDocTemplate(
            self.filename,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build the story
        story = []
        
        # Add title page
        story.extend(self._create_title_page(analysis_results))
        story.append(PageBreak())
        
        # Add executive summary
        story.extend(self._create_executive_summary(analysis_results))
        story.append(PageBreak())
        
        # Add detailed analysis sections
        if 'issue_types_analysis' in analysis_results:
            story.extend(self._create_issue_types_section(analysis_results['issue_types_analysis']))
            story.append(PageBreak())
        
        if 'applications_analysis' in analysis_results:
            story.extend(self._create_applications_section(analysis_results['applications_analysis']))
            story.append(PageBreak())
        
        if 'resolution_times_analysis' in analysis_results:
            story.extend(self._create_resolution_times_section(analysis_results['resolution_times_analysis']))
            story.append(PageBreak())
        
        if 'team_performance_analysis' in analysis_results:
            story.extend(self._create_team_performance_section(analysis_results['team_performance_analysis']))
            story.append(PageBreak())
        
        if 'trends_analysis' in analysis_results:
            story.extend(self._create_trends_section(analysis_results['trends_analysis']))
            story.append(PageBreak())
        
        if 'bottlenecks_analysis' in analysis_results:
            story.extend(self._create_bottlenecks_section(analysis_results['bottlenecks_analysis']))
            story.append(PageBreak())
        
        # Add insights summary
        if 'insights' in analysis_results:
            story.extend(self._create_insights_summary(analysis_results['insights']))
        
        # Build PDF
        doc.build(story)
        print(f"✅ PDF report generated: {self.filename}")
    
    def _create_title_page(self, results: Dict[str, Any]) -> List:
        """Create the title page"""
        elements = []
        
        # Title
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph("Jira Support Ticket Analysis Report", self.styles['CustomTitle']))
        
        # Subtitle with date
        elements.append(Spacer(1, 0.5*inch))
        date_str = datetime.now().strftime("%B %d, %Y")
        elements.append(Paragraph(f"Generated on {date_str}", self.styles['Normal']))
        
        # Key metrics summary
        elements.append(Spacer(1, inch))
        
        metrics_data = []
        if 'total_tickets' in results:
            metrics_data.append(['Total Tickets Analyzed:', f"{results['total_tickets']:,}"])
        if 'date_range' in results:
            metrics_data.append(['Date Range:', results['date_range']])
        if 'total_applications' in results:
            metrics_data.append(['Applications:', str(results['total_applications'])])
        if 'team_size' in results:
            metrics_data.append(['Team Members:', str(results['team_size'])])
        
        if metrics_data:
            metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            elements.append(metrics_table)
        
        return elements
    
    def _create_executive_summary(self, results: Dict[str, Any]) -> List:
        """Create executive summary section"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['CustomHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Key findings
        summary_text = """
        This report provides a comprehensive analysis of the Jira support tickets, 
        identifying key patterns, bottlenecks, and opportunities for improvement in 
        the support process.
        """
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Top insights
        if 'insights' in results and results['insights']:
            elements.append(Paragraph("Key Insights:", self.styles['CustomHeading2']))
            for insight in results['insights'][:5]:  # Top 5 insights
                elements.append(Paragraph(f"• {insight}", self.styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_issue_types_section(self, analysis: Dict[str, Any]) -> List:
        """Create issue types analysis section"""
        elements = []
        
        elements.append(Paragraph("Issue Types Analysis", self.styles['CustomHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Summary text
        total_issues = analysis.get('total_issues', 0)
        unique_types = analysis.get('unique_issue_types', 0)
        
        summary = f"""
        Total tickets analyzed: {total_issues:,}<br/>
        Unique issue types identified: {unique_types}
        """
        elements.append(Paragraph(summary, self.styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Top issue types table
        if 'top_issue_types' in analysis:
            elements.append(Paragraph("Top Issue Types", self.styles['CustomHeading2']))
            
            table_data = [['Issue Type', 'Count', 'Percentage']]
            for issue in analysis['top_issue_types'][:10]:
                table_data.append([
                    issue['issue_type'],
                    f"{issue['count']:,}",
                    f"{issue['percentage']}%"
                ])
            
            table = Table(table_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
        
        # Add visualization if available
        if 'visualization' in analysis:
            elements.append(Spacer(1, 0.3*inch))
            elements.append(PlotlyImageFlowable(analysis['visualization']))
        
        return elements
    
    def _create_applications_section(self, analysis: Dict[str, Any]) -> List:
        """Create applications analysis section"""
        elements = []
        
        elements.append(Paragraph("Applications Analysis", self.styles['CustomHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Summary
        total_apps = analysis.get('total_applications', 0)
        elements.append(Paragraph(f"Total applications with issues: {total_apps}", self.styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Applications table
        if 'application_analysis' in analysis:
            elements.append(Paragraph("Applications by Issue Count", self.styles['CustomHeading2']))
            
            for app_data in analysis['application_analysis'][:5]:
                # Application name and total
                app_name = app_data['application']
                total_issues = app_data['total_issues']
                
                elements.append(Paragraph(
                    f"<b>{app_name}</b> - {total_issues:,} issues",
                    self.styles['Normal']
                ))
                
                # Top issues for this app
                if 'top_issues' in app_data:
                    issue_list = []
                    for issue in app_data['top_issues'][:3]:
                        issue_list.append(f"{issue['issue_type']} ({issue['count']})")
                    
                    elements.append(Paragraph(
                        f"Top issues: {', '.join(issue_list)}",
                        self.styles['Normal']
                    ))
                
                elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_resolution_times_section(self, analysis: Dict[str, Any]) -> List:
        """Create resolution times analysis section"""
        elements = []
        
        elements.append(Paragraph("Resolution Times Analysis", self.styles['CustomHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Overall statistics
        if 'overall_stats' in analysis and analysis['overall_stats']:
            stats = analysis['overall_stats']
            elements.append(Paragraph("Overall Resolution Statistics", self.styles['CustomHeading2']))
            
            stats_data = [
                ['Metric', 'Value'],
                ['Average Resolution Time', f"{stats.get('mean_hours', 0):.1f} hours"],
                ['Median Resolution Time', f"{stats.get('median_hours', 0):.1f} hours"],
                ['Standard Deviation', f"{stats.get('std_hours', 0):.1f} hours"],
                ['Minimum Time', f"{stats.get('min_hours', 0):.1f} hours"],
                ['Maximum Time', f"{stats.get('max_hours', 0):.1f} hours"],
                ['Total Resolved Tickets', f"{stats.get('count', 0):,}"]
            ]
            
            table = Table(stats_data, colWidths=[3*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.3*inch))
        
        # By priority
        if 'by_priority' in analysis and analysis['by_priority']:
            elements.append(Paragraph("Resolution Times by Priority", self.styles['CustomHeading2']))
            
            priority_data = [['Priority', 'Avg Hours', 'Median Hours', 'Count']]
            for priority, stats in analysis['by_priority'].items():
                priority_data.append([
                    priority,
                    f"{stats.get('mean_hours', 0):.1f}",
                    f"{stats.get('median_hours', 0):.1f}",
                    f"{stats.get('count', 0):,}"
                ])
            
            table = Table(priority_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
        
        return elements
    
    def _create_team_performance_section(self, analysis: Dict[str, Any]) -> List:
        """Create team performance analysis section"""
        elements = []
        
        elements.append(Paragraph("Team Performance Analysis", self.styles['CustomHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Team size
        team_size = analysis.get('team_size', 0)
        elements.append(Paragraph(f"Active team members: {team_size}", self.styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Top performers table
        if 'top_performers' in analysis:
            elements.append(Paragraph("Team Performance Metrics", self.styles['CustomHeading2']))
            
            table_data = [['Team Member', 'Total Tickets', 'Resolved', 'Resolution Rate', 'Avg Resolution (hrs)']]
            
            for performer in analysis['top_performers'][:10]:
                table_data.append([
                    performer['assignee'][:30],  # Truncate long names
                    f"{performer['total_tickets']:,}",
                    f"{performer['resolved_tickets']:,}",
                    f"{performer['resolution_rate']}%",
                    f"{performer.get('avg_resolution_hours', 'N/A')}"
                ])
            
            table = Table(table_data, colWidths=[2.5*inch, 1*inch, 1*inch, 1.2*inch, 1.3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
        
        return elements
    
    def _create_trends_section(self, analysis: Dict[str, Any]) -> List:
        """Create trends analysis section"""
        elements = []
        
        elements.append(Paragraph("Trends Over Time", self.styles['CustomHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Trend summary
        if 'trend_data' in analysis:
            periods = len(analysis['trend_data'])
            elements.append(Paragraph(f"Analysis period: Last {periods} months", self.styles['Normal']))
            elements.append(Spacer(1, 0.3*inch))
            
            # Recent trends
            if analysis['trend_data']:
                recent = analysis['trend_data'][-3:]  # Last 3 months
                elements.append(Paragraph("Recent Monthly Trends", self.styles['CustomHeading2']))
                
                for month_data in recent:
                    period = month_data['period']
                    total = month_data['total_tickets']
                    
                    elements.append(Paragraph(
                        f"<b>{period}</b>: {total:,} tickets",
                        self.styles['Normal']
                    ))
                    
                    # Top issues for the month
                    if 'top_issues' in month_data:
                        issues = [f"{i['type']} ({i['count']})" for i in month_data['top_issues'][:2]]
                        elements.append(Paragraph(
                            f"Top issues: {', '.join(issues)}",
                            self.styles['Normal']
                        ))
                    
                    elements.append(Spacer(1, 0.15*inch))
        
        # Add trend visualization if available
        if 'visualization' in analysis:
            elements.append(PlotlyImageFlowable(analysis['visualization']))
        
        return elements
    
    def _create_bottlenecks_section(self, analysis: Dict[str, Any]) -> List:
        """Create bottlenecks analysis section"""
        elements = []
        
        elements.append(Paragraph("Process Bottlenecks", self.styles['CustomHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Stale tickets
        if 'stale_tickets' in analysis:
            stale = analysis['stale_tickets']
            total_stale = stale.get('total', 0)
            threshold = stale.get('threshold_days', 30)
            
            elements.append(Paragraph(
                f"<b>Stale Tickets:</b> {total_stale:,} tickets not updated in {threshold} days",
                self.styles['Normal']
            ))
            elements.append(Spacer(1, 0.2*inch))
        
        # Overloaded assignees
        if 'overloaded_assignees' in analysis:
            elements.append(Paragraph("Workload Distribution Issues", self.styles['CustomHeading2']))
            
            table_data = [['Assignee', 'Open Tickets']]
            for assignee in analysis['overloaded_assignees'][:5]:
                table_data.append([
                    assignee['assignee'],
                    f"{assignee['open_tickets']:,}"
                ])
            
            table = Table(table_data, colWidths=[3*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.3*inch))
        
        # Status bottlenecks
        if 'status_bottlenecks' in analysis:
            elements.append(Paragraph("Status Bottlenecks", self.styles['CustomHeading2']))
            
            for status in analysis['status_bottlenecks'][:3]:
                elements.append(Paragraph(
                    f"• {status['status']}: {status['count']:,} tickets",
                    self.styles['Normal']
                ))
        
        return elements
    
    def _create_insights_summary(self, insights: List[str]) -> List:
        """Create insights summary section"""
        elements = []
        
        elements.append(Paragraph("Key Insights and Recommendations", self.styles['CustomHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        for i, insight in enumerate(insights, 1):
            # Use the Insight style for each insight
            elements.append(Paragraph(f"{i}. {insight}", self.styles['Insight']))
            elements.append(Spacer(1, 0.1*inch))
        
        return elements 