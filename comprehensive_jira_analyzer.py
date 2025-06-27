#!/usr/bin/env python3
"""
Comprehensive Jira Analysis Tool
Analyzes CSV files containing Jira ticket data and generates detailed statistical reports in PDF format.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class JiraAnalyzer:
    """Comprehensive Jira ticket analyzer with PDF report generation"""
    
    def __init__(self, csv_path: str):
        """Initialize analyzer with CSV file path"""
        self.csv_path = csv_path
        self.df = None
        self.stats = {}
        self.charts = []
        
    def load_data(self):
        """Load and prepare the data"""
        print(f"📊 Loading data from {self.csv_path}...")
        
        # Load data in chunks for large files
        chunk_size = 50000
        chunks = []
        
        for chunk in pd.read_csv(self.csv_path, chunksize=chunk_size):
            chunks.append(chunk)
        
        self.df = pd.concat(chunks, ignore_index=True)
        
        # Clean and prepare data
        self._prepare_data()
        
        print(f"✅ Loaded {len(self.df):,} tickets with {len(self.df.columns)} columns")
        
    def _prepare_data(self):
        """Clean and prepare the data for analysis"""
        # Convert date columns
        date_columns = ['Created', 'Updated', 'Resolved']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Calculate resolution times
        if 'Created' in self.df.columns and 'Resolved' in self.df.columns:
            resolved_mask = self.df['Resolved'].notna()
            self.df.loc[resolved_mask, 'Resolution_Hours'] = (
                self.df.loc[resolved_mask, 'Resolved'] - 
                self.df.loc[resolved_mask, 'Created']
            ).dt.total_seconds() / 3600
            
            self.df.loc[resolved_mask, 'Resolution_Days'] = (
                self.df.loc[resolved_mask, 'Resolution_Hours'] / 24
            )
        
        # Extract time features
        if 'Created' in self.df.columns:
            self.df['Created_Year'] = self.df['Created'].dt.year
            self.df['Created_Month'] = self.df['Created'].dt.month
            self.df['Created_Day_of_Week'] = self.df['Created'].dt.day_name()
            self.df['Created_Hour'] = self.df['Created'].dt.hour
    
    def analyze_overview(self):
        """Generate overview statistics"""
        print("🔍 Analyzing overview statistics...")
        
        overview = {
            'total_tickets': len(self.df),
            'date_range': {
                'start': self.df['Created'].min() if 'Created' in self.df.columns else None,
                'end': self.df['Created'].max() if 'Created' in self.df.columns else None
            },
            'columns': list(self.df.columns),
            'file_size_mb': os.path.getsize(self.csv_path) / (1024 * 1024)
        }
        
        # Resolution statistics
        if 'Resolved' in self.df.columns:
            resolved_tickets = self.df['Resolved'].notna().sum()
            overview['resolution_stats'] = {
                'total_resolved': resolved_tickets,
                'resolution_rate': (resolved_tickets / len(self.df)) * 100,
                'pending_tickets': len(self.df) - resolved_tickets
            }
        
        self.stats['overview'] = overview
    
    def analyze_issue_types(self):
        """Analyze issue type distribution"""
        print("🏷️ Analyzing issue types...")
        
        if 'Issue Type' not in self.df.columns:
            return
        
        issue_counts = self.df['Issue Type'].value_counts()
        issue_percentages = (issue_counts / len(self.df)) * 100
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(issue_counts.index, issue_counts.values)
        ax.set_title('Distribution of Issue Types', fontsize=16, fontweight='bold')
        ax.set_xlabel('Issue Type', fontsize=12)
        ax.set_ylabel('Number of Tickets', fontsize=12)
        
        # Add value labels on bars
        for bar, count in zip(bars, issue_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(issue_counts.values),
                   f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        chart_path = self._save_chart('issue_types_distribution')
        self.charts.append(chart_path)
        
        self.stats['issue_types'] = {
            'distribution': issue_counts.to_dict(),
            'percentages': issue_percentages.to_dict(),
            'most_common': issue_counts.index[0],
            'total_types': len(issue_counts)
        }
    
    def analyze_priorities(self):
        """Analyze priority distribution"""
        print("🎯 Analyzing priorities...")
        
        if 'Priority' not in self.df.columns:
            return
        
        priority_counts = self.df['Priority'].value_counts()
        
        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Pie chart
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(priority_counts)))
        wedges, texts, autotexts = ax1.pie(priority_counts.values, labels=priority_counts.index, 
                                          autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax1.set_title('Priority Distribution', fontsize=16, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(priority_counts.index, priority_counts.values, color=colors_pie)
        ax2.set_title('Priority Counts', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Priority', fontsize=12)
        ax2.set_ylabel('Number of Tickets', fontsize=12)
        
        for bar, count in zip(bars, priority_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(priority_counts.values),
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        chart_path = self._save_chart('priority_distribution')
        self.charts.append(chart_path)
        
        self.stats['priorities'] = {
            'distribution': priority_counts.to_dict(),
            'high_priority_count': priority_counts.get('Critical', 0) + priority_counts.get('High', 0),
            'total_priorities': len(priority_counts)
        }
    
    def analyze_status(self):
        """Analyze status distribution"""
        print("📊 Analyzing status distribution...")
        
        if 'Status' not in self.df.columns:
            return
        
        status_counts = self.df['Status'].value_counts()
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(status_counts.index, status_counts.values)
        ax.set_title('Status Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Number of Tickets', fontsize=12)
        ax.set_ylabel('Status', fontsize=12)
        
        # Add value labels
        for bar, count in zip(bars, status_counts.values):
            ax.text(bar.get_width() + 0.01*max(status_counts.values), bar.get_y() + bar.get_height()/2,
                   f'{count:,}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        chart_path = self._save_chart('status_distribution')
        self.charts.append(chart_path)
        
        self.stats['status'] = {
            'distribution': status_counts.to_dict(),
            'total_statuses': len(status_counts)
        }
    
    def analyze_applications(self):
        """Analyze application distribution"""
        print("📱 Analyzing applications...")
        
        if 'Application' not in self.df.columns:
            return
        
        app_counts = self.df['Application'].value_counts()
        
        # Create chart for top applications
        top_apps = app_counts.head(10)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.bar(range(len(top_apps)), top_apps.values)
        ax.set_title('Top 10 Applications by Ticket Count', fontsize=16, fontweight='bold')
        ax.set_xlabel('Application', fontsize=12)
        ax.set_ylabel('Number of Tickets', fontsize=12)
        ax.set_xticks(range(len(top_apps)))
        ax.set_xticklabels(top_apps.index, rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, top_apps.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(top_apps.values),
                   f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        chart_path = self._save_chart('application_distribution')
        self.charts.append(chart_path)
        
        self.stats['applications'] = {
            'distribution': app_counts.to_dict(),
            'top_10': top_apps.to_dict(),
            'total_applications': len(app_counts)
        }
    
    def analyze_resolution_times(self):
        """Analyze resolution times"""
        print("⏱️ Analyzing resolution times...")
        
        if 'Resolution_Hours' not in self.df.columns:
            return
        
        resolved_df = self.df[self.df['Resolution_Hours'].notna()]
        
        if len(resolved_df) == 0:
            return
        
        # Calculate statistics
        resolution_stats = {
            'mean_hours': resolved_df['Resolution_Hours'].mean(),
            'median_hours': resolved_df['Resolution_Hours'].median(),
            'std_hours': resolved_df['Resolution_Hours'].std(),
            'min_hours': resolved_df['Resolution_Hours'].min(),
            'max_hours': resolved_df['Resolution_Hours'].max(),
            'percentiles': {
                '25th': resolved_df['Resolution_Hours'].quantile(0.25),
                '75th': resolved_df['Resolution_Hours'].quantile(0.75),
                '90th': resolved_df['Resolution_Hours'].quantile(0.90),
                '95th': resolved_df['Resolution_Hours'].quantile(0.95)
            }
        }
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Histogram
        ax1.hist(resolved_df['Resolution_Hours'], bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title('Distribution of Resolution Times', fontweight='bold')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Frequency')
        ax1.axvline(resolution_stats['mean_hours'], color='red', linestyle='--', label=f'Mean: {resolution_stats["mean_hours"]:.1f}h')
        ax1.axvline(resolution_stats['median_hours'], color='green', linestyle='--', label=f'Median: {resolution_stats["median_hours"]:.1f}h')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(resolved_df['Resolution_Hours'])
        ax2.set_title('Resolution Time Box Plot', fontweight='bold')
        ax2.set_ylabel('Hours')
        
        # Resolution by Priority
        if 'Priority' in resolved_df.columns:
            priority_resolution = resolved_df.groupby('Priority')['Resolution_Hours'].mean().sort_values()
            bars = ax3.bar(priority_resolution.index, priority_resolution.values)
            ax3.set_title('Average Resolution Time by Priority', fontweight='bold')
            ax3.set_xlabel('Priority')
            ax3.set_ylabel('Average Hours')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, time in zip(bars, priority_resolution.values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(priority_resolution.values),
                        f'{time:.1f}h', ha='center', va='bottom', fontweight='bold')
        
        # Resolution by Application (top 10)
        if 'Application' in resolved_df.columns:
            app_resolution = resolved_df.groupby('Application')['Resolution_Hours'].mean().sort_values(ascending=False).head(10)
            bars = ax4.barh(range(len(app_resolution)), app_resolution.values)
            ax4.set_title('Top 10 Apps by Avg Resolution Time', fontweight='bold')
            ax4.set_xlabel('Average Hours')
            ax4.set_yticks(range(len(app_resolution)))
            ax4.set_yticklabels(app_resolution.index)
            
            for bar, time in zip(bars, app_resolution.values):
                ax4.text(bar.get_width() + 0.01*max(app_resolution.values), bar.get_y() + bar.get_height()/2,
                        f'{time:.1f}h', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        chart_path = self._save_chart('resolution_times_analysis')
        self.charts.append(chart_path)
        
        self.stats['resolution_times'] = resolution_stats
    
    def analyze_team_performance(self):
        """Analyze team performance"""
        print("👥 Analyzing team performance...")
        
        if 'Assignee' not in self.df.columns:
            return
        
        # Remove unassigned tickets for team analysis
        team_df = self.df[self.df['Assignee'] != 'unassigned'].copy()
        
        assignee_stats = team_df.groupby('Assignee').agg({
            'Issue Key': 'count',
            'Resolution_Hours': ['mean', 'count']
        }).round(2)
        
        assignee_stats.columns = ['Total_Tickets', 'Avg_Resolution_Hours', 'Resolved_Count']
        assignee_stats['Resolution_Rate'] = (assignee_stats['Resolved_Count'] / assignee_stats['Total_Tickets'] * 100).round(2)
        
        # Get top performers
        top_assignees = assignee_stats.sort_values('Total_Tickets', ascending=False).head(10)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Tickets per assignee
        bars1 = ax1.bar(range(len(top_assignees)), top_assignees['Total_Tickets'])
        ax1.set_title('Top 10 Assignees by Ticket Count', fontweight='bold')
        ax1.set_xlabel('Assignee')
        ax1.set_ylabel('Total Tickets')
        ax1.set_xticks(range(len(top_assignees)))
        ax1.set_xticklabels(top_assignees.index, rotation=45, ha='right')
        
        for bar, count in zip(bars1, top_assignees['Total_Tickets']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(top_assignees['Total_Tickets']),
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Resolution rates
        resolution_rates = top_assignees['Resolution_Rate'].dropna()
        bars2 = ax2.bar(range(len(resolution_rates)), resolution_rates.values)
        ax2.set_title('Resolution Rates (%)', fontweight='bold')
        ax2.set_xlabel('Assignee')
        ax2.set_ylabel('Resolution Rate (%)')
        ax2.set_xticks(range(len(resolution_rates)))
        ax2.set_xticklabels(resolution_rates.index, rotation=45, ha='right')
        ax2.set_ylim(0, 100)
        
        for bar, rate in zip(bars2, resolution_rates.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Average resolution times
        avg_times = top_assignees['Avg_Resolution_Hours'].dropna()
        bars3 = ax3.bar(range(len(avg_times)), avg_times.values)
        ax3.set_title('Average Resolution Times (Hours)', fontweight='bold')
        ax3.set_xlabel('Assignee')
        ax3.set_ylabel('Average Hours')
        ax3.set_xticks(range(len(avg_times)))
        ax3.set_xticklabels(avg_times.index, rotation=45, ha='right')
        
        for bar, time in zip(bars3, avg_times.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(avg_times.values),
                    f'{time:.1f}h', ha='center', va='bottom', fontweight='bold')
        
        # Workload distribution pie chart
        workload = team_df['Assignee'].value_counts().head(8)
        ax4.pie(workload.values, labels=workload.index, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Workload Distribution (Top 8)', fontweight='bold')
        
        plt.tight_layout()
        
        chart_path = self._save_chart('team_performance')
        self.charts.append(chart_path)
        
        self.stats['team_performance'] = {
            'top_performers': top_assignees.to_dict('index'),
            'total_team_members': len(assignee_stats),
            'avg_team_resolution_rate': assignee_stats['Resolution_Rate'].mean()
        }
    
    def analyze_time_trends(self):
        """Analyze trends over time"""
        print("📈 Analyzing time trends...")
        
        if 'Created' not in self.df.columns:
            return
        
        # Monthly trends
        monthly_counts = self.df.groupby(self.df['Created'].dt.to_period('M')).size()
        
        # Daily trends
        daily_trends = self.df.groupby('Created_Day_of_Week').size()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_trends = daily_trends.reindex(day_order, fill_value=0)
        
        # Hourly trends
        hourly_trends = self.df.groupby('Created_Hour').size()
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Monthly trend
        ax1.plot(monthly_counts.index.astype(str), monthly_counts.values, marker='o', linewidth=2, markersize=6)
        ax1.set_title('Monthly Ticket Creation Trend', fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Number of Tickets')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Daily trend
        bars2 = ax2.bar(daily_trends.index, daily_trends.values)
        ax2.set_title('Tickets by Day of Week', fontweight='bold')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Number of Tickets')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars2, daily_trends.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(daily_trends.values),
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Hourly trend
        ax3.plot(hourly_trends.index, hourly_trends.values, marker='o', linewidth=2, markersize=6)
        ax3.set_title('Tickets by Hour of Day', fontweight='bold')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Number of Tickets')
        ax3.set_xticks(range(0, 24, 2))
        ax3.grid(True, alpha=0.3)
        
        # Issue type trends over time
        if 'Issue Type' in self.df.columns and len(monthly_counts) > 1:
            issue_monthly = self.df.groupby([self.df['Created'].dt.to_period('M'), 'Issue Type']).size().unstack(fill_value=0)
            
            for issue_type in issue_monthly.columns[:5]:  # Top 5 issue types
                ax4.plot(issue_monthly.index.astype(str), issue_monthly[issue_type], marker='o', label=issue_type, linewidth=2)
            
            ax4.set_title('Issue Type Trends Over Time', fontweight='bold')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Number of Tickets')
            ax4.legend()
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        chart_path = self._save_chart('time_trends')
        self.charts.append(chart_path)
        
        self.stats['time_trends'] = {
            'monthly_trend': monthly_counts.to_dict(),
            'daily_trend': daily_trends.to_dict(),
            'hourly_trend': hourly_trends.to_dict(),
            'peak_day': daily_trends.idxmax(),
            'peak_hour': hourly_trends.idxmax()
        }
    
    def _save_chart(self, name):
        """Save a chart and return the path"""
        chart_path = f"temp_chart_{name}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        return chart_path
    
    def generate_pdf_report(self, output_path="jira_analysis_report.pdf"):
        """Generate comprehensive PDF report"""
        print(f"📄 Generating PDF report: {output_path}")
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Title page
        story.append(Paragraph("Comprehensive Jira Analysis Report", title_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Overview section
        story.append(Paragraph("Executive Summary", heading_style))
        
        overview = self.stats.get('overview', {})
        overview_data = [
            ['Metric', 'Value'],
            ['Total Tickets', f"{overview.get('total_tickets', 0):,}"],
            ['File Size', f"{overview.get('file_size_mb', 0):.2f} MB"],
            ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M')],
        ]
        
        if 'date_range' in overview and overview['date_range']['start']:
            overview_data.extend([
                ['Date Range Start', overview['date_range']['start'].strftime('%Y-%m-%d')],
                ['Date Range End', overview['date_range']['end'].strftime('%Y-%m-%d')]
            ])
        
        if 'resolution_stats' in overview:
            res_stats = overview['resolution_stats']
            overview_data.extend([
                ['Resolved Tickets', f"{res_stats['total_resolved']:,}"],
                ['Resolution Rate', f"{res_stats['resolution_rate']:.1f}%"],
                ['Pending Tickets', f"{res_stats['pending_tickets']:,}"]
            ])
        
        overview_table = Table(overview_data, colWidths=[2*inch, 2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(overview_table)
        story.append(PageBreak())
        
        # Add sections for each analysis
        sections = [
            ('issue_types', 'Issue Types Analysis'),
            ('priorities', 'Priority Analysis'),
            ('status', 'Status Analysis'),
            ('applications', 'Application Analysis'),
            ('resolution_times', 'Resolution Times Analysis'),
            ('team_performance', 'Team Performance Analysis'),
            ('time_trends', 'Time Trends Analysis')
        ]
        
        for section_key, section_title in sections:
            if section_key in self.stats:
                story.append(Paragraph(section_title, heading_style))
                
                # Add corresponding chart if available
                chart_name = section_key.replace('_', '_')
                chart_path = f"temp_chart_{chart_name}.png"
                
                if os.path.exists(chart_path):
                    try:
                        img = Image(chart_path, width=6*inch, height=4*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.2*inch))
                    except:
                        pass
                
                # Add key statistics
                self._add_section_stats(story, section_key, styles)
                story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        
        # Clean up temporary chart files
        for chart_path in self.charts:
            if os.path.exists(chart_path):
                os.remove(chart_path)
        
        print(f"✅ PDF report generated: {output_path}")
    
    def _add_section_stats(self, story, section_key, styles):
        """Add statistics for a specific section"""
        data = self.stats.get(section_key, {})
        
        if section_key == 'issue_types' and 'distribution' in data:
            dist = data['distribution']
            table_data = [['Issue Type', 'Count', 'Percentage']]
            for issue_type, count in list(dist.items())[:10]:  # Top 10
                percentage = data['percentages'].get(issue_type, 0)
                table_data.append([issue_type, f"{count:,}", f"{percentage:.1f}%"])
            
            table = Table(table_data, colWidths=[2.5*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        elif section_key == 'resolution_times' and 'mean_hours' in data:
            stats_data = [
                ['Metric', 'Value'],
                ['Average Resolution Time', f"{data['mean_hours']:.1f} hours ({data['mean_hours']/24:.1f} days)"],
                ['Median Resolution Time', f"{data['median_hours']:.1f} hours"],
                ['Fastest Resolution', f"{data['min_hours']:.1f} hours"],
                ['Slowest Resolution', f"{data['max_hours']:.1f} hours"],
                ['95th Percentile', f"{data['percentiles']['95th']:.1f} hours"]
            ]
            
            table = Table(stats_data, colWidths=[2.5*inch, 2.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        story.append(Spacer(1, 0.2*inch))
    
    def run_full_analysis(self, output_pdf="jira_comprehensive_report.pdf"):
        """Run complete analysis and generate report"""
        print("🚀 Starting Comprehensive Jira Analysis")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Run all analyses
        self.analyze_overview()
        self.analyze_issue_types()
        self.analyze_priorities()
        self.analyze_status()
        self.analyze_applications()
        self.analyze_resolution_times()
        self.analyze_team_performance()
        self.analyze_time_trends()
        
        # Generate PDF report
        self.generate_pdf_report(output_pdf)
        
        print("✅ Analysis Complete!")
        print(f"📄 Report saved as: {output_pdf}")
        
        return self.stats


def main():
    """Main function for command line usage"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python comprehensive_jira_analyzer.py <csv_file_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found!")
        sys.exit(1)
    
    analyzer = JiraAnalyzer(csv_path)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main() 