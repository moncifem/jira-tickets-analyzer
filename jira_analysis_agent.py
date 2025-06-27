"""
Jira Support Ticket Analysis Agent using LangGraph
This module provides an intelligent agent for analyzing large CSV files of Jira support tickets.
It uses LangGraph for orchestration and various tools for different types of analyses.
Generates a comprehensive PDF report at the end.
"""

import os
import pandas as pd
import dask.dataframe as dd
import numpy as np
from typing import Dict, List, Any, Optional, Annotated, Literal, TypedDict
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path
import logging
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import tempfile
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import PDF utilities
from pdf_utils import JiraAnalysisPDFGenerator

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JiraTicketState(MessagesState):
    """State for Jira ticket analysis"""
    csv_path: Optional[str]
    db_connection: Optional[str]
    current_analysis: Optional[Dict[str, Any]]
    visualizations: List[Dict[str, Any]]
    insights: List[str]
    total_rows: Optional[int]
    columns: Optional[List[str]]
    all_analyses: Dict[str, Any]  # Store all analysis results


class JiraAnalysisAgent:
    """Agent for analyzing Jira support tickets using LangGraph"""
    
    def __init__(self, csv_path: Optional[str] = None, use_sqlite: bool = None, sqlite_path: Optional[str] = None):
        """
        Initialize the Jira Analysis Agent
        
        Args:
            csv_path: Path to the CSV file
            use_sqlite: Whether to use SQLite for better performance (auto-detected if None)
            sqlite_path: Path to SQLite database file (auto-generated if None)
        """
        self.csv_path = csv_path
        self.sqlite_path = sqlite_path
        self.db_engine = None
        self.data_info = {}
        self.all_analyses = {}  # Store all analysis results
        
        # Auto-detect if SQLite should be used based on file size
        if csv_path and use_sqlite is None:
            file_size_gb = os.path.getsize(csv_path) / (1024**3)
            # Use SQLite for files larger than 1GB
            self.use_sqlite = file_size_gb > 1.0
            if self.use_sqlite:
                logger.info(f"File size is {file_size_gb:.2f}GB. Will use SQLite for better performance.")
        else:
            self.use_sqlite = use_sqlite if use_sqlite is not None else False
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0
        )
        
        # Initialize tools
        self.tools = [
            self.load_data_info,
            self.analyze_issue_types,
            self.analyze_applications,
            self.analyze_resolution_times,
            self.analyze_team_performance,
            self.analyze_trends_over_time,
            self.analyze_priority_distribution,
            self.find_bottlenecks,
            self.generate_insights,
            self.create_visualization
        ]
        
        # Create the agent
        self.memory = MemorySaver()
        self.agent = self._create_agent()
    
    def _get_sqlite_path(self, csv_path: str) -> str:
        """Generate SQLite database path based on CSV filename"""
        if self.sqlite_path:
            return self.sqlite_path
        
        # Create SQLite file in same directory as CSV
        csv_base = os.path.splitext(csv_path)[0]
        return f"{csv_base}_analysis.db"
    
    def _setup_sqlite_if_needed(self, csv_path: str):
        """Automatically set up SQLite database if needed"""
        if not self.use_sqlite or self.db_engine is not None:
            return
        
        sqlite_path = self._get_sqlite_path(csv_path)
        
        # Check if database already exists
        if os.path.exists(sqlite_path):
            logger.info(f"Using existing SQLite database: {sqlite_path}")
            self.db_engine = create_engine(f"sqlite:///{sqlite_path}")
            return
        
        logger.info(f"Creating SQLite database for better performance: {sqlite_path}")
        self.load_to_sqlite(csv_path, sqlite_path)
    
    def load_to_sqlite(self, csv_path: str, sqlite_path: Optional[str] = None):
        """
        Load CSV data into SQLite database for faster querying
        
        Args:
            csv_path: Path to CSV file
            sqlite_path: Path to SQLite database (auto-generated if None)
        """
        if sqlite_path is None:
            sqlite_path = self._get_sqlite_path(csv_path)
        
        logger.info(f"Loading data to SQLite: {sqlite_path}")
        
        # Create database connection
        engine = create_engine(f"sqlite:///{sqlite_path}")
        
        # Use chunking to load data
        chunk_size = 100000
        total_rows = 0
        
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            # Clean column names (remove spaces, special characters)
            chunk.columns = [col.replace(' ', '_').replace('/', '_').replace('-', '_') for col in chunk.columns]
            
            chunk.to_sql(
                'jira_tickets',
                engine,
                if_exists='append' if i > 0 else 'replace',
                index=False,
                method='multi'
            )
            total_rows += len(chunk)
            logger.info(f"Loaded {total_rows:,} rows...")
        
        # Create indexes for better performance
        with engine.connect() as conn:
            # Create indexes on commonly used columns
            try:
                conn.execute("CREATE INDEX idx_issue_type ON jira_tickets(Issue_Type)")
                conn.execute("CREATE INDEX idx_application ON jira_tickets(Application)")
                conn.execute("CREATE INDEX idx_created ON jira_tickets(Created)")
                conn.execute("CREATE INDEX idx_priority ON jira_tickets(Priority)")
                conn.execute("CREATE INDEX idx_status ON jira_tickets(Status)")
                conn.execute("CREATE INDEX idx_assignee ON jira_tickets(Assignee)")
                conn.commit()
            except Exception as e:
                logger.warning(f"Could not create indexes: {e}")
        
        self.db_engine = engine
        logger.info(f"✅ Data loaded to SQLite: {total_rows:,} rows")
        
        return sqlite_path
    
    def _read_data_smart(self, csv_path: str, columns: List[str], chunk_size: int = 100000):
        """
        Smart data reader that uses SQLite if available, otherwise reads CSV
        
        Args:
            csv_path: Path to CSV file
            columns: Columns to read
            chunk_size: Chunk size for CSV reading
            
        Yields:
            DataFrame chunks
        """
        # Setup SQLite if needed
        self._setup_sqlite_if_needed(csv_path)
        
        if self.db_engine:
            # Read from SQLite in chunks
            # Clean column names for SQL query
            sql_columns = [col.replace(' ', '_').replace('/', '_').replace('-', '_') for col in columns]
            query = f"SELECT {', '.join(sql_columns)} FROM jira_tickets"
            
            for chunk in pd.read_sql(query, self.db_engine, chunksize=chunk_size):
                # Restore original column names
                chunk.columns = columns
                yield chunk
        else:
            # Read from CSV
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=columns):
                yield chunk
    
    @tool
    def load_data_info(self, csv_path: str) -> Dict[str, Any]:
        """
        Load basic information about the CSV file without loading all data into memory.
        This tool samples the data to understand its structure.
        """
        logger.info(f"Loading data info from {csv_path}")
        
        # First, get basic file info
        file_size = os.path.getsize(csv_path)
        file_size_gb = file_size / (1024**3)
        
        # Sample first 10000 rows to understand structure
        sample_df = pd.read_csv(csv_path, nrows=10000)
        
        # Get column information
        columns = list(sample_df.columns)
        dtypes = {col: str(dtype) for col, dtype in sample_df.dtypes.items()}
        
        # Estimate total rows (rough estimate based on file size)
        avg_row_size = file_size / 10000 if len(sample_df) == 10000 else file_size / len(sample_df)
        estimated_rows = int(file_size / avg_row_size)
        
        # Identify date columns
        date_columns = []
        for col in columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'created' in col.lower() or 'resolved' in col.lower() or 'updated' in col.lower():
                date_columns.append(col)
        
        # Basic statistics
        null_counts = sample_df.isnull().sum().to_dict()
        
        # Get date range if possible
        date_range = "Unknown"
        if date_columns:
            try:
                sample_df[date_columns[0]] = pd.to_datetime(sample_df[date_columns[0]], errors='coerce')
                min_date = sample_df[date_columns[0]].min()
                max_date = sample_df[date_columns[0]].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            except:
                pass
        
        # Check if SQLite is recommended
        sqlite_recommended = file_size_gb > 1.0
        sqlite_status = "Not using SQLite"
        if self.db_engine:
            sqlite_status = "Using SQLite for better performance"
        elif sqlite_recommended:
            sqlite_status = "SQLite recommended for this file size"
        
        info = {
            "file_path": csv_path,
            "file_size_gb": round(file_size_gb, 2),
            "estimated_rows": estimated_rows,
            "columns": columns,
            "dtypes": dtypes,
            "date_columns": date_columns,
            "null_counts_sample": null_counts,
            "sample_rows": sample_df.head(5).to_dict('records'),
            "date_range": date_range,
            "sqlite_status": sqlite_status
        }
        
        return info
    
    @tool
    def analyze_issue_types(self, 
                          csv_path: str,
                          issue_type_column: str = "Issue Type",
                          chunk_size: int = 100000) -> Dict[str, Any]:
        """
        Analyze the distribution of issue types in the Jira tickets.
        Uses chunking to process large files efficiently.
        """
        logger.info("Analyzing issue types...")
        
        issue_counts = {}
        total_processed = 0
        
        # Process in chunks using smart reader
        for chunk in self._read_data_smart(csv_path, [issue_type_column], chunk_size):
            chunk_counts = chunk[issue_type_column].value_counts()
            for issue_type, count in chunk_counts.items():
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + count
            total_processed += len(chunk)
            
            if total_processed % 1000000 == 0:
                logger.info(f"Processed {total_processed:,} rows...")
        
        # Sort by count
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate percentages
        total_issues = sum(issue_counts.values())
        issue_distribution = [
            {
                "issue_type": issue_type,
                "count": count,
                "percentage": round((count / total_issues) * 100, 2)
            }
            for issue_type, count in sorted_issues[:20]  # Top 20 issue types
        ]
        
        # Create visualization
        if issue_distribution:
            top_10 = issue_distribution[:10]
            fig = px.bar(
                top_10,
                x="issue_type",
                y="count",
                title="Top 10 Issue Types Distribution",
                text="percentage",
                labels={"issue_type": "Issue Type", "count": "Number of Tickets"}
            )
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45)
            
            visualization = fig.to_json()
        else:
            visualization = None
        
        result = {
            "total_issues": total_issues,
            "unique_issue_types": len(issue_counts),
            "top_issue_types": issue_distribution,
            "analysis_type": "issue_types",
            "visualization": visualization
        }
        
        # Store in agent's memory
        self.all_analyses['issue_types_analysis'] = result
        
        return result
    
    @tool
    def analyze_applications(self,
                           csv_path: str,
                           app_column: str = "Application",
                           issue_column: str = "Issue Type",
                           chunk_size: int = 100000) -> Dict[str, Any]:
        """
        Analyze which applications have the most issues and what types of issues they have.
        """
        logger.info("Analyzing applications...")
        
        app_stats = {}
        total_processed = 0
        
        # Process in chunks using smart reader
        for chunk in self._read_data_smart(csv_path, [app_column, issue_column], chunk_size):
            for _, row in chunk.iterrows():
                app = row[app_column]
                issue = row[issue_column]
                
                if app not in app_stats:
                    app_stats[app] = {"total": 0, "issues": {}}
                
                app_stats[app]["total"] += 1
                app_stats[app]["issues"][issue] = app_stats[app]["issues"].get(issue, 0) + 1
            
            total_processed += len(chunk)
            if total_processed % 1000000 == 0:
                logger.info(f"Processed {total_processed:,} rows...")
        
        # Sort applications by total issues
        sorted_apps = sorted(app_stats.items(), key=lambda x: x[1]["total"], reverse=True)
        
        # Format results
        app_analysis = []
        for app, stats in sorted_apps[:15]:  # Top 15 applications
            top_issues = sorted(stats["issues"].items(), key=lambda x: x[1], reverse=True)[:5]
            app_analysis.append({
                "application": app,
                "total_issues": stats["total"],
                "top_issues": [
                    {"issue_type": issue, "count": count}
                    for issue, count in top_issues
                ]
            })
        
        # Create visualization
        if app_analysis:
            top_10_apps = app_analysis[:10]
            fig = px.bar(
                x=[app["application"] for app in top_10_apps],
                y=[app["total_issues"] for app in top_10_apps],
                title="Top 10 Applications by Issue Count",
                labels={"x": "Application", "y": "Number of Issues"}
            )
            fig.update_layout(xaxis_tickangle=-45)
            visualization = fig.to_json()
        else:
            visualization = None
        
        result = {
            "total_applications": len(app_stats),
            "application_analysis": app_analysis,
            "analysis_type": "applications",
            "visualization": visualization
        }
        
        # Store in agent's memory
        self.all_analyses['applications_analysis'] = result
        
        return result
    
    @tool
    def analyze_resolution_times(self,
                               csv_path: str,
                               created_column: str = "Created",
                               resolved_column: str = "Resolved",
                               app_column: str = "Application",
                               priority_column: str = "Priority",
                               chunk_size: int = 100000) -> Dict[str, Any]:
        """
        Analyze resolution times for tickets across different dimensions.
        """
        logger.info("Analyzing resolution times...")
        
        resolution_stats = {
            "by_application": {},
            "by_priority": {},
            "overall": []
        }
        
        total_processed = 0
        
        # Process in chunks using smart reader
        for chunk in self._read_data_smart(csv_path, 
                                         [created_column, resolved_column, app_column, priority_column], 
                                         chunk_size):
            # Convert to datetime
            chunk[created_column] = pd.to_datetime(chunk[created_column], errors='coerce')
            chunk[resolved_column] = pd.to_datetime(chunk[resolved_column], errors='coerce')
            
            # Calculate resolution time in hours
            chunk['resolution_hours'] = (chunk[resolved_column] - chunk[created_column]).dt.total_seconds() / 3600
            
            # Filter out invalid data
            valid_chunk = chunk[chunk['resolution_hours'].notna() & (chunk['resolution_hours'] > 0)]
            
            # Overall stats
            resolution_stats["overall"].extend(valid_chunk['resolution_hours'].tolist())
            
            # By application
            for app in valid_chunk[app_column].unique():
                if pd.notna(app):
                    app_data = valid_chunk[valid_chunk[app_column] == app]['resolution_hours']
                    if app not in resolution_stats["by_application"]:
                        resolution_stats["by_application"][app] = []
                    resolution_stats["by_application"][app].extend(app_data.tolist())
            
            # By priority
            for priority in valid_chunk[priority_column].unique():
                if pd.notna(priority):
                    priority_data = valid_chunk[valid_chunk[priority_column] == priority]['resolution_hours']
                    if priority not in resolution_stats["by_priority"]:
                        resolution_stats["by_priority"][priority] = []
                    resolution_stats["by_priority"][priority].extend(priority_data.tolist())
            
            total_processed += len(chunk)
            if total_processed % 1000000 == 0:
                logger.info(f"Processed {total_processed:,} rows...")
        
        # Calculate statistics
        def calculate_stats(data):
            if not data:
                return None
            return {
                "mean_hours": round(np.mean(data), 2),
                "median_hours": round(np.median(data), 2),
                "std_hours": round(np.std(data), 2),
                "min_hours": round(np.min(data), 2),
                "max_hours": round(np.max(data), 2),
                "count": len(data)
            }
        
        # Overall statistics
        overall_stats = calculate_stats(resolution_stats["overall"])
        
        # Application statistics (top 10)
        app_stats = {}
        for app, times in resolution_stats["by_application"].items():
            stats = calculate_stats(times)
            if stats:
                app_stats[app] = stats
        
        sorted_apps = sorted(app_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        
        # Priority statistics
        priority_stats = {}
        for priority, times in resolution_stats["by_priority"].items():
            stats = calculate_stats(times)
            if stats:
                priority_stats[priority] = stats
        
        # Create visualization for priority
        if priority_stats:
            priorities = list(priority_stats.keys())
            mean_times = [priority_stats[p]["mean_hours"] for p in priorities]
            
            fig = px.bar(
                x=priorities,
                y=mean_times,
                title="Average Resolution Time by Priority",
                labels={"x": "Priority", "y": "Average Hours"}
            )
            visualization = fig.to_json()
        else:
            visualization = None
        
        result = {
            "overall_stats": overall_stats,
            "by_application": dict(sorted_apps),
            "by_priority": priority_stats,
            "analysis_type": "resolution_times",
            "visualization": visualization
        }
        
        # Store in agent's memory
        self.all_analyses['resolution_times_analysis'] = result
        
        return result
    
    @tool
    def analyze_team_performance(self,
                               csv_path: str,
                               assignee_column: str = "Assignee",
                               resolved_column: str = "Resolved",
                               created_column: str = "Created",
                               priority_column: str = "Priority",
                               chunk_size: int = 100000) -> Dict[str, Any]:
        """
        Analyze team member performance metrics.
        """
        logger.info("Analyzing team performance...")
        
        team_stats = {}
        total_processed = 0
        
        # Process in chunks using smart reader
        for chunk in self._read_data_smart(csv_path,
                                         [assignee_column, created_column, resolved_column, priority_column],
                                         chunk_size):
            # Convert to datetime
            chunk[created_column] = pd.to_datetime(chunk[created_column], errors='coerce')
            chunk[resolved_column] = pd.to_datetime(chunk[resolved_column], errors='coerce')
            
            for _, row in chunk.iterrows():
                assignee = row[assignee_column]
                if pd.notna(assignee):
                    if assignee not in team_stats:
                        team_stats[assignee] = {
                            "total_tickets": 0,
                            "resolved_tickets": 0,
                            "resolution_times": [],
                            "priority_counts": {}
                        }
                    
                    team_stats[assignee]["total_tickets"] += 1
                    
                    if pd.notna(row[resolved_column]):
                        team_stats[assignee]["resolved_tickets"] += 1
                        
                        if pd.notna(row[created_column]):
                            resolution_time = (row[resolved_column] - row[created_column]).total_seconds() / 3600
                            if resolution_time > 0:
                                team_stats[assignee]["resolution_times"].append(resolution_time)
                    
                    priority = row[priority_column]
                    if pd.notna(priority):
                        team_stats[assignee]["priority_counts"][priority] = \
                            team_stats[assignee]["priority_counts"].get(priority, 0) + 1
            
            total_processed += len(chunk)
            if total_processed % 1000000 == 0:
                logger.info(f"Processed {total_processed:,} rows...")
        
        # Calculate performance metrics
        team_performance = []
        for assignee, stats in team_stats.items():
            if stats["total_tickets"] > 10:  # Filter out users with very few tickets
                avg_resolution = np.mean(stats["resolution_times"]) if stats["resolution_times"] else None
                
                performance = {
                    "assignee": assignee,
                    "total_tickets": stats["total_tickets"],
                    "resolved_tickets": stats["resolved_tickets"],
                    "resolution_rate": round((stats["resolved_tickets"] / stats["total_tickets"]) * 100, 2),
                    "avg_resolution_hours": round(avg_resolution, 2) if avg_resolution else None,
                    "priority_distribution": stats["priority_counts"]
                }
                team_performance.append(performance)
        
        # Sort by total tickets
        team_performance.sort(key=lambda x: x["total_tickets"], reverse=True)
        
        result = {
            "team_size": len(team_performance),
            "top_performers": team_performance[:20],
            "analysis_type": "team_performance"
        }
        
        # Store in agent's memory
        self.all_analyses['team_performance_analysis'] = result
        
        return result
    
    @tool
    def analyze_trends_over_time(self,
                               csv_path: str,
                               created_column: str = "Created",
                               issue_column: str = "Issue Type",
                               app_column: str = "Application",
                               chunk_size: int = 100000) -> Dict[str, Any]:
        """
        Analyze trends over time for ticket creation.
        """
        logger.info("Analyzing trends over time...")
        
        monthly_stats = {}
        total_processed = 0
        
        # Process in chunks using smart reader
        for chunk in self._read_data_smart(csv_path,
                                         [created_column, issue_column, app_column],
                                         chunk_size):
            # Convert to datetime
            chunk[created_column] = pd.to_datetime(chunk[created_column], errors='coerce')
            
            # Extract year-month
            chunk['year_month'] = chunk[created_column].dt.to_period('M')
            
            for _, row in chunk.iterrows():
                if pd.notna(row['year_month']):
                    period = str(row['year_month'])
                    
                    if period not in monthly_stats:
                        monthly_stats[period] = {
                            "total": 0,
                            "issues": {},
                            "applications": {}
                        }
                    
                    monthly_stats[period]["total"] += 1
                    
                    # Track issues
                    issue = row[issue_column]
                    if pd.notna(issue):
                        monthly_stats[period]["issues"][issue] = \
                            monthly_stats[period]["issues"].get(issue, 0) + 1
                    
                    # Track applications
                    app = row[app_column]
                    if pd.notna(app):
                        monthly_stats[period]["applications"][app] = \
                            monthly_stats[period]["applications"].get(app, 0) + 1
            
            total_processed += len(chunk)
            if total_processed % 1000000 == 0:
                logger.info(f"Processed {total_processed:,} rows...")
        
        # Sort by period
        sorted_periods = sorted(monthly_stats.items())
        
        # Format for output
        trend_data = []
        for period, stats in sorted_periods[-24:]:  # Last 24 months
            top_issues = sorted(stats["issues"].items(), key=lambda x: x[1], reverse=True)[:3]
            top_apps = sorted(stats["applications"].items(), key=lambda x: x[1], reverse=True)[:3]
            
            trend_data.append({
                "period": period,
                "total_tickets": stats["total"],
                "top_issues": [{"type": issue, "count": count} for issue, count in top_issues],
                "top_applications": [{"app": app, "count": count} for app, count in top_apps]
            })
        
        # Create visualization
        if trend_data:
            periods = [d["period"] for d in trend_data]
            totals = [d["total_tickets"] for d in trend_data]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=periods,
                y=totals,
                mode='lines+markers',
                name='Total Tickets',
                line=dict(width=3)
            ))
            fig.update_layout(
                title="Ticket Volume Trend Over Time",
                xaxis_title="Period",
                yaxis_title="Number of Tickets",
                hovermode='x unified'
            )
            visualization = fig.to_json()
        else:
            visualization = None
        
        result = {
            "trend_data": trend_data,
            "total_periods": len(monthly_stats),
            "analysis_type": "trends",
            "visualization": visualization
        }
        
        # Store in agent's memory
        self.all_analyses['trends_analysis'] = result
        
        return result
    
    @tool
    def analyze_priority_distribution(self,
                                    csv_path: str,
                                    priority_column: str = "Priority",
                                    status_column: str = "Status",
                                    chunk_size: int = 100000) -> Dict[str, Any]:
        """
        Analyze the distribution of ticket priorities and their current status.
        """
        logger.info("Analyzing priority distribution...")
        
        priority_stats = {}
        total_processed = 0
        
        # Process in chunks using smart reader
        for chunk in self._read_data_smart(csv_path,
                                         [priority_column, status_column],
                                         chunk_size):
            for _, row in chunk.iterrows():
                priority = row[priority_column]
                status = row[status_column]
                
                if pd.notna(priority):
                    if priority not in priority_stats:
                        priority_stats[priority] = {
                            "total": 0,
                            "statuses": {}
                        }
                    
                    priority_stats[priority]["total"] += 1
                    
                    if pd.notna(status):
                        priority_stats[priority]["statuses"][status] = \
                            priority_stats[priority]["statuses"].get(status, 0) + 1
            
            total_processed += len(chunk)
            if total_processed % 1000000 == 0:
                logger.info(f"Processed {total_processed:,} rows...")
        
        # Calculate totals
        total_tickets = sum(stats["total"] for stats in priority_stats.values())
        
        # Format results
        priority_analysis = []
        for priority, stats in priority_stats.items():
            priority_analysis.append({
                "priority": priority,
                "count": stats["total"],
                "percentage": round((stats["total"] / total_tickets) * 100, 2),
                "status_breakdown": stats["statuses"]
            })
        
        # Sort by count
        priority_analysis.sort(key=lambda x: x["count"], reverse=True)
        
        # Create visualization
        if priority_analysis:
            priorities = [p["priority"] for p in priority_analysis]
            counts = [p["count"] for p in priority_analysis]
            
            fig = px.pie(
                values=counts,
                names=priorities,
                title="Ticket Distribution by Priority"
            )
            visualization = fig.to_json()
        else:
            visualization = None
        
        result = {
            "priority_distribution": priority_analysis,
            "total_tickets": total_tickets,
            "analysis_type": "priority",
            "visualization": visualization
        }
        
        # Store in agent's memory
        self.all_analyses['priority_analysis'] = result
        
        return result
    
    @tool
    def find_bottlenecks(self,
                        csv_path: str,
                        status_column: str = "Status",
                        created_column: str = "Created",
                        updated_column: str = "Updated",
                        assignee_column: str = "Assignee",
                        chunk_size: int = 100000) -> Dict[str, Any]:
        """
        Identify bottlenecks in the support process.
        """
        logger.info("Finding bottlenecks...")
        
        status_stats = {}
        stale_tickets = []
        overloaded_assignees = {}
        total_processed = 0
        
        current_date = datetime.now()
        stale_threshold = current_date - timedelta(days=30)  # Tickets not updated in 30 days
        
        # Process in chunks using smart reader
        for chunk in self._read_data_smart(csv_path,
                                         [status_column, created_column, updated_column, assignee_column],
                                         chunk_size):
            # Convert to datetime
            chunk[created_column] = pd.to_datetime(chunk[created_column], errors='coerce')
            chunk[updated_column] = pd.to_datetime(chunk[updated_column], errors='coerce')
            
            # Status distribution
            status_counts = chunk[status_column].value_counts()
            for status, count in status_counts.items():
                status_stats[status] = status_stats.get(status, 0) + count
            
            # Find stale tickets
            stale_mask = (chunk[updated_column] < stale_threshold) & \
                        (chunk[status_column].notna()) & \
                        (~chunk[status_column].str.lower().isin(['closed', 'resolved', 'done']))
            
            stale_count = stale_mask.sum()
            if stale_count > 0:
                stale_tickets.append({
                    "count": int(stale_count),
                    "sample_statuses": chunk[stale_mask][status_column].value_counts().head(5).to_dict()
                })
            
            # Assignee workload
            open_tickets = chunk[~chunk[status_column].str.lower().isin(['closed', 'resolved', 'done'])]
            assignee_counts = open_tickets[assignee_column].value_counts()
            for assignee, count in assignee_counts.items():
                if pd.notna(assignee):
                    overloaded_assignees[assignee] = overloaded_assignees.get(assignee, 0) + count
            
            total_processed += len(chunk)
            if total_processed % 1000000 == 0:
                logger.info(f"Processed {total_processed:,} rows...")
        
        # Identify bottlenecks
        total_stale = sum(item["count"] for item in stale_tickets)
        
        # Find overloaded assignees (top 10 with most open tickets)
        sorted_assignees = sorted(overloaded_assignees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Status bottlenecks (statuses with high counts that aren't closed)
        open_statuses = {k: v for k, v in status_stats.items() 
                        if k and not any(closed in str(k).lower() for closed in ['closed', 'resolved', 'done'])}
        sorted_statuses = sorted(open_statuses.items(), key=lambda x: x[1], reverse=True)[:10]
        
        result = {
            "stale_tickets": {
                "total": total_stale,
                "threshold_days": 30
            },
            "overloaded_assignees": [
                {"assignee": assignee, "open_tickets": count}
                for assignee, count in sorted_assignees
            ],
            "status_bottlenecks": [
                {"status": status, "count": count}
                for status, count in sorted_statuses
            ],
            "analysis_type": "bottlenecks"
        }
        
        # Store in agent's memory
        self.all_analyses['bottlenecks_analysis'] = result
        
        return result
    
    @tool
    def generate_insights(self, analysis_results: List[Dict[str, Any]]) -> List[str]:
        """
        Generate actionable insights based on the analysis results.
        """
        logger.info("Generating insights...")
        
        insights = []
        
        # Use all stored analyses if available
        all_results = list(self.all_analyses.values()) if self.all_analyses else analysis_results
        
        # Analyze each result type
        for result in all_results:
            if not isinstance(result, dict) or "analysis_type" not in result:
                continue
                
            analysis_type = result.get("analysis_type")
            
            if analysis_type == "issue_types":
                top_issue = result.get("top_issue_types", [{}])[0]
                if top_issue:
                    insights.append(
                        f"📊 The most common issue type is '{top_issue.get('issue_type')}' "
                        f"accounting for {top_issue.get('percentage')}% of all tickets. "
                        f"Consider creating automated solutions or documentation for this issue type."
                    )
                
                # Check for issue concentration
                top_5_issues = result.get("top_issue_types", [])[:5]
                if top_5_issues:
                    total_percentage = sum(issue.get('percentage', 0) for issue in top_5_issues)
                    if total_percentage > 60:
                        insights.append(
                            f"🎯 The top 5 issue types account for {total_percentage}% of all tickets. "
                            f"Focusing on these issues could significantly reduce support load."
                        )
            
            elif analysis_type == "applications":
                app_analysis = result.get("application_analysis", [])
                if app_analysis:
                    top_app = app_analysis[0]
                    insights.append(
                        f"🎯 '{top_app.get('application')}' has the most issues "
                        f"({top_app.get('total_issues'):,} tickets). "
                        f"This application may need architectural review or additional resources."
                    )
                    
                    # Check for disproportionate issues
                    if len(app_analysis) > 1:
                        top_3_total = sum(app.get('total_issues', 0) for app in app_analysis[:3])
                        all_total = sum(app.get('total_issues', 0) for app in app_analysis)
                        if all_total > 0:
                            percentage = (top_3_total / all_total) * 100
                            if percentage > 70:
                                insights.append(
                                    f"⚠️ The top 3 applications account for {percentage:.1f}% of all issues. "
                                    f"Consider prioritizing improvements for these applications."
                                )
            
            elif analysis_type == "resolution_times":
                overall = result.get("overall_stats", {})
                if overall:
                    avg_hours = overall.get("mean_hours", 0)
                    median_hours = overall.get("median_hours", 0)
                    
                    if avg_hours > 72:
                        insights.append(
                            f"⏱️ Average resolution time is {avg_hours:.1f} hours "
                            f"({avg_hours/24:.1f} days), which is quite high. "
                            f"Consider implementing SLA policies and escalation procedures."
                        )
                    
                    # Check for outliers
                    if avg_hours > median_hours * 2:
                        insights.append(
                            f"📈 There's a significant difference between average ({avg_hours:.1f} hrs) "
                            f"and median ({median_hours:.1f} hrs) resolution times, "
                            f"indicating some tickets take exceptionally long to resolve."
                        )
                    
                    # Priority-based insights
                    by_priority = result.get("by_priority", {})
                    if by_priority:
                        high_priority = None
                        low_priority = None
                        
                        for priority, stats in by_priority.items():
                            if 'high' in priority.lower() or 'critical' in priority.lower() or 'urgent' in priority.lower():
                                high_priority = stats
                            elif 'low' in priority.lower():
                                low_priority = stats
                        
                        if high_priority and low_priority:
                            high_avg = high_priority.get('mean_hours', 0)
                            low_avg = low_priority.get('mean_hours', 0)
                            
                            if high_avg > low_avg:
                                insights.append(
                                    f"⚠️ High priority tickets take longer to resolve ({high_avg:.1f} hrs) "
                                    f"than low priority tickets ({low_avg:.1f} hrs). "
                                    f"This suggests a need to review the prioritization process."
                                )
            
            elif analysis_type == "team_performance":
                performers = result.get("top_performers", [])
                if performers:
                    # Workload distribution
                    total_tickets = sum(p.get('total_tickets', 0) for p in performers)
                    top_5_tickets = sum(p.get('total_tickets', 0) for p in performers[:5])
                    
                    if total_tickets > 0:
                        percentage = (top_5_tickets / total_tickets) * 100
                        if percentage > 50:
                            insights.append(
                                f"👥 The top 5 team members handle {percentage:.1f}% of all tickets. "
                                f"Consider distributing workload more evenly to prevent burnout."
                            )
                    
                    # Resolution rates
                    low_resolution_members = [
                        p for p in performers 
                        if p.get('resolution_rate', 100) < 70 and p.get('total_tickets', 0) > 50
                    ]
                    
                    if low_resolution_members:
                        insights.append(
                            f"📉 {len(low_resolution_members)} team members have resolution rates below 70%. "
                            f"They may need additional training or support."
                        )
            
            elif analysis_type == "bottlenecks":
                stale = result.get("stale_tickets", {})
                if stale.get("total", 0) > 1000:
                    insights.append(
                        f"🚨 There are {stale.get('total'):,} stale tickets "
                        f"(not updated in {stale.get('threshold_days')} days). "
                        f"Implement a regular review process for aging tickets."
                    )
                
                overloaded = result.get("overloaded_assignees", [])
                if overloaded and overloaded[0].get('open_tickets', 0) > 100:
                    top_assignee = overloaded[0]
                    insights.append(
                        f"🔥 {top_assignee.get('assignee')} has {top_assignee.get('open_tickets')} open tickets. "
                        f"Consider reassigning some tickets to balance the workload."
                    )
            
            elif analysis_type == "trends":
                trend_data = result.get("trend_data", [])
                if len(trend_data) >= 3:
                    # Check recent trend
                    recent_3 = trend_data[-3:]
                    if all('total_tickets' in t for t in recent_3):
                        totals = [t['total_tickets'] for t in recent_3]
                        
                        # Check if increasing
                        if totals[2] > totals[1] > totals[0]:
                            increase_pct = ((totals[2] - totals[0]) / totals[0]) * 100
                            insights.append(
                                f"📈 Ticket volume has increased by {increase_pct:.1f}% over the last 3 months. "
                                f"Consider scaling support resources accordingly."
                            )
                        
                        # Check if decreasing
                        elif totals[2] < totals[1] < totals[0]:
                            decrease_pct = ((totals[0] - totals[2]) / totals[0]) * 100
                            insights.append(
                                f"📉 Ticket volume has decreased by {decrease_pct:.1f}% over the last 3 months. "
                                f"This could indicate improved product stability or reduced usage."
                            )
        
        # Store insights
        self.all_analyses['insights'] = insights
        
        return insights
    
    @tool
    def create_visualization(self,
                           analysis_data: Dict[str, Any],
                           viz_type: str = "auto") -> Dict[str, Any]:
        """
        Create visualizations based on analysis data.
        """
        logger.info(f"Creating visualization of type: {viz_type}")
        
        analysis_type = analysis_data.get("analysis_type")
        
        if analysis_type == "issue_types":
            # Create bar chart for issue types
            data = analysis_data.get("top_issue_types", [])
            if data:
                fig = px.bar(
                    data,
                    x="issue_type",
                    y="count",
                    title="Top Issue Types Distribution",
                    text="percentage"
                )
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                
                return {
                    "type": "bar_chart",
                    "title": "Issue Types Distribution",
                    "figure": fig.to_json()
                }
        
        elif analysis_type == "trends":
            # Create line chart for trends
            data = analysis_data.get("trend_data", [])
            if data:
                periods = [d["period"] for d in data]
                totals = [d["total_tickets"] for d in data]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=periods,
                    y=totals,
                    mode='lines+markers',
                    name='Total Tickets'
                ))
                fig.update_layout(
                    title="Ticket Volume Trend Over Time",
                    xaxis_title="Period",
                    yaxis_title="Number of Tickets"
                )
                
                return {
                    "type": "line_chart",
                    "title": "Ticket Volume Trend",
                    "figure": fig.to_json()
                }
        
        return {"type": "none", "message": "No visualization available for this data type"}
    
    def _create_agent(self):
        """Create the LangGraph agent"""
        
        # Create the graph
        workflow = StateGraph(JiraTicketState)
        
        # Define the agent node
        def agent_node(state: JiraTicketState):
            """Main agent logic"""
            messages = state["messages"]
            
            # If this is the first message, provide context
            if len(messages) == 1:
                system_message = SystemMessage(content="""
                You are an expert data analyst specializing in Jira support ticket analysis.
                You have access to tools for analyzing large CSV files containing support ticket data.
                
                Your goal is to help the team understand:
                1. What types of issues they're dealing with
                2. Which applications have the most problems
                3. How long it takes to resolve tickets
                4. Team performance metrics
                5. Trends and bottlenecks
                
                Always start by loading the data info to understand the structure.
                Then perform comprehensive analysis using all available tools.
                Generate insights and visualizations for each analysis.
                """)
                messages = [system_message] + messages
            
            # Use React agent pattern
            response = self.llm.invoke(messages)
            
            return {"messages": [response]}
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Define conditional edges
        def should_continue(state: JiraTicketState) -> Literal["tools", "end"]:
            messages = state["messages"]
            last_message = messages[-1]
            
            # If the LLM makes a tool call, execute tools
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            
            # Otherwise, end
            return "end"
        
        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        # Compile
        return workflow.compile(checkpointer=self.memory)
    
    async def analyze(self, query: str, csv_path: str) -> Dict[str, Any]:
        """
        Run analysis based on user query
        
        Args:
            query: User's analysis request
            csv_path: Path to the CSV file
            
        Returns:
            Analysis results with insights and visualizations
        """
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=f"Analyze this Jira data: {query}. The CSV file is at: {csv_path}")],
            "csv_path": csv_path,
            "visualizations": [],
            "insights": [],
            "all_analyses": {}
        }
        
        # Run the agent
        config = {"configurable": {"thread_id": "jira-analysis"}}
        
        result = await self.agent.ainvoke(initial_state, config)
        
        return {
            "response": result["messages"][-1].content,
            "insights": result.get("insights", []),
            "visualizations": result.get("visualizations", []),
            "all_analyses": self.all_analyses
        }
    
    async def run_comprehensive_analysis(self, csv_path: str, output_pdf: str = "jira_analysis_report.pdf"):
        """
        Run a comprehensive analysis of the Jira data and generate PDF report
        
        Args:
            csv_path: Path to the CSV file
            output_pdf: Name of the output PDF file
        """
        print(f"\n{'='*60}")
        print("🚀 Starting Comprehensive Jira Analysis")
        print(f"{'='*60}\n")
        
        # Load data info first
        print("📊 Loading data information...")
        data_info = self.load_data_info.invoke({"csv_path": csv_path})
        
        print(f"✅ File: {csv_path}")
        print(f"✅ Size: {data_info['file_size_gb']} GB")
        print(f"✅ Estimated rows: {data_info['estimated_rows']:,}")
        print(f"✅ Columns: {len(data_info['columns'])}")
        print(f"✅ Date range: {data_info.get('date_range', 'Unknown')}")
        print(f"✅ SQLite status: {data_info.get('sqlite_status', 'Not using SQLite')}")
        
        # Store basic info
        self.all_analyses['total_tickets'] = data_info['estimated_rows']
        self.all_analyses['date_range'] = data_info.get('date_range', 'Unknown')
        
        print("\n📈 Running analyses...")
        
        # 1. Issue Types Analysis
        print("\n1️⃣ Analyzing issue types...")
        issue_analysis = self.analyze_issue_types.invoke({
            "csv_path": csv_path,
            "issue_type_column": "Issue Type"
        })
        print(f"   ✓ Found {issue_analysis['unique_issue_types']} unique issue types")
        
        # 2. Applications Analysis
        print("\n2️⃣ Analyzing applications...")
        app_analysis = self.analyze_applications.invoke({
            "csv_path": csv_path,
            "app_column": "Application",
            "issue_column": "Issue Type"
        })
        print(f"   ✓ Found {app_analysis['total_applications']} applications with issues")
        self.all_analyses['total_applications'] = app_analysis['total_applications']
        
        # 3. Resolution Times Analysis
        print("\n3️⃣ Analyzing resolution times...")
        resolution_analysis = self.analyze_resolution_times.invoke({
            "csv_path": csv_path,
            "created_column": "Created",
            "resolved_column": "Resolved",
            "app_column": "Application",
            "priority_column": "Priority"
        })
        if resolution_analysis['overall_stats']:
            avg_hours = resolution_analysis['overall_stats']['mean_hours']
            print(f"   ✓ Average resolution time: {avg_hours:.1f} hours ({avg_hours/24:.1f} days)")
        
        # 4. Team Performance Analysis
        print("\n4️⃣ Analyzing team performance...")
        team_analysis = self.analyze_team_performance.invoke({
            "csv_path": csv_path,
            "assignee_column": "Assignee",
            "resolved_column": "Resolved",
            "created_column": "Created",
            "priority_column": "Priority"
        })
        print(f"   ✓ Found {team_analysis['team_size']} active team members")
        self.all_analyses['team_size'] = team_analysis['team_size']
        
        # 5. Trends Analysis
        print("\n5️⃣ Analyzing trends over time...")
        trends_analysis = self.analyze_trends_over_time.invoke({
            "csv_path": csv_path,
            "created_column": "Created",
            "issue_column": "Issue Type",
            "app_column": "Application"
        })
        print(f"   ✓ Analyzed {len(trends_analysis['trend_data'])} time periods")
        
        # 6. Priority Analysis
        print("\n6️⃣ Analyzing priority distribution...")
        priority_analysis = self.analyze_priority_distribution.invoke({
            "csv_path": csv_path,
            "priority_column": "Priority",
            "status_column": "Status"
        })
        print(f"   ✓ Analyzed {len(priority_analysis['priority_distribution'])} priority levels")
        
        # 7. Bottlenecks Analysis
        print("\n7️⃣ Finding process bottlenecks...")
        bottlenecks_analysis = self.find_bottlenecks.invoke({
            "csv_path": csv_path,
            "status_column": "Status",
            "created_column": "Created",
            "updated_column": "Updated",
            "assignee_column": "Assignee"
        })
        stale_count = bottlenecks_analysis['stale_tickets']['total']
        print(f"   ✓ Found {stale_count:,} stale tickets")
        
        # 8. Generate Insights
        print("\n8️⃣ Generating insights...")
        all_analyses = list(self.all_analyses.values())
        insights = self.generate_insights.invoke({"analysis_results": all_analyses})
        print(f"   ✓ Generated {len(insights)} key insights")
        
        # Generate PDF Report
        print(f"\n📄 Generating PDF report: {output_pdf}")
        pdf_generator = JiraAnalysisPDFGenerator(output_pdf)
        pdf_generator.generate_report(self.all_analyses)
        
        print(f"\n{'='*60}")
        print("✅ Analysis Complete!")
        print(f"📊 PDF Report: {output_pdf}")
        print(f"💡 Key Insights: {len(insights)}")
        if self.db_engine:
            print(f"🗄️ SQLite database: {self._get_sqlite_path(csv_path)}")
        print(f"{'='*60}\n")
        
        # Print top insights
        if insights:
            print("🔍 Top Insights:")
            for i, insight in enumerate(insights[:5], 1):
                print(f"\n{i}. {insight}")
        
        return self.all_analyses
    
    def load_to_database(self, csv_path: str, db_url: str, table_name: str = "jira_tickets"):
        """
        Load CSV data into a database for faster querying
        
        Args:
            csv_path: Path to CSV file
            db_url: Database connection URL
            table_name: Name of the table to create
        """
        logger.info(f"Loading data to database table: {table_name}")
        
        engine = create_engine(db_url)
        
        # Use chunking to load data
        chunk_size = 100000
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            chunk.to_sql(
                table_name,
                engine,
                if_exists='append' if i > 0 else 'replace',
                index=False,
                method='multi'
            )
            logger.info(f"Loaded chunk {i+1} ({len(chunk)} rows)")
        
        self.db_engine = engine
        logger.info("Data loading complete")


# Main function for standalone execution
async def main():
    """Main function to run the Jira analysis"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Jira support tickets and generate PDF report")
    parser.add_argument("csv_file", help="Path to the Jira CSV file")
    parser.add_argument("-o", "--output", default="jira_analysis_report.pdf", help="Output PDF filename")
    parser.add_argument("-d", "--database", help="Database URL for loading data (optional)")
    parser.add_argument("--no-sqlite", action="store_true", help="Disable automatic SQLite usage for large files")
    
    args = parser.parse_args()
    
    # Validate CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"❌ Error: CSV file not found: {args.csv_file}")
        return
    
    # Initialize the agent
    agent = JiraAnalysisAgent(
        csv_path=args.csv_file,
        use_sqlite=not args.no_sqlite  # Use SQLite by default unless disabled
    )
    
    # Load to database if requested
    if args.database:
        print(f"📦 Loading data to database...")
        agent.load_to_database(args.csv_file, args.database)
        print(f"✅ Data loaded to database")
    
    # Run comprehensive analysis
    await agent.run_comprehensive_analysis(args.csv_file, args.output)


if __name__ == "__main__":
    asyncio.run(main()) 