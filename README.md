# Jira Support Ticket Analysis Agent

A powerful LangGraph-based agent for analyzing large Jira support ticket CSV files (even with billions of rows) and generating comprehensive PDF reports with insights and visualizations.

## Features

- **Handles Large Files**: Efficiently processes CSV files with billions of rows using chunking and automatic SQLite optimization
- **Smart SQLite Integration**: 
  - Automatically uses SQLite for files > 1GB for better performance
  - Creates local `.db` file alongside your CSV
  - Reuses existing SQLite database on subsequent runs
  - No external database setup required!
- **Comprehensive Analysis**: 
  - Issue type distribution
  - Application performance metrics
  - Resolution time analysis
  - Team performance tracking
  - Trend analysis over time
  - Priority distribution
  - Bottleneck identification
- **Intelligent Insights**: AI-powered insights generation using GPT-4
- **Professional PDF Reports**: Beautiful PDF reports with charts, tables, and actionable insights
- **Memory Efficient**: Uses chunking to process files larger than available RAM

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. Clone or download this project:
```bash
cd jira
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

### Basic Usage

Run comprehensive analysis on your Jira CSV file:

```bash
python jira_analysis_agent.py path/to/your/jira_tickets.csv
```

This will:
1. Automatically detect if SQLite should be used (for files > 1GB)
2. Load and analyze your CSV file
3. Run all available analyses
4. Generate insights
5. Create a PDF report named `jira_analysis_report.pdf`

### Advanced Usage

#### Custom Output Filename
```bash
python jira_analysis_agent.py tickets.csv -o my_custom_report.pdf
```

#### Disable Automatic SQLite (for smaller files)
```bash
python jira_analysis_agent.py small_file.csv --no-sqlite
```

#### Use External Database (PostgreSQL, MySQL, etc.)
```bash
python jira_analysis_agent.py tickets.csv -d postgresql://user:password@localhost/dbname
```

### How SQLite Optimization Works

For files larger than 1GB, the agent automatically:

1. **Creates a SQLite database**: Named `your_file_analysis.db` in the same directory
2. **Loads data with indexes**: Creates indexes on commonly queried columns
3. **Reuses on subsequent runs**: If the `.db` file exists, it uses it directly
4. **No manual setup needed**: Completely automatic!

Example:
```
jira_tickets.csv (5GB) → jira_tickets_analysis.db (created automatically)
```

Benefits:
- **10-50x faster** queries on large datasets
- **Lower memory usage** - doesn't load entire file into RAM
- **Persistent**: Reuse the database for multiple analyses

### Expected CSV Format

The agent expects a CSV file with the following columns (typical Jira export):
- **Issue Type**: Type of issue (Bug, Feature Request, Support, etc.)
- **Application**: Application or component name
- **Created**: Creation timestamp
- **Resolved**: Resolution timestamp (can be empty)
- **Updated**: Last update timestamp
- **Priority**: Priority level (High, Medium, Low, etc.)
- **Status**: Current status (Open, In Progress, Resolved, etc.)
- **Assignee**: Team member assigned to the ticket

Note: Column names are configurable in the code if your CSV has different names.

## What the Agent Analyzes

### 1. Issue Types Analysis
- Distribution of different issue types
- Top 20 most common issues
- Percentage breakdown
- Visual bar chart

### 2. Applications Analysis
- Which applications have the most issues
- Top issues per application
- Application ranking by ticket volume

### 3. Resolution Times
- Overall resolution statistics (mean, median, min, max)
- Resolution times by priority
- Resolution times by application
- Identifies outliers and anomalies

### 4. Team Performance
- Individual team member metrics
- Resolution rates
- Workload distribution
- Priority handling patterns

### 5. Trends Over Time
- Monthly ticket volume trends
- Seasonal patterns
- Growth/decline analysis
- Top issues by month

### 6. Priority Distribution
- How tickets are distributed across priorities
- Status breakdown by priority
- Priority handling effectiveness

### 7. Bottlenecks
- Stale tickets (not updated in 30+ days)
- Overloaded team members
- Process bottlenecks by status

## Generated PDF Report

The PDF report includes:
- **Title Page**: Overview with key metrics
- **Executive Summary**: High-level insights and recommendations
- **Detailed Analysis Sections**: Each analysis with tables and charts
- **Visualizations**: Interactive charts converted to images
- **Insights & Recommendations**: AI-generated actionable insights

## Example Insights

The agent generates insights like:
- "The most common issue type is 'Login Problems' accounting for 23.5% of all tickets. Consider creating automated solutions or documentation for this issue type."
- "Average resolution time is 156.3 hours (6.5 days), which is quite high. Consider implementing SLA policies and escalation procedures."
- "The top 5 team members handle 67.2% of all tickets. Consider distributing workload more evenly to prevent burnout."
- "There are 5,234 stale tickets (not updated in 30 days). Implement a regular review process for aging tickets."

## Performance Tips

1. **For files < 1GB**: The agent processes directly from CSV efficiently

2. **For files > 1GB**: SQLite is automatically used:
   ```bash
   # First run - creates SQLite database
   python jira_analysis_agent.py large_file.csv
   # Creates: large_file_analysis.db
   
   # Subsequent runs - uses existing database (much faster!)
   python jira_analysis_agent.py large_file.csv
   ```

3. **For repeated analyses**: The SQLite database is preserved, making subsequent runs much faster

4. **Memory optimization**: The agent processes data in chunks of 100,000 rows by default

## Troubleshooting

### Out of Memory Errors
- The agent is designed to handle large files through chunking and SQLite
- SQLite is automatically used for files > 1GB
- For extremely large files (>10GB), ensure you have enough disk space for the SQLite database

### Slow Performance
- First run creates SQLite database (one-time cost)
- Subsequent runs are much faster using the database
- Processing billions of rows takes time (expect 30-60 minutes for initial load)
- Use SSD storage for better I/O performance

### Missing Columns
- Check that your CSV has the expected column names
- You can modify the column names in the agent code to match your CSV

### SQLite Issues
- Ensure you have write permissions in the CSV directory
- The `.db` file will be about the same size as your CSV
- Delete the `.db` file to force a fresh analysis

## Customization

You can customize the agent by modifying:
- Column names in each analysis tool
- Chunk size for processing (default: 100,000 rows)
- SQLite threshold (default: 1GB)
- Analysis thresholds and parameters
- PDF report styling and content
- Additional analyses by creating new tools

## Example Output

```
🚀 Starting Comprehensive Jira Analysis
============================================================

📊 Loading data information...
✅ File: support_tickets.csv
✅ Size: 45.3 GB
✅ Estimated rows: 1,234,567,890
✅ Columns: 15
✅ Date range: 2020-01-01 to 2024-01-15
✅ SQLite status: Creating SQLite database for better performance

📈 Running analyses...

1️⃣ Analyzing issue types...
   ✓ Found 127 unique issue types

2️⃣ Analyzing applications...
   ✓ Found 45 applications with issues

[... continued analysis ...]

📄 Generating PDF report: jira_analysis_report.pdf

============================================================
✅ Analysis Complete!
📊 PDF Report: jira_analysis_report.pdf
💡 Key Insights: 15
🗄️ SQLite database: support_tickets_analysis.db
============================================================ 