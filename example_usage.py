"""
Example usage of the Jira Analysis Agent

This script demonstrates how to use the agent programmatically
for custom analysis workflows.
"""

import asyncio
from jira_analysis_agent import JiraAnalysisAgent
import os
import time


async def custom_analysis_example():
    """Example of running custom analyses"""
    
    # Initialize the agent
    agent = JiraAnalysisAgent()
    
    # Path to your CSV file
    csv_path = "sample_tickets.csv"  # Replace with your actual file
    
    if not os.path.exists(csv_path):
        print(f"❌ Please provide a valid CSV file path")
        return
    
    print("🔍 Running custom analysis...\n")
    
    # Example 1: Analyze specific questions
    questions = [
        "What are the top 5 applications with the longest resolution times?",
        "Which team members handle the most high-priority tickets?",
        "Show me the trend of bug reports vs feature requests over time",
        "What percentage of tickets are resolved within 24 hours?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        
        result = await agent.analyze(question, csv_path)
        print("\nAnswer:", result["response"])
        
        if result["insights"]:
            print("\nRelated Insights:")
            for insight in result["insights"]:
                print(f"• {insight}")
    
    # Example 2: Run comprehensive analysis with custom output
    print(f"\n\n{'='*60}")
    print("Running comprehensive analysis...")
    print('='*60)
    
    await agent.run_comprehensive_analysis(
        csv_path, 
        output_pdf="custom_report.pdf"
    )


async def sqlite_auto_example():
    """Example showing automatic SQLite usage for large files"""
    
    csv_path = "large_tickets.csv"  # Your large CSV file (>1GB)
    
    if not os.path.exists(csv_path):
        print(f"❌ This example requires a large CSV file (>1GB) at: {csv_path}")
        print("The agent will automatically use SQLite for better performance.")
        return
    
    # First run - will create SQLite database automatically
    print("📊 First run - Creating SQLite database...")
    start_time = time.time()
    
    agent = JiraAnalysisAgent(csv_path)
    await agent.run_comprehensive_analysis(csv_path, "first_run_report.pdf")
    
    first_run_time = time.time() - start_time
    print(f"\n⏱️ First run took: {first_run_time/60:.1f} minutes")
    
    # Second run - will use existing SQLite database (much faster!)
    print("\n📊 Second run - Using existing SQLite database...")
    start_time = time.time()
    
    agent2 = JiraAnalysisAgent(csv_path)
    await agent2.run_comprehensive_analysis(csv_path, "second_run_report.pdf")
    
    second_run_time = time.time() - start_time
    print(f"\n⏱️ Second run took: {second_run_time/60:.1f} minutes")
    print(f"🚀 Speed improvement: {first_run_time/second_run_time:.1f}x faster!")


async def sqlite_manual_example():
    """Example of manually controlling SQLite usage"""
    
    csv_path = "medium_tickets.csv"  # Any CSV file
    
    if not os.path.exists(csv_path):
        print(f"❌ Please provide a CSV file at: {csv_path}")
        return
    
    print("🔧 Manual SQLite control example\n")
    
    # Option 1: Force SQLite usage even for small files
    print("1. Forcing SQLite usage...")
    agent_with_sqlite = JiraAnalysisAgent(csv_path, use_sqlite=True)
    
    # The SQLite database will be created as: medium_tickets_analysis.db
    db_path = agent_with_sqlite._get_sqlite_path(csv_path)
    print(f"   SQLite database: {db_path}")
    
    # Option 2: Disable SQLite even for large files
    print("\n2. Disabling SQLite (direct CSV processing)...")
    agent_no_sqlite = JiraAnalysisAgent(csv_path, use_sqlite=False)
    
    # Option 3: Use custom SQLite path
    print("\n3. Using custom SQLite path...")
    custom_db = "my_custom_database.db"
    agent_custom = JiraAnalysisAgent(csv_path, use_sqlite=True, sqlite_path=custom_db)
    print(f"   Custom SQLite database: {custom_db}")


async def specific_analysis_example():
    """Example of running specific analyses only"""
    
    agent = JiraAnalysisAgent()
    csv_path = "sample_tickets.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Please provide a valid CSV file path")
        return
    
    print("📊 Running specific analyses...\n")
    
    # Load data info
    data_info = agent.load_data_info.invoke({"csv_path": csv_path})
    print(f"Data loaded: {data_info['estimated_rows']:,} rows")
    print(f"File size: {data_info['file_size_gb']} GB")
    print(f"SQLite status: {data_info['sqlite_status']}")
    
    # Run specific analyses
    print("\n1. Issue Types Analysis")
    issue_analysis = agent.analyze_issue_types.invoke({
        "csv_path": csv_path,
        "issue_type_column": "Issue Type"
    })
    print(f"   Found {issue_analysis['unique_issue_types']} unique issue types")
    print(f"   Top issue: {issue_analysis['top_issue_types'][0]['issue_type']} "
          f"({issue_analysis['top_issue_types'][0]['percentage']}%)")
    
    print("\n2. Resolution Times Analysis")
    resolution_analysis = agent.analyze_resolution_times.invoke({
        "csv_path": csv_path,
        "created_column": "Created",
        "resolved_column": "Resolved",
        "app_column": "Application",
        "priority_column": "Priority"
    })
    if resolution_analysis['overall_stats']:
        avg_hours = resolution_analysis['overall_stats']['mean_hours']
        print(f"   Average resolution: {avg_hours:.1f} hours ({avg_hours/24:.1f} days)")
    
    print("\n3. Bottlenecks Analysis")
    bottlenecks = agent.find_bottlenecks.invoke({
        "csv_path": csv_path,
        "status_column": "Status",
        "created_column": "Created",
        "updated_column": "Updated",
        "assignee_column": "Assignee"
    })
    print(f"   Stale tickets: {bottlenecks['stale_tickets']['total']:,}")
    
    # Generate insights based on analyses
    print("\n4. Generating Insights")
    insights = agent.generate_insights.invoke({
        "analysis_results": [issue_analysis, resolution_analysis, bottlenecks]
    })
    
    print("\n💡 Key Insights:")
    for i, insight in enumerate(insights[:5], 1):
        print(f"\n{i}. {insight}")


async def check_sqlite_files():
    """Utility to check for existing SQLite databases"""
    
    print("🔍 Checking for SQLite database files...\n")
    
    # Look for .db files in current directory
    db_files = [f for f in os.listdir('.') if f.endswith('_analysis.db')]
    
    if db_files:
        print(f"Found {len(db_files)} SQLite database(s):")
        for db_file in db_files:
            size_mb = os.path.getsize(db_file) / (1024**2)
            print(f"  • {db_file} ({size_mb:.1f} MB)")
            
            # Derive the original CSV name
            csv_name = db_file.replace('_analysis.db', '.csv')
            if os.path.exists(csv_name):
                csv_size_mb = os.path.getsize(csv_name) / (1024**2)
                print(f"    Original CSV: {csv_name} ({csv_size_mb:.1f} MB)")
    else:
        print("No SQLite databases found.")
        print("SQLite databases are created automatically for CSV files > 1GB")
        print("or when explicitly requested.")


if __name__ == "__main__":
    print("Jira Analysis Agent - Example Usage\n")
    print("Choose an example to run:")
    print("1. Custom analysis with specific questions")
    print("2. SQLite auto-usage demo (requires large file)")
    print("3. Manual SQLite control examples")
    print("4. Specific analyses only")
    print("5. Full comprehensive analysis")
    print("6. Check for existing SQLite databases")
    
    choice = input("\nEnter your choice (1-6): ")
    
    if choice == "1":
        asyncio.run(custom_analysis_example())
    elif choice == "2":
        asyncio.run(sqlite_auto_example())
    elif choice == "3":
        asyncio.run(sqlite_manual_example())
    elif choice == "4":
        asyncio.run(specific_analysis_example())
    elif choice == "5":
        csv_path = input("Enter CSV file path: ")
        agent = JiraAnalysisAgent(csv_path)
        asyncio.run(agent.run_comprehensive_analysis(csv_path))
    elif choice == "6":
        asyncio.run(check_sqlite_files())
    else:
        print("Invalid choice. Please run the script again.") 