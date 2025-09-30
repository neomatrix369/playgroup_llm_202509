"""
Output generation module for batch experiments.

Handles all output format generation including:
- Console tables
- CSV files
- HTML reports
- Ranking analysis
- Summary logs
- Persistent summaries

Following SOLID principles:
- Single Responsibility: Handles output generation only
- Open/Closed: Extensible for new output formats
- Dependency Inversion: Depends on abstractions (data structures)

Extracted from BatchExperimentRunner to reduce complexity and improve maintainability.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

try:
    import pandas as pd
except ImportError:
    pd = None

from analysis.performance_grader import PerformanceGrader
from analysis.difficulty_classifier import DifficultyClassifier
from analysis.statistics_aggregator import (
    TemplateStatisticsAggregator,
    ProblemStatisticsAggregator,
    ExperimentStatisticsAggregator,
    BestTemplateRecommender
)
from output.report_writer import ReportWriter
from core.timing_tracker import TimingTracker


class OutputGenerator:
    """Generates all output formats from experiment results.
    
    Centralizes output generation logic that was scattered across
    BatchExperimentRunner, reducing the main class by ~400 lines.
    """
    
    def __init__(self, results_data: List[Dict[str, Any]], timing_tracker: TimingTracker):
        """Initialize output generator.
        
        Args:
            results_data: List of experiment result dictionaries
            timing_tracker: TimingTracker instance for duration data
        """
        self.results_data = results_data
        self.timing = timing_tracker
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string (e.g., "1h 23m 45s")
        """
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def generate_console_table(self) -> None:
        """Generate formatted console table."""
        if not self.results_data:
            return
        
        print(f"\n📋 " + "═" * 90)
        print(f"📊 DETAILED RESULTS TABLE 📊")
        print("═" * 95)
        
        # Header with better formatting
        print(f"{'📝 Template':<40} {'🎲 Problem':<12} {'🎯 All%':<8} {'⚠️ Part%':<9} {'⏱️ Time':<10} {'📊 Grade':<8}")
        print("─" * 95)
        
        # Data rows with visual indicators
        grader = PerformanceGrader()
        for result in self.results_data:
            template_short = result['template'][:35] + "..." if len(result['template']) > 35 else result['template']
            test_time = self.format_duration(result['individual_duration'])
            
            grade = grader.grade_with_icon(result['all_correct_rate'])
            
            print(f"{template_short:<40} {result['problem']:<12} {result['all_correct_rate']:<7.1%} "
                  f"{result['at_least_one_correct_rate']:<8.1%} {test_time:<10} {grade:<8}")
        
        print("─" * 95)
        print(f"Legend: 🎯A(≥80%) ✅B(60-79%) ⚠️C(40-59%) 🔶D(partial) ❌F(<40%)")
        print("═" * 95)
    
    def generate_csv_output(self, output_dir: Path) -> str:
        """Generate CSV output file with ranking data.
        
        Args:
            output_dir: Directory to save CSV files
            
        Returns:
            Path to generated CSV file, or empty string if failed
        """
        if not self.results_data:
            return ""
        
        if pd is None:
            print("Warning: pandas not available, skipping CSV output")
            return ""
        
        # Main results data
        df = pd.DataFrame(self.results_data)
        
        # Add timing columns
        df['individual_duration_formatted'] = df['individual_duration'].apply(self.format_duration)
        df['problem_duration_formatted'] = df['problem_duration'].apply(self.format_duration)
        
        # Add ranking information
        analysis = self.generate_ranking_analysis()
        if analysis:
            # Add experiment rank
            experiment_rankings = {f"{r['template']}|{r['problem']}": i+1
                                 for i, r in enumerate(analysis['experiment_ranking'])}
            df['experiment_rank'] = df.apply(lambda row: experiment_rankings.get(f"{row['template']}|{row['problem']}", 0), axis=1)
            
            # Add template rank
            template_rankings = {r['template']: i+1 for i, r in enumerate(analysis['template_ranking'])}
            df['template_rank'] = df['template'].map(template_rankings)
            
            # Add problem difficulty
            problem_difficulties = {r['problem']: r['difficulty'] for r in analysis['problem_analysis']}
            df['problem_difficulty'] = df['problem'].map(problem_difficulties)
            
            # Add grade based on performance
            grader = PerformanceGrader()
            df['performance_grade'] = df['all_correct_rate'].apply(grader.grade)
        
        # Use the output directory name as timestamp for consistency
        timestamp = output_dir.name
        csv_file = output_dir / f"batch_results_{timestamp}.csv"
        
        # Sort by experiment rank for better readability
        if 'experiment_rank' in df.columns:
            df = df.sort_values('experiment_rank')
        
        df.to_csv(csv_file, index=False)
        
        # Also generate separate ranking files
        if analysis:
            # Template ranking CSV
            template_df = pd.DataFrame(analysis['template_ranking'])
            template_csv = output_dir / f"template_ranking_{timestamp}.csv"
            template_df.to_csv(template_csv, index=False)
            
            # Problem analysis CSV
            problem_df = pd.DataFrame(analysis['problem_analysis'])
            problem_csv = output_dir / f"problem_analysis_{timestamp}.csv"
            problem_df.to_csv(problem_csv, index=False)
        
        return str(csv_file)
    
    def generate_html_output(self, output_dir: Path) -> str:
        """Generate HTML output file.
        
        Args:
            output_dir: Directory to save HTML file
            
        Returns:
            Path to generated HTML file, or empty string if failed
        """
        if not self.results_data:
            return ""
        
        # Use the output directory name as timestamp for consistency
        timestamp = output_dir.name
        html_file = output_dir / f"batch_results_{timestamp}.html"
        
        total_duration = self.timing.get_global_duration()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ARC-AGI Batch Experiment Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .success {{ background-color: #d4edda; }}
        .partial {{ background-color: #fff3cd; }}
        .failed {{ background-color: #f8d7da; }}
        .timing {{ font-family: monospace; font-size: 0.9em; }}
        .summary {{ background-color: #e9ecef; padding: 15px; margin: 20px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>ARC-AGI Batch Experiment Results</h1>

    <div class="summary">
        <h2>Experiment Summary</h2>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total Duration:</strong> {self.format_duration(total_duration)}</p>
        <p><strong>Total Tests:</strong> {len(self.results_data)}</p>
        <p><strong>Average Success Rate:</strong> {sum(r['all_correct_rate'] for r in self.results_data) / len(self.results_data):.2%}</p>
    </div>

    <table>
        <tr>
            <th>Template</th>
            <th>Problem</th>
            <th>All Correct Rate</th>
            <th>At Least One Correct</th>
            <th>Test Duration</th>
            <th>Branch Duration</th>
        </tr>
"""
        
        for result in self.results_data:
            # Determine row class based on success rate
            row_class = ""
            if result['all_correct_rate'] >= 0.8:
                row_class = "success"
            elif result['all_correct_rate'] >= 0.3:
                row_class = "partial"
            else:
                row_class = "failed"
            
            test_time = self.format_duration(result['individual_duration'])
            branch_time = self.format_duration(result['problem_duration'])
            
            html_content += f"""
        <tr class="{row_class}">
            <td>{result['template']}</td>
            <td>{result['problem']}</td>
            <td>{result['all_correct_rate']:.2%}</td>
            <td>{result['at_least_one_correct_rate']:.2%}</td>
            <td class="timing">{test_time}</td>
            <td class="timing">{branch_time}</td>
        </tr>"""
        
        html_content += """
    </table>
</body>
</html>
"""
        
        html_file.write_text(html_content)
        return str(html_file)
    
    def generate_ranking_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive ranking and best-performance analysis.
        
        Returns:
            Dictionary containing experiment ranking, template ranking,
            problem analysis, and best template recommendations
        """
        if not self.results_data:
            return {}
        
        # 1. Experiment ranking (template+problem combinations)
        experiment_ranking = sorted(
            self.results_data,
            key=lambda x: (x['all_correct_rate'], x['at_least_one_correct_rate']),
            reverse=True
        )
        
        # 2. Template ranking (average performance across all problems)
        template_aggregator = TemplateStatisticsAggregator(self.results_data)
        template_ranking = template_aggregator.aggregate_to_ranking()
        
        # 3. Problem difficulty analysis (average performance across all templates)
        problem_aggregator = ProblemStatisticsAggregator(self.results_data)
        problem_analysis = problem_aggregator.aggregate_to_analysis()
        
        # 4. Best template recommendations per problem
        recommender = BestTemplateRecommender(self.results_data)
        best_template_per_problem = recommender.recommend_all()
        
        return {
            'experiment_ranking': experiment_ranking,
            'template_ranking': template_ranking,
            'problem_analysis': problem_analysis,
            'best_template_per_problem': best_template_per_problem
        }
    
    def generate_ranking_report(self, output_dir: Path, timestamp: str) -> str:
        """Generate comprehensive ranking and analysis report.
        
        Args:
            output_dir: Directory to save report
            timestamp: Timestamp string for filename
            
        Returns:
            Path to generated report file, or empty string if failed
        """
        if not self.results_data:
            return ""
        
        analysis = self.generate_ranking_analysis()
        if not analysis:
            return ""
        
        ranking_file = output_dir / f"ranking_analysis_{timestamp}.log"
        
        with ReportWriter.open(ranking_file) as writer:
            writer.section_header("🏆  COMPREHENSIVE RANKING & PERFORMANCE ANALYSIS  🏆")
            writer.blank_line()
            
            # 1. Top performing experiments
            writer.subsection_header("🥇 TOP PERFORMING EXPERIMENTS (Template + Problem Combinations):")
            for i, result in enumerate(analysis['experiment_ranking'][:10], 1):
                grade = "🎯" if result['all_correct_rate'] >= 0.8 else "✅" if result['all_correct_rate'] >= 0.6 else "⚠️"
                writer.writeln(f"{grade} #{i:2d}: {result['template'][:35]:35s} + {result['problem']}")
                writer.writeln(f"      📊 {result['all_correct_rate']:6.1%} all correct, {result['at_least_one_correct_rate']:6.1%} partial")
                writer.writeln(f"      ⏱️  {self.format_duration(result['individual_duration'])}\n")
            
            # 2. Template ranking
            writer.section_separator()
            writer.subsection_header("📝 TEMPLATE RANKING (Overall Performance Across All Problems):")
            for i, template_data in enumerate(analysis['template_ranking'], 1):
                grade = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                writer.writeln(f"{grade} #{i}: {template_data['template']}")
                writer.writeln(f"    📊 Average: {template_data['avg_all_correct_rate']:6.1%} all correct, {template_data['avg_partial_rate']:6.1%} partial")
                writer.writeln(f"    🎯 Excellent problems: {template_data['excellent_problems']}/{template_data['total_problems']} ({template_data['excellent_problems']/template_data['total_problems']:.1%})")
                writer.writeln(f"    ✅ Good problems: {template_data['good_problems']}/{template_data['total_problems']} ({template_data['good_problems']/template_data['total_problems']:.1%})")
                writer.writeln(f"    ⏱️  Average duration: {self.format_duration(template_data['avg_duration'])}")
                writer.writeln(f"    📈 Weighted score: {template_data['score']:.3f}\n")
            
            # 3. Problem difficulty analysis
            writer.section_separator()
            writer.subsection_header("🎯 PROBLEM DIFFICULTY ANALYSIS:")
            
            difficulty_groups = {}
            for problem_data in analysis['problem_analysis']:
                difficulty = problem_data['difficulty']
                if difficulty not in difficulty_groups:
                    difficulty_groups[difficulty] = []
                difficulty_groups[difficulty].append(problem_data)
            
            for difficulty in ['EASY', 'MEDIUM', 'HARD', 'VERY_HARD']:
                if difficulty in difficulty_groups:
                    problems = difficulty_groups[difficulty]
                    writer.writeln(f"🌟 {difficulty} Problems ({len(problems)}):")
                    for problem_data in problems:
                        difficulty_icon = "🟢" if difficulty == "EASY" else "🟡" if difficulty == "MEDIUM" else "🟠" if difficulty == "HARD" else "🔴"
                        writer.writeln(f"  {difficulty_icon} {problem_data['problem']}: {problem_data['avg_all_correct_rate']:6.1%} avg success")
                        writer.writeln(f"     🏆 Best template: {problem_data['best_template'][:40]:40s} ({problem_data['best_score']:6.1%})")
                    writer.blank_line()
            
            # 4. Best template recommendations
            writer.section_separator()
            writer.subsection_header("🎯 OPTIMAL TEMPLATE RECOMMENDATIONS PER PROBLEM:")
            for problem, recommendation in analysis['best_template_per_problem'].items():
                writer.writeln(f"🎲 Problem: {problem}")
                writer.writeln(f"   🏆 Best: {recommendation['best_template'][:50]:50s} ({recommendation['best_score']:6.1%})")
                if recommendation['alternatives']:
                    writer.writeln("   📋 Alternatives:")
                    for alt in recommendation['alternatives']:
                        writer.writeln(f"      • {alt['template'][:45]:45s} ({alt['all_correct_rate']:6.1%})")
                writer.blank_line()
            
            # 5. Comparative analysis summary
            writer.section_separator()
            writer.subsection_header("📊 COMPARATIVE ANALYSIS SUMMARY:")
            
            best_template = analysis['template_ranking'][0]
            worst_template = analysis['template_ranking'][-1]
            easiest_problem = analysis['problem_analysis'][0]
            hardest_problem = analysis['problem_analysis'][-1]
            
            writer.writeln(f"🏆 Best Overall Template: {best_template['template']}")
            writer.writeln(f"   📊 Average success rate: {best_template['avg_all_correct_rate']:.1%}")
            writer.writeln(f"   🎯 Excellent on {best_template['excellent_problems']}/{best_template['total_problems']} problems\n")
            
            writer.writeln(f"⚠️  Most Challenging Template: {worst_template['template']}")
            writer.writeln(f"   📊 Average success rate: {worst_template['avg_all_correct_rate']:.1%}\n")
            
            writer.writeln(f"🟢 Easiest Problem: {easiest_problem['problem']}")
            writer.writeln(f"   📊 Average success rate: {easiest_problem['avg_all_correct_rate']:.1%}")
            writer.writeln(f"   🏆 Best template: {easiest_problem['best_template']}\n")
            
            writer.writeln(f"🔴 Hardest Problem: {hardest_problem['problem']}")
            writer.writeln(f"   📊 Average success rate: {hardest_problem['avg_all_correct_rate']:.1%}")
            writer.writeln(f"   🏆 Best template: {hardest_problem['best_template']}\n")
            
            writer.section_separator()
        
        return str(ranking_file)
    
    def generate_persistent_summary(self, base_output_dir: Path, aggregated_data: Dict[str, Any]) -> List[str]:
        """Generate persistent summary files that aggregate all experiments.
        
        Args:
            base_output_dir: Base directory for summary files
            aggregated_data: Aggregated experiment data dictionary
            
        Returns:
            List of paths to generated files
        """
        if not aggregated_data:
            return []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_files = []
        
        # 1. Overall experiment summary
        summary_file = base_output_dir / "experiment_summary_latest.log"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("📊  COMPREHENSIVE EXPERIMENT SUMMARY (ALL RUNS)  📊\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"🧮 Total experiments analyzed: {aggregated_data['total_experiments']}\n")
            f.write(f"📊 Total individual results: {aggregated_data['total_results']}\n\n")
            
            # Recent experiments
            f.write("🕒 RECENT EXPERIMENTS:\n")
            f.write("─" * 40 + "\n")
            for exp in aggregated_data['experiment_metadata'][:5]:
                date_str = datetime.fromtimestamp(exp['date']).strftime('%Y-%m-%d %H:%M')
                f.write(f"  📁 {exp['timestamp']}: {exp['result_count']} results ({date_str})\n")
            
            # Template performance
            if aggregated_data.get('analysis', {}).get('template_ranking'):
                f.write(f"\n🏆 TOP TEMPLATES (All-Time Performance):\n")
                f.write("─" * 50 + "\n")
                for i, template_data in enumerate(aggregated_data['analysis']['template_ranking'][:5], 1):
                    f.write(f"  #{i}: {template_data['template'][:45]:45s}\n")
                    f.write(f"      📊 Avg: {template_data['avg_all_correct_rate']:6.1%} | Score: {template_data['score']:.3f}\n")
            
            # Problem insights
            if aggregated_data.get('analysis', {}).get('problem_analysis'):
                f.write(f"\n🎯 PROBLEM DIFFICULTY INSIGHTS:\n")
                f.write("─" * 40 + "\n")
                easy_problems = [p for p in aggregated_data['analysis']['problem_analysis'] if p['difficulty'] == 'EASY']
                hard_problems = [p for p in aggregated_data['analysis']['problem_analysis'] if p['difficulty'] in ['HARD', 'VERY_HARD']]
                
                if easy_problems:
                    f.write(f"  🟢 Easiest: {', '.join([p['problem'] for p in easy_problems[:3]])}\n")
                if hard_problems:
                    f.write(f"  🔴 Hardest: {', '.join([p['problem'] for p in hard_problems[:3]])}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        generated_files.append(str(summary_file))
        
        # 2. Template performance trends (CSV)
        if pd is not None and aggregated_data['template_stats']:
            template_trends_data = []
            for template, stats in aggregated_data['template_stats'].items():
                template_trends_data.append({
                    'template': template,
                    'total_experiments': stats['experiments'],
                    'avg_success_rate': stats['avg_success_rate'],
                    'max_success_rate': stats['max_success_rate'],
                    'avg_duration': stats['avg_duration'],
                    'consistency_score': 1.0 - (max(stats['success_rates']) - min(stats['success_rates']))
                })
            
            template_df = pd.DataFrame(template_trends_data)
            template_df = template_df.sort_values('avg_success_rate', ascending=False)
            template_trends_file = base_output_dir / "template_performance_trends.csv"
            template_df.to_csv(template_trends_file, index=False)
            generated_files.append(str(template_trends_file))
        
        # 3. Problem performance analysis (CSV)
        if pd is not None and aggregated_data['problem_stats']:
            problem_trends_data = []
            classifier = DifficultyClassifier()
            for problem, stats in aggregated_data['problem_stats'].items():
                difficulty = classifier.classify_simple(stats['avg_success_rate'])
                problem_trends_data.append({
                    'problem': problem,
                    'total_experiments': stats['experiments'],
                    'avg_success_rate': stats['avg_success_rate'],
                    'max_success_rate': stats['max_success_rate'],
                    'templates_tested': len(stats['templates_used']),
                    'difficulty_rating': difficulty
                })
            
            problem_df = pd.DataFrame(problem_trends_data)
            problem_df = problem_df.sort_values('avg_success_rate', ascending=False)
            problem_trends_file = base_output_dir / "problem_difficulty_trends.csv"
            problem_df.to_csv(problem_trends_file, index=False)
            generated_files.append(str(problem_trends_file))
        
        # 4. Best combinations lookup table
        if aggregated_data.get('analysis', {}).get('best_template_per_problem'):
            lookup_file = base_output_dir / "best_template_lookup.json"
            
            lookup_data = {
                'generated': datetime.now().isoformat(),
                'recommendations': {}
            }
            
            for problem, recommendation in aggregated_data['analysis']['best_template_per_problem'].items():
                lookup_data['recommendations'][problem] = {
                    'best_template': recommendation['best_template'],
                    'success_rate': recommendation['best_score'],
                    'alternatives': [
                        {
                            'template': alt['template'],
                            'success_rate': alt['all_correct_rate']
                        }
                        for alt in recommendation['alternatives'][:3]
                    ]
                }
            
            with open(lookup_file, 'w') as f:
                json.dump(lookup_data, f, indent=2)
            generated_files.append(str(lookup_file))
        
        return generated_files
    
    def generate_summary_log(
        self,
        output_dir: Path,
        timestamp: str,
        total_duration: float,
        templates_to_use: List[str],
        problems_to_use: List[str],
        total_combinations: int
    ) -> str:
        """Generate detailed summary log file.
        
        Args:
            output_dir: Directory to save log
            timestamp: Timestamp string for filename
            total_duration: Total experiment duration in seconds
            templates_to_use: List of templates used
            problems_to_use: List of problems used
            total_combinations: Total number of test combinations
            
        Returns:
            Path to generated log file, or empty string if failed
        """
        if not self.results_data:
            return ""
        
        summary_log = output_dir / f"batch_summary_{timestamp}.log"
        
        with open(summary_log, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("🧠  ARC-AGI BATCH EXPERIMENT SUMMARY  🧠\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"📅 Experiment started: {self.timing.format_global_start()}\n\n")
            f.write("📋 CONFIGURATION:\n")
            f.write("─" * 40 + "\n")
            f.write(f"  🔧 Method: method1_text_prompt\n")
            f.write(f"  🤖 Model: openrouter/deepseek/deepseek-chat-v3-0324\n")
            f.write(f"  🔄 Iterations: 1\n\n")
            f.write(f"📝 Templates ({len(templates_to_use)}):\n")
            for i, template in enumerate(templates_to_use, 1):
                f.write(f"  {i}. {template}\n")
            f.write(f"\n🎯 Problems ({len(problems_to_use)}):\n")
            for i, problem in enumerate(problems_to_use, 1):
                f.write(f"  {i}. {problem}\n")
            f.write(f"\n🧮 Total combinations: {total_combinations}\n")
            f.write("=" * 80 + "\n\n")
            f.write("📊 INDIVIDUAL TEST RESULTS:\n")
            f.write("─" * 40 + "\n")
            
            grader = PerformanceGrader()
            for i, result in enumerate(self.results_data, 1):
                success_icon = grader.success_icon(result['all_correct_rate'], result['at_least_one_correct_rate'])
                f.write(f"{success_icon} Test {i:02d}: {result['template']} + {result['problem']}\n")
                f.write(f"    📊 Results: {result['all_correct']}/{result['total_runs']} all correct ({result['all_correct_rate']:.1%}), ")
                f.write(f"{result['at_least_one_correct']}/{result['total_runs']} partial ({result['at_least_one_correct_rate']:.1%})\n")
                f.write(f"    ⏱️  Duration: {self.format_duration(result['individual_duration'])}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("🎉  EXPERIMENT COMPLETED  🎉\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"📅 Completed at: {self.timing.format_global_end()}\n")
            f.write(f"⏱️  Total duration: {self.format_duration(total_duration)}\n\n")
            
            if self.results_data:
                successful_tests = len([r for r in self.results_data if r['all_correct_rate'] > 0])
                f.write("📊 FINAL RESULTS SUMMARY:\n")
                f.write("─" * 40 + "\n")
                f.write(f"🎯 Tests with success: {successful_tests}/{len(self.results_data)} ({successful_tests / len(self.results_data):.1%})\n")
                f.write(f"⏱️  Average time per test: {self.format_duration(total_duration / total_combinations)}\n\n")
                
                # Template timing breakdown
                f.write("🕐 TEMPLATE TIMING BREAKDOWN:\n")
                f.write("─" * 40 + "\n")
                for template in templates_to_use:
                    template_time = self.timing.get_template_duration(template)
                    f.write(f"  📝 {template}: {self.format_duration(template_time)}\n")
            f.write("\n" + "=" * 80 + "\n")
        
        return str(summary_log)
