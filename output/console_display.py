"""
Console display module for batch experiments.

Handles all console output formatting including:
- Experiment configuration display
- Template performance summaries
- Key insights and rankings
- Final experiment summaries

Following SOLID principles:
- Single Responsibility: Handles console display only
- Dependency Inversion: Depends on data structures, not implementations

Extracted from BatchExperimentRunner to reduce complexity and improve maintainability.
"""

from collections import Counter
from typing import Dict, List, Any
import argparse

from core.timing_tracker import TimingTracker


class ConsoleDisplay:
    """Handles all console output and display formatting.
    
    Centralizes console display logic that was scattered across
    BatchExperimentRunner, reducing the main class by ~260 lines.
    """
    
    def __init__(self, timing_tracker: TimingTracker):
        """Initialize console display.
        
        Args:
            timing_tracker: TimingTracker instance for duration data
        """
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
    
    def print_experiment_configuration(
        self,
        args: argparse.Namespace,
        templates_to_use: List[str],
        problems_to_use: List[str]
    ) -> int:
        """Print experiment configuration.
        
        Args:
            args: Parsed command line arguments
            templates_to_use: List of templates selected
            problems_to_use: List of problems selected
            
        Returns:
            Total number of test combinations
        """
        print(f"\n📋 EXPERIMENT CONFIGURATION")
        print("─" * 50)
        print(f"  🔧 Method Module: {args.method}")
        print(f"  🤖 Model: {args.model}")
        print(f"  🔄 Iterations per test: {args.iterations}")
        
        print(f"\n📝 Templates to test ({len(templates_to_use)}):")
        for i, template in enumerate(templates_to_use):
            print(f"    {i+1}. {template}")
        
        print(f"\n🎯 Problems to test ({len(problems_to_use)}):")
        for i, problem in enumerate(problems_to_use):
            print(f"    {i+1}. {problem}")
        
        total_combinations = len(templates_to_use) * len(problems_to_use)
        print(f"\n🧮 Total test combinations: {total_combinations}")
        print("─" * 50)
        
        return total_combinations
    
    def print_template_performance_summary(
        self,
        template: str,
        template_results: List[Dict[str, Any]],
        formatted_duration: str
    ) -> None:
        """Print performance summary for a completed template.
        
        Args:
            template: Template name
            template_results: List of result dictionaries for this template
            formatted_duration: Formatted duration string
        """
        excellent_count = len([r for r in template_results if r['all_correct_rate'] >= 0.8])
        good_count = len([r for r in template_results if 0.5 <= r['all_correct_rate'] < 0.8])
        partial_count = len([r for r in template_results if r['all_correct_rate'] < 0.5 and r['at_least_one_correct_rate'] >= 0.5])
        poor_count = len([r for r in template_results if r['at_least_one_correct_rate'] < 0.5])
        
        avg_all_correct = sum(r['all_correct_rate'] for r in template_results) / len(template_results)
        avg_partial = sum(r['at_least_one_correct_rate'] for r in template_results) / len(template_results)
        
        print(f"\n📊 " + "─" * 60)
        print(f"📈 TEMPLATE PERFORMANCE SUMMARY: {template}")
        print("─" * 65)
        print(f"    🎯 Excellent (≥80%): {excellent_count:2d} problems")
        print(f"    ✅ Good (50-79%):   {good_count:2d} problems")
        print(f"    ⚠️  Partial (<50% all, ≥50% some): {partial_count:2d} problems")
        print(f"    ❌ Poor (<50% any): {poor_count:2d} problems")
        print(f"    📊 Average success: {avg_all_correct:.1%} all correct, {avg_partial:.1%} partial")
        print(f"    ⏱️  Total duration: {formatted_duration}")
        print("─" * 65)
    
    def print_key_insights(self, analysis: Dict[str, Any]) -> None:
        """Print key insights and rankings from experiment analysis.
        
        Args:
            analysis: Analysis dictionary from generate_ranking_analysis()
        """
        print(f"\n🏆 " + "=" * 70)
        print(f"🎯 KEY INSIGHTS & RANKINGS")
        print("=" * 75)
        
        # Best template overall
        best_template = analysis['template_ranking'][0]
        print(f"🥇 Best Overall Template: {best_template['template']}")
        print(f"   📊 Average success: {best_template['avg_all_correct_rate']:.1%} all correct")
        print(f"   🎯 Excellent on {best_template['excellent_problems']}/{best_template['total_problems']} problems")
        
        # Top 3 experiments
        print(f"\n🌟 Top 3 Performing Experiments:")
        for i, result in enumerate(analysis['experiment_ranking'][:3], 1):
            grade = "🎯" if result['all_correct_rate'] >= 0.8 else "✅" if result['all_correct_rate'] >= 0.6 else "⚠️"
            template_short = result['template'][:30] + "..." if len(result['template']) > 30 else result['template']
            print(f"   {grade} #{i}: {template_short} + {result['problem']} ({result['all_correct_rate']:.1%})")
        
        # Problem difficulty insights
        easy_problems = [p for p in analysis['problem_analysis'] if p['difficulty'] == 'EASY']
        hard_problems = [p for p in analysis['problem_analysis'] if p['difficulty'] in ['HARD', 'VERY_HARD']]
        
        if easy_problems:
            print(f"\n🟢 Easiest Problems: {', '.join([p['problem'] for p in easy_problems[:3]])}")
        if hard_problems:
            print(f"🔴 Hardest Problems: {', '.join([p['problem'] for p in hard_problems[:3]])}")
        
        print("=" * 75)
    
    def print_final_summary(
        self,
        total_duration: float,
        total_combinations: int,
        templates_to_use: List[str],
        problems_to_use: List[str],
        results_data: List[Dict[str, Any]],
        all_llm_responses: List[Any]
    ) -> None:
        """Print final experiment summary including performance breakdown and LLM statistics.
        
        Args:
            total_duration: Total experiment duration in seconds
            total_combinations: Total number of test combinations
            templates_to_use: List of templates used
            problems_to_use: List of problems used
            results_data: List of experiment result dictionaries
            all_llm_responses: List of all LLM response objects
        """
        print(f"\n🏆 " + "=" * 70)
        print("🎉 FINAL EXPERIMENT SUMMARY 🎉")
        print("=" * 75)
        print(f"⏰ Start Time:  {self.timing.format_global_start()}")
        print(f"🏁 End Time:    {self.timing.format_global_end()}")
        print(f"⏱️  Duration:    {self.format_duration(total_duration)}")
        print(f"🧮 Total tests: {total_combinations}")
        
        if results_data:
            successful_tests = len([r for r in results_data if r['all_correct_rate'] > 0])
            excellent_tests = len([r for r in results_data if r['all_correct_rate'] >= 0.8])
            good_tests = len([r for r in results_data if 0.5 <= r['all_correct_rate'] < 0.8])
            
            print(f"\n📊 PERFORMANCE BREAKDOWN:")
            print("─" * 40)
            print(f"🎯 Excellent (≥80%): {excellent_tests:2d} tests")
            print(f"✅ Good (50-79%):   {good_tests:2d} tests")
            print(f"⚠️  Some success:    {successful_tests - excellent_tests - good_tests:2d} tests")
            print(f"❌ No success:      {len(results_data) - successful_tests:2d} tests")
            print(f"📈 Overall success rate: {successful_tests / len(results_data):.1%}")
            print(f"⚡ Average time per test: {self.format_duration(total_duration / total_combinations)}")
            
            # Template timing breakdown
            print(f"\n🕐 TEMPLATE PERFORMANCE:")
            print("─" * 50)
            for template in templates_to_use:
                template_time = self.timing.get_template_duration(template)
                avg_per_problem = template_time / len(problems_to_use)
                template_results = [r for r in results_data if r['template'] == template]
                avg_success = sum(r['all_correct_rate'] for r in template_results) / len(template_results) if template_results else 0
                
                print(f"📝 {template[:40]:40s}")
                print(f"    ⏱️  {self.format_duration(template_time):>10s} (avg: {self.format_duration(avg_per_problem):>8s}/problem)")
                print(f"    📊 {avg_success:>9.1%} average success rate")
        
        # LLM usage statistics
        if all_llm_responses:
            print(f"\n🤖 LLM USAGE STATISTICS:")
            print("─" * 30)
            provider_counts = Counter([response.provider for response in all_llm_responses])
            for provider, count in provider_counts.items():
                print(f"    🔗 {provider}: {count} calls")
            
            token_usages = [response.usage.total_tokens for response in all_llm_responses]
            print(f"    🎯 Max tokens: {max(token_usages):,}")
            print(f"    📊 Median tokens: {sorted(token_usages)[len(token_usages)//2]:,}")
            print(f"    📈 Total tokens: {sum(token_usages):,}")
        
        print("=" * 75)
