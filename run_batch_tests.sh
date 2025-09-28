#!/usr/bin/env zsh

# Script compatibility: Works with both zsh and bash 4+
# Detect shell and enable required features
if [[ -n "${ZSH_VERSION:-}" ]]; then
    # Running in zsh - enable bash-like associative arrays
    setopt KSH_ARRAYS  # Use 0-based indexing like bash
    setopt SH_WORD_SPLIT  # Enable word splitting like bash
elif [[ -n "${BASH_VERSION:-}" ]]; then
    # Running in bash - check version
    if [ "${BASH_VERSION%%.*}" -lt 4 ]; then
        echo "Error: This script requires bash version 4.0 or higher for associative arrays"
        echo "Current version: $BASH_VERSION"
        echo "On macOS, install with: brew install bash"
        exit 1
    fi
else
    echo "Error: This script requires either zsh or bash 4.0+"
    exit 1
fi

set -e
set -u
set -o pipefail

# Batch testing script for ARC-AGI method1_text_prompt.py
# Iterates through J2 template files and problem IDs

# Configuration arrays
declare -a J2_TEMPLATES=(
    # "baseline_wplaingrid_spelke.j2"
    # "baseline_justjson_spelke.j2"
    # "reflexion_spelke.j2"
    "baseline_wplaingrid_enhanced.j2"
    "baseline_wquotedgridcsv_excel_enhanced.j2"
    "baseline_justjson_enhanced.j2"
    "reflexion_enhanced.j2"
)

declare -a PROBLEM_IDS=(
    "0d3d703e"
    "08ed6ac7"
    "9565186b"
    "178fcbfb"
    "0a938d79"
    "1a07d186"
)

# Default configuration
DEFAULT_ITERATIONS=1
DEFAULT_PYTHON_SCRIPT="method1_text_prompt.py"
OUTPUT_DIR="batch_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Command line options
ITERATIONS=$DEFAULT_ITERATIONS
PYTHON_SCRIPT=$DEFAULT_PYTHON_SCRIPT
DRY_RUN=false
VERBOSE=false
SELECTED_TEMPLATES=()
SELECTED_PROBLEMS=()

# Results tracking for table generation
declare -A TEST_RESULTS
declare -a RESULT_ORDER

# Timing and tracking variables
GLOBAL_START_TIME=""
GLOBAL_END_TIME=""
declare -A TEMPLATE_TIMINGS
declare -A PROBLEM_TIMINGS
declare -A INDIVIDUAL_TIMINGS

# Help function
show_help() {
    local script_name
    if [[ -n "${ZSH_VERSION:-}" ]]; then
        script_name="${0:t}"  # zsh: get basename
    else
        script_name="$(basename "$0")"  # bash: get basename
    fi

    cat << EOF
Usage: $script_name [OPTIONS]

Batch testing script for ARC-AGI method1_text_prompt.py

OPTIONS:
    -i, --iterations NUM     Number of iterations per test (default: $DEFAULT_ITERATIONS)
    -s, --script SCRIPT      Python script to run (default: $DEFAULT_PYTHON_SCRIPT)
    -t, --templates LIST     Comma-separated list of template indices or names
    -p, --problems LIST      Comma-separated list of problem indices or IDs
    -o, --output DIR         Output directory (default: $OUTPUT_DIR)
    -d, --dry-run            Show commands without executing
    -v, --verbose            Verbose output
    -h, --help               Show this help

EXAMPLES:
    # Run all combinations
    $script_name

    # Run with 5 iterations, dry-run mode
    $script_name -i 5 -d

    # Run specific templates (by index or name)
    $script_name -t "0,2,baseline_wplaingrid_enhanced.j2"

    # Run specific problems (by index or ID)
    $script_name -p "0,1,0d3d703e"

    # Verbose mode with custom output directory
    $script_name -v -o "results_\$(date +%Y%m%d)"

AVAILABLE TEMPLATES:
EOF
    # List templates with indices (compatible with both bash and zsh)
    local i=0
    for template in "${J2_TEMPLATES[@]}"; do
        echo "    $i: $template"
        i=$((i + 1))
    done

    echo ""
    echo "AVAILABLE PROBLEMS:"
    # List problems with indices
    i=0
    for problem in "${PROBLEM_IDS[@]}"; do
        echo "    $i: $problem"
        i=$((i + 1))
    done
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -s|--script)
            PYTHON_SCRIPT="$2"
            shift 2
            ;;
        -t|--templates)
            # Split comma-separated values (compatible with both bash and zsh)
            SELECTED_TEMPLATES=()
            local temp_ifs="$IFS"
            IFS=','
            for item in $2; do
                SELECTED_TEMPLATES+=("$item")
            done
            IFS="$temp_ifs"
            shift 2
            ;;
        -p|--problems)
            # Split comma-separated values (compatible with both bash and zsh)
            SELECTED_PROBLEMS=()
            local temp_ifs="$IFS"
            IFS=','
            for item in $2; do
                SELECTED_PROBLEMS+=("$item")
            done
            IFS="$temp_ifs"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Function to resolve template selection
resolve_templates() {
    local result=()

    if [[ ${#SELECTED_TEMPLATES[@]} -eq 0 ]]; then
        # Use all templates
        result=("${J2_TEMPLATES[@]}")
    else
        # Process selected templates
        for selection in "${SELECTED_TEMPLATES[@]}"; do
            if [[ "$selection" =~ ^[0-9]+$ ]]; then
                # Index selection (0-based with KSH_ARRAYS option)
                if [[ $selection -lt ${#J2_TEMPLATES[@]} ]]; then
                    result+=("${J2_TEMPLATES[$selection]}")
                else
                    echo "Warning: Template index $selection out of range"
                fi
            else
                # Name selection - check if it exists in array
                local found=false
                for template in "${J2_TEMPLATES[@]}"; do
                    if [[ "$template" == "$selection" ]]; then
                        result+=("$template")
                        found=true
                        break
                    fi
                done
                if [[ "$found" == false ]]; then
                    echo "Warning: Template '$selection' not found in available templates"
                fi
            fi
        done
    fi

    echo "${result[@]}"
}

# Function to resolve problem selection
resolve_problems() {
    local result=()

    if [[ ${#SELECTED_PROBLEMS[@]} -eq 0 ]]; then
        # Use all problems
        result=("${PROBLEM_IDS[@]}")
    else
        # Process selected problems
        for selection in "${SELECTED_PROBLEMS[@]}"; do
            if [[ "$selection" =~ ^[0-9]+$ ]]; then
                # Index selection (0-based with KSH_ARRAYS option)
                if [[ $selection -lt ${#PROBLEM_IDS[@]} ]]; then
                    result+=("${PROBLEM_IDS[$selection]}")
                else
                    echo "Warning: Problem index $selection out of range"
                fi
            else
                # ID selection - check if it exists in array
                local found=false
                for problem in "${PROBLEM_IDS[@]}"; do
                    if [[ "$problem" == "$selection" ]]; then
                        result+=("$problem")
                        found=true
                        break
                    fi
                done
                if [[ "$found" == false ]]; then
                    echo "Warning: Problem ID '$selection' not found in available problems"
                fi
            fi
        done
    fi

    echo "${result[@]}"
}

# Function to format elapsed time
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))

    if [[ $hours -gt 0 ]]; then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [[ $minutes -gt 0 ]]; then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}

# Function to log timestamp with message
log_timestamp() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message"
}

# Function to parse test results from log file
parse_test_results() {
    local log_file="$1"
    local template="$2"
    local problem="$3"

    if [[ ! -f "$log_file" ]]; then
        echo "UNKNOWN|0|0|N/A|N/A"
        return
    fi

    # Extract key metrics from log file
    local status="UNKNOWN"
    local score=0
    local correctness=0
    local duration="N/A"
    local error_msg="N/A"

    # Check if test completed successfully
    if grep -q "Completed successfully" "$log_file"; then
        status="SUCCESS"
    elif grep -q "Failed with exit code" "$log_file"; then
        status="FAILED"
        error_msg=$(grep "Failed with exit code" "$log_file" | tail -1 | sed 's/.*Failed with exit code: /Exit /')
    fi

    # Extract score/correctness if available (adjust patterns based on your script output)
    if grep -q "Correctness:" "$log_file"; then
        correctness=$(grep "Correctness:" "$log_file" | tail -1 | sed 's/.*Correctness: *\([0-9.]*\).*/\1/')
    elif grep -q "correct.*%" "$log_file"; then
        correctness=$(grep "correct.*%" "$log_file" | tail -1 | sed 's/.*\([0-9.]*\)%.*/\1/')
    fi

    # Extract timing if available
    if grep -q "Duration:" "$log_file"; then
        duration=$(grep "Duration:" "$log_file" | tail -1 | sed 's/.*Duration: *\([0-9.]*[ms]*\).*/\1/')
    elif grep -q "took.*seconds" "$log_file"; then
        duration=$(grep "took.*seconds" "$log_file" | tail -1 | sed 's/.*took *\([0-9.]*\) seconds.*/\1s/')
    fi

    # Extract score if available
    if grep -q "Score:" "$log_file"; then
        score=$(grep "Score:" "$log_file" | tail -1 | sed 's/.*Score: *\([0-9.]*\).*/\1/')
    fi

    echo "${status}|${score}|${correctness}|${duration}|${error_msg}"
}

# Function to execute or display command
execute_command() {
    local cmd="$1"
    local log_file="$2"
    local template="$3"
    local problem="$4"

    log_timestamp "Starting individual test: $template + $problem"

    if [[ "$VERBOSE" == true ]]; then
        echo "Executing: $cmd"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $cmd"
        # Store dummy result for dry run
        TEST_RESULTS["${template}|${problem}"]="DRY_RUN|0|0|N/A|Dry run mode"
        RESULT_ORDER+=("${template}|${problem}")
        INDIVIDUAL_TIMINGS["${template}|${problem}"]="0"
        return 0
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting: $cmd" >> "$log_file"

    local start_time=$(date +%s)

    # Execute command and capture both stdout and stderr
    if eval "$cmd" >> "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local formatted_duration=$(format_duration $duration)

        echo "$(date '+%Y-%m-%d %H:%M:%S') - Completed successfully in ${duration}s" >> "$log_file"
        log_timestamp "Individual test completed successfully in $formatted_duration"

        # Store timing
        INDIVIDUAL_TIMINGS["${template}|${problem}"]="$duration"

        # Parse and store results
        local results=$(parse_test_results "$log_file" "$template" "$problem")
        TEST_RESULTS["${template}|${problem}"]="$results"
        RESULT_ORDER+=("${template}|${problem}")

        return 0
    else
        local exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local formatted_duration=$(format_duration $duration)

        echo "$(date '+%Y-%m-%d %H:%M:%S') - Failed with exit code: $exit_code in ${duration}s" >> "$log_file"
        log_timestamp "Individual test failed (exit code: $exit_code) after $formatted_duration"

        # Store timing
        INDIVIDUAL_TIMINGS["${template}|${problem}"]="$duration"

        # Parse and store failure results
        local results=$(parse_test_results "$log_file" "$template" "$problem")
        TEST_RESULTS["${template}|${problem}"]="$results"
        RESULT_ORDER+=("${template}|${problem}")

        return $exit_code
    fi
}

# Main execution
main() {
    # Record global start time
    GLOBAL_START_TIME=$(date +%s)
    local start_timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "=== ARC-AGI Batch Testing Script ==="
    log_timestamp "EXPERIMENT STARTED"
    echo "Timestamp: $TIMESTAMP"
    echo "Iterations: $ITERATIONS"
    echo "Python Script: $PYTHON_SCRIPT"
    echo "Output Directory: $OUTPUT_DIR"
    echo "Dry Run: $DRY_RUN"
    echo ""

    # Resolve selections
    log_timestamp "Resolving template and problem selections..."
    local templates_to_use=($(resolve_templates))
    local problems_to_use=($(resolve_problems))

    echo "Templates to test (${#templates_to_use[@]}):"
    printf '  %s\n' "${templates_to_use[@]}"
    echo ""

    echo "Problems to test (${#problems_to_use[@]}):"
    printf '  %s\n' "${problems_to_use[@]}"
    echo ""

    # Calculate total combinations
    local total_combinations=$((${#templates_to_use[@]} * ${#problems_to_use[@]}))
    echo "Total test combinations: $total_combinations"
    log_timestamp "Configuration complete. Starting $total_combinations test combinations."
    echo ""

    # Create output directory
    if [[ "$DRY_RUN" == false ]]; then
        mkdir -p "$OUTPUT_DIR"

        # Create summary log
        local summary_log="$OUTPUT_DIR/batch_summary_$TIMESTAMP.log"
        echo "Batch test started at $start_timestamp" > "$summary_log"
        echo "Configuration: iterations=$ITERATIONS, script=$PYTHON_SCRIPT" >> "$summary_log"
        echo "Total combinations: $total_combinations" >> "$summary_log"
        echo "" >> "$summary_log"
    fi

    # Execute tests with branching timing
    local current_test=0
    local successful_tests=0
    local failed_tests=0

    # Template-level timing loop
    for template in "${templates_to_use[@]}"; do
        local template_start_time=$(date +%s)
        log_timestamp "Starting template branch: $template (${#problems_to_use[@]} problems)"

        # Problem-level timing loop
        for problem in "${problems_to_use[@]}"; do
            local problem_start_time=$(date +%s)
            current_test=$((current_test + 1))

            echo ""
            echo "[$current_test/$total_combinations] Testing: $template with $problem"
            log_timestamp "Branch: $template → $problem (Test $current_test/$total_combinations)"

            # Construct command
            local cmd="python $PYTHON_SCRIPT -p $problem -i $ITERATIONS -t $template"

            # Create individual log file
            local log_file="$OUTPUT_DIR/test_${template%.*}_${problem}_${TIMESTAMP}.log"

            if execute_command "$cmd" "$log_file" "$template" "$problem"; then
                successful_tests=$((successful_tests + 1))
                echo "  ✓ Success"
            else
                failed_tests=$((failed_tests + 1))
                echo "  ✗ Failed (see $log_file)"
            fi

            # Problem-level timing summary
            local problem_end_time=$(date +%s)
            local problem_duration=$((problem_end_time - problem_start_time))
            local problem_formatted=$(format_duration $problem_duration)
            PROBLEM_TIMINGS["${template}|${problem}"]="$problem_duration"

            log_timestamp "Problem completed: $problem in $problem_formatted"

            # Add to summary log
            if [[ "$DRY_RUN" == false ]]; then
                echo "Test $current_test: $template + $problem = $([ $? -eq 0 ] && echo "SUCCESS" || echo "FAILED") (${problem_formatted})" >> "$summary_log"
            fi
        done

        # Template-level timing summary
        local template_end_time=$(date +%s)
        local template_duration=$((template_end_time - template_start_time))
        local template_formatted=$(format_duration $template_duration)
        TEMPLATE_TIMINGS["$template"]="$template_duration"

        echo ""
        log_timestamp "Template branch completed: $template in $template_formatted (${#problems_to_use[@]} problems)"
    done

    # Record global end time
    GLOBAL_END_TIME=$(date +%s)
    local total_duration=$((GLOBAL_END_TIME - GLOBAL_START_TIME))
    local total_formatted=$(format_duration $total_duration)

    echo ""
    log_timestamp "All tests completed. Generating results table..."

    # Generate results table with timing data
    generate_results_table "${templates_to_use[@]}" "--" "${problems_to_use[@]}"

    echo ""
    log_timestamp "EXPERIMENT COMPLETED"

    # Final summary with comprehensive timing
    echo ""
    echo "=== Batch Test Summary ==="
    echo "Start Time: $start_timestamp"
    echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Total Duration: $total_formatted"
    echo "Total tests: $total_combinations"
    echo "Successful: $successful_tests"
    echo "Failed: $failed_tests"
    echo "Success rate: $(( successful_tests * 100 / total_combinations ))%"
    echo "Average time per test: $(format_duration $((total_duration / total_combinations)))"

    # Template timing breakdown
    echo ""
    echo "=== Template Timing Breakdown ==="
    for template in "${templates_to_use[@]}"; do
        local template_time="${TEMPLATE_TIMINGS[$template]:-0}"
        local template_formatted=$(format_duration $template_time)
        local avg_per_problem=$((template_time / ${#problems_to_use[@]}))
        local avg_formatted=$(format_duration $avg_per_problem)
        echo "  $template: $template_formatted (avg: $avg_formatted per problem)"
    done

    if [[ "$DRY_RUN" == false ]]; then
        echo ""
        echo "Detailed logs available in: $OUTPUT_DIR"
        echo "Summary log: $summary_log"

        # Add comprehensive timing summary to log
        echo "" >> "$summary_log"
        echo "Batch test completed at $(date '+%Y-%m-%d %H:%M:%S')" >> "$summary_log"
        echo "Total duration: $total_formatted" >> "$summary_log"
        echo "Results: $successful_tests/$total_combinations successful ($(( successful_tests * 100 / total_combinations ))%)" >> "$summary_log"
        echo "Average time per test: $(format_duration $((total_duration / total_combinations)))" >> "$summary_log"

        # Add template timing breakdown to log
        echo "" >> "$summary_log"
        echo "Template timing breakdown:" >> "$summary_log"
        for template in "${templates_to_use[@]}"; do
            local template_time="${TEMPLATE_TIMINGS[$template]:-0}"
            local template_formatted=$(format_duration $template_time)
            echo "  $template: $template_formatted" >> "$summary_log"
        done
    fi
}

# Function to generate results table
generate_results_table() {
    local templates_ref=("$@")
    local templates=()
    local problems=()

    # Split the combined argument back into templates and problems
    local in_templates=true
    for arg in "${templates_ref[@]}"; do
        if [[ "$arg" == "--" ]]; then
            in_templates=false
            continue
        fi

        if [[ "$in_templates" == true ]]; then
            templates+=("$arg")
        else
            problems+=("$arg")
        fi
    done

    echo ""
    echo "=== Detailed Results Table ==="
    echo ""

    # Create CSV and formatted table files
    local csv_file="$OUTPUT_DIR/results_table_$TIMESTAMP.csv"
    local table_file="$OUTPUT_DIR/results_table_$TIMESTAMP.txt"
    local html_file="$OUTPUT_DIR/results_table_$TIMESTAMP.html"

    if [[ "$DRY_RUN" == false ]]; then
        # CSV Header with timing columns
        echo "Template,Problem_ID,Status,Score,Correctness%,Test_Duration,Branch_Duration,Total_Experiment_Duration,Error" > "$csv_file"

        # Calculate total experiment duration for CSV
        local total_experiment_duration=$((GLOBAL_END_TIME - GLOBAL_START_TIME))
        local total_experiment_formatted=$(format_duration $total_experiment_duration)

        # HTML Header
        cat > "$html_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>ARC-AGI Batch Test Results</title>
    <style>
        table { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .success { background-color: #d4edda; }
        .failed { background-color: #f8d7da; }
        .dry-run { background-color: #fff3cd; }
        .unknown { background-color: #e2e3e5; }
        .timing { font-family: monospace; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>ARC-AGI Batch Test Results</h1>
    <p>Generated: $(date)</p>
    <p>Iterations: $ITERATIONS | Total Tests: ${#RESULT_ORDER[@]} | Total Duration: $total_experiment_formatted</p>
    <table>
        <tr>
            <th>Template</th>
            <th>Problem ID</th>
            <th>Status</th>
            <th>Score</th>
            <th>Correctness %</th>
            <th>Test Duration</th>
            <th>Branch Duration</th>
            <th>Error/Notes</th>
        </tr>
EOF
    fi

    # Table header for console output
    printf "%-35s %-12s %-10s %-8s %-12s %-12s %-12s %s\n" \
        "Template" "Problem" "Status" "Score" "Correct%" "Test_Time" "Branch_Time" "Error/Notes"
    printf "%s\n" "$(printf '=%.0s' {1..140})"

    # Process results in order
    for key in "${RESULT_ORDER[@]}"; do
        IFS='|' read -r template problem <<< "$key"

        # Get result data - handle both bash and zsh associative array syntax
        local result_data=""
        if [[ -n "${ZSH_VERSION:-}" ]]; then
            result_data="${TEST_RESULTS[$key]:-}"
        else
            result_data="${TEST_RESULTS[$key]:-}"
        fi

        if [[ -z "$result_data" ]]; then
            if [[ "$DRY_RUN" == false ]]; then
                echo "Warning: No results found for $key"
            fi
            result_data="UNKNOWN|0|0|N/A|No data"
        fi

        IFS='|' read -r test_status score correctness duration error_msg <<< "$result_data"

        # Get timing data
        local individual_time="${INDIVIDUAL_TIMINGS[$key]:-0}"
        local branch_time="${PROBLEM_TIMINGS[$key]:-0}"
        local individual_formatted=$(format_duration $individual_time)
        local branch_formatted=$(format_duration $branch_time)

        # Truncate template name for console display
        local short_template="${template%.*}"
        if [[ ${#short_template} -gt 32 ]]; then
            short_template="${short_template:0:29}..."
        fi

        # Console output with timing data
        printf "%-35s %-12s %-10s %-8s %-12s %-12s %-12s %s\n" \
            "$short_template" "$problem" "$test_status" "$score" "$correctness" "$individual_formatted" "$branch_formatted" "$error_msg"

        if [[ "$DRY_RUN" == false ]]; then
            # CSV output with comprehensive timing data
            local total_experiment_duration=$((GLOBAL_END_TIME - GLOBAL_START_TIME))
            echo "$template,$problem,$test_status,$score,$correctness,$individual_formatted,$branch_formatted,$(format_duration $total_experiment_duration),\"$error_msg\"" >> "$csv_file"

            # HTML output
            local row_class=""
            case "$test_status" in
                "SUCCESS") row_class="success" ;;
                "FAILED") row_class="failed" ;;
                "DRY_RUN") row_class="dry-run" ;;
                *) row_class="unknown" ;;
            esac

            echo "        <tr class=\"$row_class\">" >> "$html_file"
            echo "            <td>$template</td>" >> "$html_file"
            echo "            <td>$problem</td>" >> "$html_file"
            echo "            <td>$test_status</td>" >> "$html_file"
            echo "            <td>$score</td>" >> "$html_file"
            echo "            <td>$correctness</td>" >> "$html_file"
            echo "            <td class=\"timing\">$individual_formatted</td>" >> "$html_file"
            echo "            <td class=\"timing\">$branch_formatted</td>" >> "$html_file"
            echo "            <td>$error_msg</td>" >> "$html_file"
            echo "        </tr>" >> "$html_file"
        fi
    done

    if [[ "$DRY_RUN" == false ]]; then
        # Close HTML with comprehensive timing statistics
        local total_experiment_duration=$((GLOBAL_END_TIME - GLOBAL_START_TIME))
        local avg_test_duration=$((total_experiment_duration / ${#RESULT_ORDER[@]}))

        cat >> "$html_file" << EOF
    </table>

    <h2>Summary Statistics</h2>
    <ul>
        <li>Total Tests: ${#RESULT_ORDER[@]}</li>
        <li>Success Rate: $(calculate_success_rate)%</li>
        <li>Average Correctness: $(calculate_avg_correctness)%</li>
        <li>Templates Tested: ${#templates[@]}</li>
        <li>Problems Tested: ${#problems[@]}</li>
        <li>Total Experiment Duration: $(format_duration $total_experiment_duration)</li>
        <li>Average Time per Test: $(format_duration $avg_test_duration)</li>
    </ul>

    <h2>Template Performance Breakdown</h2>
    <table>
        <tr>
            <th>Template</th>
            <th>Total Duration</th>
            <th>Avg per Problem</th>
            <th>Problems Tested</th>
        </tr>
EOF

        # Add template timing breakdown to HTML
        for template in "${templates[@]}"; do
            local template_time="${TEMPLATE_TIMINGS[$template]:-0}"
            local template_formatted=$(format_duration $template_time)
            local avg_per_problem=$((template_time / ${#problems[@]}))
            local avg_formatted=$(format_duration $avg_per_problem)

            cat >> "$html_file" << EOF
        <tr>
            <td>$template</td>
            <td class="timing">$template_formatted</td>
            <td class="timing">$avg_formatted</td>
            <td>${#problems[@]}</td>
        </tr>
EOF
        done

        cat >> "$html_file" << EOF
    </table>
</body>
</html>
EOF

        echo ""
        echo "Results saved to:"
        echo "  CSV: $csv_file"
        echo "  HTML: $html_file"
        echo "  Console table above"
    fi

    echo ""
    printf "%s\n" "$(printf '=%.0s' {1..120})"
}

# Helper function to calculate success rate
calculate_success_rate() {
    local success_count=0
    for key in "${RESULT_ORDER[@]}"; do
        local result_data="${TEST_RESULTS[$key]:-}"
        if [[ -n "$result_data" ]]; then
            IFS='|' read -r test_status _ _ _ _ <<< "$result_data"
            if [[ "$test_status" == "SUCCESS" ]]; then
                success_count=$((success_count + 1))
            fi
        fi
    done
    if [[ ${#RESULT_ORDER[@]} -gt 0 ]]; then
        echo $(( success_count * 100 / ${#RESULT_ORDER[@]} ))
    else
        echo "0"
    fi
}

# Helper function to calculate average correctness
calculate_avg_correctness() {
    local total_correctness=0
    local count=0
    for key in "${RESULT_ORDER[@]}"; do
        local result_data="${TEST_RESULTS[$key]:-}"
        if [[ -n "$result_data" ]]; then
            IFS='|' read -r _ _ correctness _ _ <<< "$result_data"
            if [[ "$correctness" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
                total_correctness=$(echo "$total_correctness + $correctness" | bc -l 2>/dev/null || echo "$total_correctness")
                count=$((count + 1))
            fi
        fi
    done
    if [[ $count -gt 0 ]]; then
        echo "scale=1; $total_correctness / $count" | bc -l 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Run main function
main "$@"