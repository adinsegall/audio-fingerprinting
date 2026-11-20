def evaluate_results(output_file, parameter_config=None):
    """
    Analyze the output file to determine audio identification accuracy and calculate
    information retrieval metrics.
    
    This function parses the output file from the audio identification system and
    evaluates its performance using multiple metrics including accuracy at different
    ranks, precision, recall, F-measure, and Mean Average Precision (MAP). It also
    performs category-specific analysis for different music genres.
    
    Args:
        output_file (str): Path to the output.txt file with matching results. Each line
                          should contain a query filename followed by up to three matches,
                          separated by tabs.
        parameter_config (dict, optional): Dictionary with parameter settings used for 
                                         sensitivity analysis. Default is None.
        
    Returns:
        dict: A dictionary with accuracy statistics including:
            - total_queries: Total number of queries processed
            - queries_with_matches: Number of queries that returned at least one match
            - correct_top1/top3: Number of queries with correct matches at rank 1 or in top 3
            - precision, recall, f_measure: Standard IR metrics
            - map: Mean Average Precision
            - Category-specific metrics for classical and pop music
    """
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    # Initialize counters for various metrics
    total_queries = 0
    correct_top1 = 0  # Correct matches in position 1
    correct_top2 = 0  # Correct matches in position 2
    correct_top3 = 0  # Correct matches in any of the top 3 positions
    queries_with_matches = 0  # Queries that have at least one match
    
    # For MAP calculation
    average_precisions = []
    
    # For category analysis
    classical_queries = 0
    classical_correct_top1 = 0
    classical_correct_top3 = 0
    pop_queries = 0
    pop_correct_top1 = 0
    pop_correct_top3 = 0
    
    # For precision and recall
    true_positives_top1 = 0  # Correct match at position 1
    false_positives_top1 = 0  # Incorrect match at position 1
    false_negatives = 0  # No correct match returned
    
    # Process each line in the output file
    for line in lines:
        parts = line.strip().split('\t')
        query_file = parts[0]
        
        # Extract expected database filename from query filename
        # Pattern: "classical.00045-snippet-10-0.wav" -> "classical.00045.wav"
        if '-snippet-' in query_file:
            prefix = query_file.split('-snippet-')[0]
            expected_match = prefix + '.wav'
            
            # Category analysis
            if query_file.startswith('classical'):
                classical_queries += 1
            elif query_file.startswith('pop'):
                pop_queries += 1
        else:
            # Handle case where query filename doesn't follow the expected pattern
            # This is a safeguard in case there are different naming conventions
            print(f"Warning: Query file {query_file} doesn't follow expected naming pattern")
            continue
        
        # Check if we have any matches
        matches = [match for match in parts[1:] if match.strip()]
        if matches:
            queries_with_matches += 1
            
            # Calculate precision at each rank for MAP
            # Records positions of relevant items for one query
            relevance = [1 if match.strip() == expected_match else 0 for match in parts[1:4]]
            
            # If any matches are correct, calculate average precision
            if sum(relevance) > 0:
                precision_at_k = []
                relevant_count = 0
                for k, is_relevant in enumerate(relevance, 1):
                    if is_relevant:
                        relevant_count += 1
                        precision_at_k.append(relevant_count / k)
                
                # Average precision for this query
                avg_precision = sum(precision_at_k) / sum(relevance)
                average_precisions.append(avg_precision)
            else:
                average_precisions.append(0)
                
            # Check for correct match in top 3 positions
            found_match = False
            for i, match in enumerate(parts[1:4]):  # Only check top 3
                if match.strip() == expected_match:
                    if i == 0:  # First position
                        correct_top1 += 1
                        correct_top3 += 1
                        
                        # Category analysis  
                        if query_file.startswith('classical'):
                            classical_correct_top1 += 1
                            classical_correct_top3 += 1
                        elif query_file.startswith('pop'):
                            pop_correct_top1 += 1
                            pop_correct_top3 += 1
                            
                        # For precision/recall
                        true_positives_top1 += 1
                    elif i == 1:  # Second position
                        correct_top2 += 1
                        correct_top3 += 1
                        
                        # Category analysis
                        if query_file.startswith('classical'):
                            classical_correct_top3 += 1
                        elif query_file.startswith('pop'):
                            pop_correct_top3 += 1
                    elif i == 2:  # Third position
                        correct_top3 += 1
                        
                        # Category analysis
                        if query_file.startswith('classical'):
                            classical_correct_top3 += 1
                        elif query_file.startswith('pop'):
                            pop_correct_top3 += 1
                    
                    found_match = True
                    break
            
            # Count false positives at rank 1
            if not found_match or (found_match and matches[0].strip() != expected_match):
                false_positives_top1 += 1
        else:
            # No matches returned, this is a false negative
            false_negatives += 1
        
        total_queries += 1
    
    # Calculate overall accuracy metrics
    top1_accuracy = (correct_top1 / total_queries) * 100 if total_queries > 0 else 0
    top3_accuracy = (correct_top3 / total_queries) * 100 if total_queries > 0 else 0
    match_rate = (queries_with_matches / total_queries) * 100 if total_queries > 0 else 0
    
    # Calculate Mean Average Precision (MAP)
    mean_average_precision = sum(average_precisions) / len(average_precisions) if average_precisions else 0
    
    # Calculate precision (proportion of retrieved items that are relevant)
    precision = true_positives_top1 / (true_positives_top1 + false_positives_top1) if (true_positives_top1 + false_positives_top1) > 0 else 0
    
    # Calculate recall (proportion of relevant items that are retrieved at rank 1)
    recall = true_positives_top1 / (true_positives_top1 + false_negatives) if (true_positives_top1 + false_negatives) > 0 else 0
    
    # Calculate F-measure (harmonic mean of precision and recall)
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate category-specific accuracy
    classical_top1_accuracy = (classical_correct_top1 / classical_queries) * 100 if classical_queries > 0 else 0
    classical_top3_accuracy = (classical_correct_top3 / classical_queries) * 100 if classical_queries > 0 else 0
    pop_top1_accuracy = (pop_correct_top1 / pop_queries) * 100 if pop_queries > 0 else 0  
    pop_top3_accuracy = (pop_correct_top3 / pop_queries) * 100 if pop_queries > 0 else 0
    
    # Print detailed summary
    print(f"=== AUDIO IDENTIFICATION EVALUATION ===")
    print(f"Total queries: {total_queries}")
    print(f"Queries with any matches: {queries_with_matches}/{total_queries} ({match_rate:.1f}%)")
    print(f"Correct matches at rank 1: {correct_top1}/{total_queries} ({top1_accuracy:.1f}%)")
    print(f"Correct matches in top 3: {correct_top3}/{total_queries} ({top3_accuracy:.1f}%)")
    
    print(f"\n=== RETRIEVAL METRICS ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F-measure: {f_measure:.3f}")
    print(f"Mean Average Precision (MAP): {mean_average_precision:.3f}")
    
    print(f"\n=== CATEGORY ANALYSIS ===")
    print(f"Classical queries: {classical_queries}")
    print(f"Classical top-1 accuracy: {classical_top1_accuracy:.1f}%")
    print(f"Classical top-3 accuracy: {classical_top3_accuracy:.1f}%")
    print(f"Pop queries: {pop_queries}")
    print(f"Pop top-1 accuracy: {pop_top1_accuracy:.1f}%")
    print(f"Pop top-3 accuracy: {pop_top3_accuracy:.1f}%")
    
    # Print parameter configuration if provided
    if parameter_config:
        print(f"\n=== PARAMETER CONFIGURATION ===")
        for param, value in parameter_config.items():
            print(f"{param}: {value}")
    
    # Return comprehensive results dictionary for further analysis or visualization
    results = {
        "total_queries": total_queries,
        "queries_with_matches": queries_with_matches,
        "correct_top1": correct_top1,
        "correct_top3": correct_top3,
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "match_rate": match_rate,
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
        "map": mean_average_precision,
        "classical_queries": classical_queries,
        "classical_correct_top1": classical_correct_top1,
        "classical_correct_top3": classical_correct_top3,
        "classical_top1_accuracy": classical_top1_accuracy,
        "classical_top3_accuracy": classical_top3_accuracy,
        "pop_queries": pop_queries,
        "pop_correct_top1": pop_correct_top1,
        "pop_correct_top3": pop_correct_top3,
        "pop_top1_accuracy": pop_top1_accuracy,
        "pop_top3_accuracy": pop_top3_accuracy,
        "parameter_config": parameter_config
    }
    return results


def parameter_sensitivity_analysis():
    """
    Run evaluation with different parameter configurations to analyze sensitivity.
    
    This function tests multiple audio fingerprinting parameter configurations to
    determine how different settings affect identification accuracy. It can help
    identify optimal parameter values and understand the trade-offs between
    different configuration choices.
    
    The function would typically:
    1. Run the audio identification with each parameter configuration
    2. Evaluate results for each configuration
    3. Compare performance metrics across configurations
    
    Returns:
        list: List of result dictionaries, one for each parameter configuration
    """
    # Sample parameter configurations to test
    parameter_configs = [
        {
            "window_size": 1024,    # Standard FFT window size
            "hop_length": 512,      # 50% overlap
            "min_distance": 10,     # Minimum distance between peaks
            "threshold_rel": 0.05   # Relative threshold for peak detection
        },
        {
            "window_size": 2048,    # Larger window - better frequency resolution
            "hop_length": 512,
            "min_distance": 10,
            "threshold_rel": 0.05
        },
        {
            "window_size": 1024,
            "hop_length": 256,      # Smaller hop - better time resolution
            "min_distance": 10,
            "threshold_rel": 0.05
        },
        {
            "window_size": 1024,
            "hop_length": 512,
            "min_distance": 15,     # Larger minimum distance - fewer peaks
            "threshold_rel": 0.05
        },
        {
            "window_size": 1024,
            "hop_length": 512,
            "min_distance": 10,
            "threshold_rel": 0.1    # Higher threshold - more selective peak picking
        }
    ]
    
    results = []
    
    for config in parameter_configs:
        # You would need to regenerate your output file with these parameters
        # For demonstration, we'll just evaluate the same file multiple times
        print(f"\n=== Testing configuration {config} ===")
        
        # In a real implementation, you would regenerate your fingerprints and run matching
        # with these parameters first, then evaluate the resulting output file
        
        result = evaluate_results("output.txt", config)
        results.append(result)
    
    # You could plot these results to visualize parameter sensitivity
    # For example, plotting F-measure for different configurations
    
    return results


# Call the function directly with your output file path
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    output_path = "output.txt"  # Change this if your file is in a different location
    results = evaluate_results(output_path)