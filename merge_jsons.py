import argparse
import json
import csv
import os

"""
Run using: 
python merge_jsons.py path_to_your_json_directory output.csv

"""
def json_to_csv(json_dir, output_csv):
    
    json_files = [os.path.join(json_dir, file) for file in os.listdir(json_dir) if file.endswith('.json')]

    # Collect data from all JSON files
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_data.append(data)
                                                        
    # Extract the headers from the first JSON object
    if all_data:
        import pdb
        pdb.set_trace()
        headers = all_data[0].keys()
    else:
        raise ValueError("No data found in JSON files")
                                                                                            
    # Write data to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_data)

# Example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert JSON files in a directory to a CSV file.')
    parser.add_argument('json_dir', type=str, help='The directory containing JSON files')
    parser.add_argument('output_csv', type=str, help='The output CSV file path')
    args = parser.parse_args()

    json_to_csv(args.json_dir, args.output_csv)

