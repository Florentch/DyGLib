from pathlib import Path
import pandas as pd
from datetime import datetime
from optc_utils import load_pickle_file
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys


class GraphProcessor:
    """Process graph data and generate CSV files for dynamic graph analysis."""
    
    def __init__(self, input_path, output_path, label_path):
        self.folder_path = Path(input_path)
        self.csv_output_path = output_path
        self.label_path = label_path
        self.files = list(self.folder_path.glob("*.pkl"))
        
        # Fixed parameters
        self.time_interval = 15  # minutes
        self.num_features = 12
        self.min_events_threshold = 5000  # Minimum events per timestamp
        
        # Load anomaly IDs
        self.anomaly_ids = self._load_anomaly_ids()
        
        # Configure dataset-specific settings
        self._configure_dataset_settings()
        
        # Define split dates
        self.val_start_date = datetime(2019, 9, 22, 12, 0, 0).timestamp()
        self.test_start_date = datetime(2019, 9, 23, 0, 0, 0).timestamp()

    def _load_anomaly_ids(self):
        """Load anomaly IDs from JSON file."""
        try:
            with open(self.label_path) as f:
                return {json.loads(line)['id'] for line in f}
        except FileNotFoundError:
            print(f"Warning: Label file not found at {self.label_path}")
            return set()
        except Exception as e:
            print(f"Error loading label file: {e}")
            return set()

    def _configure_dataset_settings(self):
        """Configure dataset-specific thresholds and filters."""
        if '201' in str(self.folder_path):
            self.start_filter_date = datetime(2019, 9, 20, 0, 0, 0).timestamp()
        else:
            self.start_filter_date = None

    @staticmethod
    def _get_node_id(node):
        """Extract node ID as string."""
        return str(getattr(node, 'id', str(node)))

    def _round_timestamp(self, timestamp_str):
        """Round timestamp to 15-minute intervals."""
        dt = datetime.fromisoformat(timestamp_str)
        minute_block = (dt.minute // 15) * 15
        dt = dt.replace(minute=minute_block, second=0, microsecond=0)
        return dt.timestamp()

    def _scan_nodes(self):
        """Scan all files to create node mapping."""
        unique_nodes = set()
        
        for file in tqdm(self.files, desc="Scanning nodes"):
            data = load_pickle_file(file)
            nodes_in_file = set()
            
            for u, v, _, _ in data.edges(keys=True, data=True):
                nodes_in_file.update([self._get_node_id(u), self._get_node_id(v)])
            
            unique_nodes.update(nodes_in_file)
            del data
        
        node_mapping = {node_id: idx for idx, node_id in enumerate(sorted(unique_nodes))}
        print(f"Unique nodes: {len(node_mapping)}")
        return node_mapping

    def _extract_edges(self, node_mapping):
        """Extract and process edges from all files."""
        all_edges = []
        
        for file in tqdm(self.files, desc="Processing files"):
            data = load_pickle_file(file)
            
            for u, v, _, attr in data.edges(keys=True, data=True):
                u_id, v_id = self._get_node_id(u), self._get_node_id(v)
                
                if u_id in node_mapping and v_id in node_mapping:
                    is_anomaly = 1 if attr.get("eventid") in self.anomaly_ids else 0
                    all_edges.append((
                        node_mapping[u_id],
                        node_mapping[v_id],
                        attr.get("time"),
                        is_anomaly
                    ))
            del data
        
        print(f"Total edges processed: {len(all_edges)}")
        return all_edges

    def _get_valid_timestamps(self, timestamp_counts):
        """Determine which timestamps to keep based on criteria."""
        valid_timestamps = set()
        excluded_timestamps = set()
        
        for _, row in timestamp_counts.iterrows():
            timestamp, count = row['timestamp'], row['event_count']
            
            # Keep all timestamps after validation start date
            if timestamp >= self.val_start_date:
                valid_timestamps.add(timestamp)
            # Before validation: only keep if >= minimum events threshold
            elif count >= self.min_events_threshold:
                valid_timestamps.add(timestamp)
            else:
                excluded_timestamps.add(timestamp)
        
        return valid_timestamps, excluded_timestamps

    def _filter_by_timestamp_criteria(self, df):
        """Filter data based on timestamp criteria and event counts."""
        # Filter by start date if specified (for dataset 201)
        if self.start_filter_date:
            initial_count = len(df)
            df = df[df['timestamp'] >= self.start_filter_date]
            print(f"Filtered by start date: {initial_count} -> {len(df)} edges")
        
        # Group by timestamp to count events (before filtering for histogram)
        all_timestamp_counts = df.groupby('timestamp').size().reset_index(name='event_count')
        
        # Get valid timestamps
        valid_timestamps, excluded_timestamps = self._get_valid_timestamps(all_timestamp_counts)
        
        # Store for histogram
        self.all_timestamp_counts = all_timestamp_counts
        self.excluded_timestamps = excluded_timestamps
        
        # Filter DataFrame
        initial_count = len(df)
        df_filtered = df[df['timestamp'].isin(valid_timestamps)]
        
        print(f"Filtered by event count (>={self.min_events_threshold}): {initial_count} -> {len(df_filtered)} edges")
        print(f"Timestamps kept: {len(valid_timestamps)}, excluded: {len(excluded_timestamps)}")
        
        return df_filtered

    def _create_dataframe(self, all_edges):
        """Create final DataFrame with all columns and apply filtering."""
        # Create DataFrame efficiently
        df = pd.DataFrame(all_edges, columns=["source", "destination", "timestamp", "state_label"])
        
        # Process timestamps using vectorized operations
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed').dt.floor('15min').astype('int64') // 10**9
        
        # Apply filtering criteria
        df_filtered = self._filter_by_timestamp_criteria(df)
        
        print(f"Unique timestamps: {df_filtered['timestamp'].nunique()}")
        
        # Add feature columns
        feature_cols = {f'features{i+1}': 0 for i in range(self.num_features)}
        df_filtered = df_filtered.assign(**feature_cols)
            
        return df_filtered.sort_values('timestamp')

    def _calculate_expected_timestamps(self, start_timestamp, end_timestamp):
        """Calculate expected number of timestamps for a given period."""
        duration_seconds = end_timestamp - start_timestamp
        duration_minutes = duration_seconds / 60
        return int(duration_minutes / self.time_interval)

    def _get_split_details(self, timestamp_counts, split_name, start_ts, end_ts):
        """Calculate details for a given data split."""
        split_data = timestamp_counts[
            (timestamp_counts['timestamp'] >= start_ts) & 
            (timestamp_counts['timestamp'] < end_ts)
        ]
        
        events_total = split_data['event_count'].sum()
        anomalies_total = split_data['anomaly_count'].sum()
        timestamps_with_anomalies = len(split_data[split_data['anomaly_count'] > 0])
        timestamps_actual = len(split_data)
        timestamps_expected = self._calculate_expected_timestamps(start_ts, end_ts)
        
        return {
            'name': split_name,
            'start_date': datetime.fromtimestamp(start_ts),
            'end_date': datetime.fromtimestamp(end_ts),
            'events_total': events_total,
            'anomalies_total': anomalies_total,
            'timestamps_with_anomalies': timestamps_with_anomalies,
            'timestamps_actual': timestamps_actual,
            'timestamps_expected': timestamps_expected,
            'timestamps_without_anomalies': timestamps_actual - timestamps_with_anomalies
        }

    def _print_split_details(self, split_details):
        """Print formatted split details."""
        d = split_details
        print(f"\n=== {d['name'].upper()} SET DETAILS ===")
        print(f"Period: {d['start_date']} -> {d['end_date']}")
        print(f"Events: {d['events_total']:,}")
        
        completeness_indicator = "✓" if d['timestamps_actual'] == d['timestamps_expected'] else "⚠️"
        print(f"Timestamps: {d['timestamps_actual']}/{d['timestamps_expected']} ({completeness_indicator})")
        
        anomaly_percentage = d['timestamps_with_anomalies'] / d['timestamps_actual'] * 100
        print(f"Timestamps with anomalies: {d['timestamps_with_anomalies']}/{d['timestamps_actual']} ({anomaly_percentage:.1f}%)")
        print(f"Total anomalies: {d['anomalies_total']}")
        print(f"Timestamps without anomalies: {d['timestamps_without_anomalies']}")

    def _calculate_distribution_stats(self, timestamp_counts):
        """Calculate and print train/validation/test distribution statistics."""
        # Calculate details for each split
        splits = [
            ('train', timestamp_counts['timestamp'].min(), self.val_start_date),
            ('validation', self.val_start_date, self.test_start_date),
            ('test', self.test_start_date, timestamp_counts['timestamp'].max() + 1)
        ]
        
        split_details = [self._get_split_details(timestamp_counts, name, start, end) 
                        for name, start, end in splits]
        
        # Calculate global percentages
        total_events = sum(d['events_total'] for d in split_details)
        
        # Display global distribution
        print(f"\n=== TRAIN/VALIDATION/TEST DISTRIBUTION (events in final CSV) ===")
        for d in split_details:
            percentage = d['events_total'] / total_events * 100
            print(f"{d['name'].upper()}: {d['events_total']:,} events ({percentage:.1f}%)")
        print(f"TOTAL: {total_events:,} events")
        
        # Display detailed information for validation and test sets
        for d in split_details[1:]:  # Skip train details
            self._print_split_details(d)

    def _prepare_histogram_data(self, df):
        """Prepare data for histogram visualization."""
        all_counts = self.all_timestamp_counts.copy()
        all_counts['datetime'] = pd.to_datetime(all_counts['timestamp'], unit='s')
        
        # Add anomaly information to complete data
        anomaly_info = df.groupby('timestamp')['state_label'].sum().reset_index()
        anomaly_info.columns = ['timestamp', 'anomaly_count']
        all_counts = all_counts.merge(anomaly_info, on='timestamp', how='left')
        all_counts['anomaly_count'] = all_counts['anomaly_count'].fillna(0)
        
        # Mark excluded timestamps
        all_counts['excluded'] = all_counts['timestamp'].isin(self.excluded_timestamps)
        
        return all_counts

    def _plot_histogram(self, all_counts):
        """Create histogram visualization."""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Separate included and excluded data
        included_data = all_counts[~all_counts['excluded']]
        excluded_data = all_counts[all_counts['excluded']]
        
        # Plot included data
        if len(included_data) > 0:
            colors = ['red' if anomalies > 0 else 'blue' for anomalies in included_data['anomaly_count']]
            ax.bar(included_data['datetime'], included_data['event_count'], 
                   color=colors, alpha=0.7, width=pd.Timedelta(minutes=10),
                   label='Included in CSV')
        
        # Plot excluded data (below zero line)
        if len(excluded_data) > 0:
            ax.bar(excluded_data['datetime'], -excluded_data['event_count'], 
                   color='orange', alpha=0.7, width=pd.Timedelta(minutes=10),
                   label=f'Excluded (< {self.min_events_threshold} events)')
        
        # Add split boundary lines
        split_lines = [
            (self.val_start_date, 'Validation Start'),
            (self.test_start_date, 'Test Start')
        ]
        
        for timestamp, label in split_lines:
            ax.axvline(pd.to_datetime(timestamp, unit='s'), color='green', 
                      linewidth=3, label=label)
        
        # Configure plot
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Events\n(↑ Included in CSV / ↓ Excluded)')
        ax.set_title(f'Event Distribution by 15-minute Intervals - Train/Val/Test Split\n'
                    f'Red: anomalies, Blue: normal, Orange: excluded (< {self.min_events_threshold} events)')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig

    def _create_histogram(self, df):
        """Create histogram with anomaly detection and excluded timestamps visualization."""
        # Group by timestamp and count events (data in CSV)
        timestamp_counts = df.groupby('timestamp').agg({
            'state_label': ['count', 'sum']
        }).reset_index()
        timestamp_counts.columns = ['timestamp', 'event_count', 'anomaly_count']
        
        # Prepare histogram data (complete data before filtering)
        all_counts = self._prepare_histogram_data(df)
        
        # Create plot
        fig = self._plot_histogram(all_counts)
        
        # Calculate distribution statistics (only on included data)
        self._calculate_distribution_stats(timestamp_counts)
        
        return fig

    def process(self):
        """Main processing pipeline."""
        print(f"Processing: {self.folder_path}")
        print(f"Label file: {self.label_path}")
        print(f"Time interval: {self.time_interval} minutes")
        print(f"Output: {self.csv_output_path}")
        
        if self.start_filter_date:
            print(f"Start filter date: {datetime.fromtimestamp(self.start_filter_date)}")
        print(f"Validation start: {datetime.fromtimestamp(self.val_start_date)}")
        print(f"Test start: {datetime.fromtimestamp(self.test_start_date)}")
        
        # Validate inputs
        if not self.files:
            print(f"Error: No .pkl files found in {self.folder_path}")
            return False
        
        # Execute processing pipeline
        node_mapping = self._scan_nodes()
        all_edges = self._extract_edges(node_mapping)
        df = self._create_dataframe(all_edges)
        
        # Create output directory if it doesn't exist
        Path(self.csv_output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        df.to_csv(self.csv_output_path, index=False)
        print(f"CSV created: {self.csv_output_path} with {len(df)} edges")
        
        # Create and save histogram
        fig = self._create_histogram(df)
        histogram_path = self.csv_output_path.replace('.csv', '_histogram.png')
        fig.savefig(histogram_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved: {histogram_path}")
        
        plt.show()
        print("-" * 50)
        return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process graph data and generate CSV files for dynamic graph analysis.")
    
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to directory containing .pkl files'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output path for the CSV file'
    )
    
    parser.add_argument(
        '--label_path',
        type=str,
        required=True,
        help='Path to the JSON file containing anomaly labels'
    )
    
    return parser.parse_args()


def validate_paths(args):
    """Validate input paths exist and output path is writable."""
    errors = []
    
    # Check input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        errors.append(f"Input path does not exist: {input_path}")
    elif not input_path.is_dir():
        errors.append(f"Input path is not a directory: {input_path}")
    
    # Check label path
    label_path = Path(args.label_path)
    if not label_path.exists():
        errors.append(f"Label file does not exist: {label_path}")
    elif not label_path.is_file():
        errors.append(f"Label path is not a file: {label_path}")
    
    # Check output path directory
    output_path = Path(args.output_path)
    output_dir = output_path.parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            errors.append(f"Cannot create output directory {output_dir}: {e}")
    
    return errors


def main():
    """Main function with command line argument parsing."""
    args = parse_arguments()
    
    # Validate paths
    validation_errors = validate_paths(args)
    if validation_errors:
        print("Validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Process the dataset
    processor = GraphProcessor(args.input_path, args.output_path, args.label_path)
    processor.process()


if __name__ == "__main__":
    main()