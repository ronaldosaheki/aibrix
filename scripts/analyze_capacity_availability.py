#!/usr/bin/env python3
# /// script
# dependencies = [
#   "boto3",
# ]
# ///
"""
Script to analyze AWS capacity block availability from S3 files.

This script:
1. Downloads files from S3 bucket
2. Parses instance type, region, and availability data
3. Generates a report showing which regions have best availability
"""

import boto3
import json
import os
import re
import csv
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import tempfile


class CapacityAvailabilityAnalyzer:
    def __init__(self, bucket_name: str, prefix: str, local_dir: str = None):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client('s3')
        self.local_dir = local_dir or tempfile.mkdtemp()
        self.availability_data = []
        
    def download_files(self) -> List[str]:
        """Download all files from S3 bucket/prefix"""
        print(f"📥 Downloading files from s3://{self.bucket_name}/{self.prefix}")
        
        # List all objects in the bucket with the prefix
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=self.prefix
        )
        
        if 'Contents' not in response:
            print("❌ No files found in S3 bucket")
            return []
        
        downloaded_files = []
        for obj in response['Contents']:
            key = obj['Key']
            # Skip directories
            if key.endswith('/'):
                continue
                
            filename = os.path.basename(key)
            local_path = os.path.join(self.local_dir, filename)
            
            print(f"  ⬇️  Downloading {filename}...")
            self.s3_client.download_file(self.bucket_name, key, local_path)
            downloaded_files.append(local_path)
        
        print(f"✅ Downloaded {len(downloaded_files)} files to {self.local_dir}")
        return downloaded_files
    
    def extract_date_from_filename(self, filename: str) -> datetime:
        """Extract date from filename (assumes format includes date)"""
        # Common date patterns: YYYY-MM-DD, YYYYMMDD, YYYY_MM_DD, etc.
        patterns = [
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
            r'(\d{4})(\d{2})(\d{2})',    # YYYYMMDD
            r'(\d{4})_(\d{2})_(\d{2})',  # YYYY_MM_DD
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                year, month, day = match.groups()
                try:
                    return datetime(int(year), int(month), int(day))
                except ValueError:
                    continue
        
        # Fallback: use file modification time
        return datetime.now()
    
    def parse_file(self, filepath: str) -> List[Dict]:
        """Parse a single file for capacity availability data"""
        filename = os.path.basename(filepath)
        file_date = self.extract_date_from_filename(filename)
        
        records = []
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Try to parse as JSON first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        records.append(self._extract_record(item, filename, file_date))
                else:
                    records.append(self._extract_record(data, filename, file_date))
            except json.JSONDecodeError:
                # Parse as text file
                records.extend(self._parse_text_file(content, filename, file_date))
        
        except Exception as e:
            print(f"⚠️  Error parsing {filename}: {e}")
        
        return records
    
    def _extract_record(self, data: Dict, filename: str, file_date: datetime) -> Dict:
        """Extract relevant fields from a data record"""
        return {
            'filename': filename,
            'file_date': file_date,
            'instance_type': data.get('InstanceType', data.get('instance_type', 'unknown')),
            'region': data.get('Region', data.get('region', data.get('AvailabilityZone', 'unknown'))),
            'available': data.get('Available', data.get('available', data.get('Capacity', True))),
            'count': data.get('Count', data.get('count', data.get('InstanceCount', 1))),
            'raw_data': data
        }
    
    def _parse_text_file(self, content: str, filename: str, file_date: datetime) -> List[Dict]:
        """Parse text file with region, instance type, and JSON capacity block offerings"""
        records = []
        
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for region pattern (first line of each block)
            region_match = re.match(r'^(us|eu|ap|sa|ca|me|af)-[a-z]+-\d[a-z]?$', line)
            if region_match:
                region = region_match.group(0)
                i += 1
                
                # Next line should be instance type
                if i < len(lines):
                    instance_type = lines[i].strip()
                    i += 1
                    
                    # Next should be JSON starting with {
                    json_lines = []
                    brace_count = 0
                    while i < len(lines):
                        json_line = lines[i]
                        json_lines.append(json_line)
                        
                        # Count braces to find end of JSON
                        brace_count += json_line.count('{') - json_line.count('}')
                        i += 1
                        
                        if brace_count == 0 and json_lines:
                            break
                    
                    # Parse the JSON
                    try:
                        json_str = '\n'.join(json_lines)
                        data = json.loads(json_str)
                        offerings = data.get('CapacityBlockOfferings', [])
                        
                        if offerings:
                            # Available capacity - create record for each offering
                            for offering in offerings:
                                records.append({
                                    'filename': filename,
                                    'file_date': file_date,
                                    'instance_type': offering.get('InstanceType', instance_type),
                                    'region': region,
                                    'availability_zone': offering.get('AvailabilityZone', region),
                                    'available': True,
                                    'count': offering.get('InstanceCount', 1),
                                    'start_date': offering.get('StartDate'),
                                    'end_date': offering.get('EndDate'),
                                    'duration_hours': offering.get('CapacityBlockDurationHours'),
                                    'upfront_fee': offering.get('UpfrontFee'),
                                    'raw_data': offering
                                })
                        else:
                            # No availability
                            records.append({
                                'filename': filename,
                                'file_date': file_date,
                                'instance_type': instance_type,
                                'region': region,
                                'availability_zone': region,
                                'available': False,
                                'count': 0,
                                'start_date': None,
                                'end_date': None,
                                'duration_hours': None,
                                'upfront_fee': None,
                                'raw_data': {}
                            })
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Error parsing JSON in {filename} for {region}/{instance_type}: {e}")
                        i += 1
                        continue
            else:
                i += 1
        
        return records
    
    def analyze_availability(self, files: List[str]) -> Dict:
        """Analyze all files and generate availability report"""
        print("\n📊 Analyzing capacity availability...")
        
        all_records = []
        for filepath in files:
            records = self.parse_file(filepath)
            all_records.extend(records)
        
        self.availability_data = all_records
        
        # Aggregate by region AND instance type
        region_instance_stats = defaultdict(lambda: {
            'total_instances': 0,
            'available_instances': 0,
            'files_with_capacity': set(),  # Files that had capacity
            'all_files': set(),  # All files checked
            'latest_date': None,
            'availability_zones': defaultdict(lambda: {
                'available_count': 0,
                'total_count': 0,
                'files_with_capacity': set(),
                'all_files': set(),
                'num_offerings': 0,  # Count of individual capacity block offerings
                'days_to_start_list': []  # List of days to start for distribution
            }),
            'days_diff_sum': 0,
            'days_diff_count': 0,
            'days_to_start_list': [],  # For overall distribution
            'instances_per_file': defaultdict(int)
        })
        
        for record in all_records:
            region = record['region']
            instance_type = record['instance_type']
            count = record.get('count', 1)
            az = record.get('availability_zone', region)
            is_available = record.get('available', False)
            filename = record['filename']
            
            # Create unique key for region + instance type
            key = f"{region}/{instance_type}"
            stats = region_instance_stats[key]
            
            # Store region and instance type separately
            stats['region'] = region
            stats['instance_type'] = instance_type
            
            # Track all files that were checked for this region/instance-type
            stats['all_files'].add(filename)
            stats['availability_zones'][az]['all_files'].add(filename)
            
            # Count instances only if available
            if is_available:
                stats['total_instances'] += count
                stats['available_instances'] += count
                stats['files_with_capacity'].add(filename)
                stats['availability_zones'][az]['available_count'] += count
                stats['availability_zones'][az]['files_with_capacity'].add(filename)
                stats['availability_zones'][az]['total_count'] += count
                stats['availability_zones'][az]['num_offerings'] += 1  # Count each offering
            
            # Track instances per file
            stats['instances_per_file'][record['filename']] += count
            
            # Calculate days between file date and start date
            if record.get('start_date') and record.get('available', False):
                try:
                    # Parse start date (with timezone)
                    start_date_str = record['start_date'].replace('Z', '+00:00')
                    start_date = datetime.fromisoformat(start_date_str)
                    
                    # Make file_date timezone-aware for comparison
                    file_date = record['file_date']
                    if file_date.tzinfo is None:
                        from datetime import timezone
                        file_date = file_date.replace(tzinfo=timezone.utc)
                    
                    # Calculate difference in hours, then convert to "days" based on 24-hour periods
                    time_diff = start_date - file_date
                    hours_diff = time_diff.total_seconds() / 3600
                    
                    # Convert to days: 0-1h = 0 days, 1-24h = 1 day, 24-48h = 2 days, etc.
                    if hours_diff <= 1:
                        days_diff = 0
                    else:
                        days_diff = int((hours_diff - 1) / 24) + 1
                    
                    stats['days_diff_sum'] += days_diff
                    stats['days_diff_count'] += 1
                    stats['days_to_start_list'].append(days_diff)
                    stats['availability_zones'][az]['days_to_start_list'].append(days_diff)
                except Exception as e:
                    # Debug: print error
                    #print(f"Error parsing date: {e}")
                    pass
            
            # Track latest date
            if stats['latest_date'] is None or \
               record['file_date'] > stats['latest_date']:
                stats['latest_date'] = record['file_date']
        
        # Convert sets to lists and calculate averages
        for key in region_instance_stats:
            stats = region_instance_stats[key]
            
            # Calculate total checks and available checks based on files
            total_checks = len(stats['all_files'])
            available_checks = len(stats['files_with_capacity'])
            
            stats['total_checks'] = total_checks
            stats['available_checks'] = available_checks
            
            # Find best AZ
            best_az = None
            best_az_count = 0
            best_az_rate = 0
            az_details = {}
            for az, az_data in stats['availability_zones'].items():
                # Availability rate = files with capacity in this AZ / total files checked for region/instance-type
                az_available_checks = len(az_data['files_with_capacity'])
                az_rate = az_available_checks / total_checks if total_checks > 0 else 0
                az_total_checks_in_az = len(az_data['all_files'])  # Files that checked this specific AZ
                
                # Calculate average instances per offering
                az_avg_instances_per_offering = az_data['available_count'] / az_data['num_offerings'] if az_data['num_offerings'] > 0 else 0
                
                # Calculate distribution of days to start for this AZ
                from collections import Counter
                az_days_dist = Counter(az_data['days_to_start_list'])
                az_days_dist_str = ','.join([f"{days}d:{count}" for days, count in sorted(az_days_dist.items())])
                
                az_details[az] = {
                    'available_count': az_data['available_count'],
                    'total_count': az_data['total_count'],
                    'checks': total_checks,  # Total files checked for this region/instance-type
                    'available_checks': az_available_checks,  # Files with capacity in this AZ
                    'availability_rate': az_rate,  # Rate relative to total checks
                    'num_offerings': az_data['num_offerings'],
                    'avg_instances_per_offering': az_avg_instances_per_offering,
                    'days_to_start_distribution': az_days_dist_str
                }
                # Best AZ is the one with highest availability rate, then most instances
                if az_rate > best_az_rate or (az_rate == best_az_rate and az_data['available_count'] > best_az_count):
                    best_az = az
                    best_az_count = az_data['available_count']
                    best_az_rate = az_rate
            
            stats['best_availability_zone'] = best_az
            stats['best_az_available_count'] = best_az_count
            stats['availability_zones'] = az_details
            
            # Calculate average days difference
            if stats['days_diff_count'] > 0:
                stats['avg_days_to_start'] = stats['days_diff_sum'] / stats['days_diff_count']
            else:
                stats['avg_days_to_start'] = None
            
            # Calculate average instances per file
            if stats['instances_per_file']:
                total_instances_across_files = sum(stats['instances_per_file'].values())
                num_files = len(stats['instances_per_file'])
                stats['avg_instances_per_file'] = total_instances_across_files / num_files
            else:
                stats['avg_instances_per_file'] = 0
            
            # Clean up temporary fields
            del stats['days_diff_sum']
            del stats['days_diff_count']
            del stats['instances_per_file']
            del stats['files_with_capacity']
            del stats['all_files']
            
            # Convert remaining sets to lists - keep file count
            stats['latest_date'] = stats['latest_date'].isoformat()
        
        return dict(region_instance_stats)
    
    def generate_report(self, region_stats: Dict) -> Dict:
        """Generate final report with rankings"""
        print("\n📈 Generating report...")
        
        current_date = datetime.now()
        
        # Calculate scores for each region/instance-type combination
        scored_entries = []
        for key, stats in region_stats.items():
            region = stats.get('region', 'unknown')
            instance_type = stats.get('instance_type', 'unknown')
            
            if region == 'unknown' or instance_type == 'unknown':
                continue
            
            latest_date = datetime.fromisoformat(stats['latest_date'])
            days_old = (current_date - latest_date).days
            
            # Score calculation:
            # - More available instances = better
            # - Lower average time to start = better
            # - Higher availability rate = better
            
            # Time-to-start score (max 100 points, lower time = higher score)
            avg_days = stats.get('avg_days_to_start')
            if avg_days is not None and avg_days >= 0:
                # 0 days = 100 points, 1 day = 90 points, 2 days = 80 points, etc.
                # Cap at 10+ days = 0 points
                time_score = max(0, 100 - (avg_days * 10))
            else:
                time_score = 0  # No data = 0 points
            
            availability_score = stats['available_instances']
            rate_score = (stats['available_checks'] / stats['total_checks'] * 100) if stats['total_checks'] > 0 else 0
            
            total_score = time_score + availability_score + rate_score
            
            scored_entries.append({
                'region': region,
                'instance_type': instance_type,
                'score': total_score,
                'available_instances': stats['available_instances'],
                'total_instances': stats['total_instances'],
                'total_checks': stats['total_checks'],
                'available_checks': stats['available_checks'],
                'availability_rate': stats['available_checks'] / stats['total_checks'] if stats['total_checks'] > 0 else 0,
                'latest_date': stats['latest_date'],
                'days_old': days_old,
                'files_count': stats['total_checks'],  # Number of files checked
                'best_availability_zone': stats['best_availability_zone'],
                'best_az_available_count': stats['best_az_available_count'],
                'availability_zones': stats['availability_zones'],
                'avg_days_to_start': stats['avg_days_to_start'],
                'avg_instances_per_file': stats['avg_instances_per_file']
            })
        
        # Sort by score (descending)
        scored_entries.sort(key=lambda x: x['score'], reverse=True)
        
        # Count unique regions
        unique_regions = len(set(entry['region'] for entry in scored_entries))
        
        # Calculate date range
        if self.availability_data:
            dates = [record['file_date'] for record in self.availability_data]
            min_date = min(dates)
            max_date = max(dates)
            date_range = {
                'start_date': min_date.isoformat(),
                'end_date': max_date.isoformat(),
                'days_covered': (max_date - min_date).days + 1
            }
        else:
            date_range = None
        
        report = {
            'generated_at': current_date.isoformat(),
            'analysis_period': date_range,
            'total_entries_analyzed': len(scored_entries),
            'total_regions': unique_regions,
            'total_records': len(self.availability_data),
            'top_entries': scored_entries[:20],  # Top 20 region/instance-type combinations
            'all_entries': scored_entries
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print human-readable report"""
        print("\n" + "="*80)
        print("🎯 AWS CAPACITY BLOCK AVAILABILITY REPORT (BY REGION & INSTANCE TYPE)")
        print("="*80)
        print(f"\n📅 Generated at: {report['generated_at']}")
        
        # Show analysis period
        if report.get('analysis_period'):
            period = report['analysis_period']
            start = datetime.fromisoformat(period['start_date']).strftime('%Y-%m-%d')
            end = datetime.fromisoformat(period['end_date']).strftime('%Y-%m-%d')
            print(f"📆 Analysis Period: {start} to {end} ({period['days_covered']} days)")
        
        print(f"📊 Total entries analyzed: {report['total_entries_analyzed']}")
        print(f"🌍 Total regions: {report['total_regions']}")
        print(f"📝 Total records: {report['total_records']}")
        
        print("\n" + "="*80)
        print("🏆 TOP 20 REGION/INSTANCE-TYPE COMBINATIONS BY AVAILABILITY")
        print("="*80)
        
        for i, entry in enumerate(report['top_entries'], 1):
            print(f"\n{i}. {entry['region']} / {entry['instance_type']}")
            print(f"   Score: {entry['score']:.0f}")
            print(f"   Available Instances: {entry['available_instances']} (from {entry['available_checks']} checks)")
            print(f"   Availability Rate: {entry['availability_rate']*100:.1f}% ({entry['available_checks']}/{entry['total_checks']} checks had capacity)")
            print(f"   Best AZ: {entry['best_availability_zone']} ({entry['best_az_available_count']} instances)")
            
            # Show all AZs with availability rate
            az_list = []
            for az, az_info in entry['availability_zones'].items():
                rate = az_info['availability_rate'] * 100
                az_list.append(f"{az}:{az_info['available_count']}inst ({rate:.0f}%)")
            print(f"   All AZs: {', '.join(az_list)}")
            
            # Show averages
            if entry['avg_days_to_start'] is not None:
                print(f"   Avg Days to Start: {entry['avg_days_to_start']:.1f} days")
            else:
                print(f"   Avg Days to Start: N/A")
            print(f"   Avg Instances per Capacity Block Offering: {entry['avg_instances_per_file']:.1f}")
            print(f"   Latest Data: {entry['latest_date']} ({entry['days_old']} days old)")
        
        print("\n" + "="*80)
    
    def save_report(self, report: Dict, output_file: str = None):
        """Save report to JSON file with date range in filename"""
        # Calculate date range from availability data
        if self.availability_data:
            dates = [record['file_date'] for record in self.availability_data]
            min_date = min(dates)
            max_date = max(dates)
            
            # Format dates for filename
            start_date_str = min_date.strftime('%Y%m%d')
            end_date_str = max_date.strftime('%Y%m%d')
            
            # Create filename with date range
            if output_file is None:
                output_file = f'capacity_availability_report_{start_date_str}_to_{end_date_str}.json'
        else:
            if output_file is None:
                output_file = 'capacity_availability_report.json'
        
        output_path = os.path.join(self.local_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Full report saved to: {output_path}")
        return output_path
    
    def save_csv_report(self, report: Dict, output_file: str = None):
        """Save report to CSV file with date range in filename"""
        # Calculate date range from availability data
        if self.availability_data:
            dates = [record['file_date'] for record in self.availability_data]
            min_date = min(dates)
            max_date = max(dates)
            
            # Format dates for filename
            start_date_str = min_date.strftime('%Y%m%d')
            end_date_str = max_date.strftime('%Y%m%d')
            
            # Create filename with date range
            if output_file is None:
                output_file = f'capacity_availability_report_{start_date_str}_to_{end_date_str}.csv'
        else:
            if output_file is None:
                output_file = 'capacity_availability_report.csv'
        
        output_path = os.path.join(self.local_dir, output_file)
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            # Define CSV columns
            fieldnames = [
                'rank',
                'region',
                'instance_type',
                'score',
                'available_instances',
                'total_instances',
                'total_checks',
                'available_checks',
                'availability_rate_percent',
                'best_availability_zone',
                'best_az_available_count',
                'best_az_availability_rate_percent',
                'num_availability_zones',
                'az_details',
                'avg_days_to_start',
                'avg_instances_per_capacity_block_offering',
                'latest_date',
                'days_old',
                'files_count'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write all entries
            for rank, entry in enumerate(report['all_entries'], 1):
                # Format AZ details as a string with all AZ info
                az_details_list = []
                best_az_rate = 0
                for az, az_info in entry['availability_zones'].items():
                    az_rate = round(az_info['availability_rate'] * 100, 2)
                    az_avg = round(az_info['avg_instances_per_offering'], 2)
                    az_dist = az_info.get('days_to_start_distribution', '')
                    az_details_list.append(
                        f"{az}:instances={az_info['available_count']},offerings={az_info['num_offerings']},avg={az_avg},checks={az_info['checks']},rate={az_rate}%,dist=[{az_dist}]"
                    )
                    # Track best AZ rate
                    if entry['best_availability_zone'] == az:
                        best_az_rate = az_rate
                
                row = {
                    'rank': rank,
                    'region': entry['region'],
                    'instance_type': entry['instance_type'],
                    'score': round(entry['score'], 2),
                    'available_instances': entry['available_instances'],
                    'total_instances': entry['total_instances'],
                    'total_checks': entry['total_checks'],
                    'available_checks': entry['available_checks'],
                    'availability_rate_percent': round(entry['availability_rate'] * 100, 2),
                    'best_availability_zone': entry['best_availability_zone'] or 'None',
                    'best_az_available_count': entry['best_az_available_count'],
                    'best_az_availability_rate_percent': best_az_rate,
                    'num_availability_zones': len(entry['availability_zones']),
                    'az_details': ' | '.join(az_details_list),
                    'avg_days_to_start': round(entry['avg_days_to_start'], 2) if entry['avg_days_to_start'] is not None else 'N/A',
                    'avg_instances_per_capacity_block_offering': round(entry['avg_instances_per_file'], 2),
                    'latest_date': entry['latest_date'],
                    'days_old': entry['days_old'],
                    'files_count': entry['files_count']
                }
                writer.writerow(row)
        
        print(f"📊 CSV report saved to: {output_path}")
        return output_path


def main():
    # Configuration (get from environment variables with defaults)
    bucket_name = os.getenv('S3_BUCKET_NAME', 'default')
    prefix = os.getenv('S3_PREFIX', 'default')
    
    # Create analyzer
    analyzer = CapacityAvailabilityAnalyzer(bucket_name, prefix)
    
    # Download files from S3
    files = analyzer.download_files()
    
    if not files:
        print("❌ No files to analyze")
        return
    
    # Analyze availability
    region_stats = analyzer.analyze_availability(files)
    
    # Generate report
    report = analyzer.generate_report(region_stats)
    
    # Print report
    analyzer.print_report(report)
    
    # Save JSON report
    json_report_path = analyzer.save_report(report)
    
    # Save CSV report
    csv_report_path = analyzer.save_csv_report(report)
    
    # Copy reports to project root for easy access
    import shutil
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Copy JSON
    json_filename = os.path.basename(json_report_path)
    json_destination = os.path.join(project_root, json_filename)
    shutil.copy(json_report_path, json_destination)
    print(f"📋 JSON report also copied to: {json_destination}")
    
    # Copy CSV
    csv_filename = os.path.basename(csv_report_path)
    csv_destination = os.path.join(project_root, csv_filename)
    shutil.copy(csv_report_path, csv_destination)
    print(f"📋 CSV report also copied to: {csv_destination}")
    
    print(f"\n✅ Analysis complete!")
    print(f"   📄 JSON: {json_destination}")
    print(f"   📊 CSV: {csv_destination}")


if __name__ == '__main__':
    main()

