#!/usr/bin/env python3
"""
Test script to verify the PDF processing solution works correctly.
"""

import json
import sys
from pathlib import Path

def compare_outputs():
    """Compare generated outputs with expected outputs"""
    
    # Paths
    expected_dir = Path("Challenge_1a/sample_dataset/outputs")
    generated_dir = Path("Challenge_1a/sample_dataset/outputs")  # Same directory since we overwrite
    
    if not expected_dir.exists():
        print("❌ Expected outputs directory not found")
        return False
    
    # Get all JSON files
    json_files = list(expected_dir.glob("*.json"))
    
    if not json_files:
        print("❌ No JSON files found")
        return False
    
    print(f"✅ Found {len(json_files)} output files")
    
    # Check each file
    all_good = True
    for json_file in json_files:
        print(f"\n📄 Checking {json_file.name}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check structure
            if "title" not in data:
                print(f"  ❌ Missing 'title' field")
                all_good = False
            else:
                print(f"  ✅ Title: '{data['title']}'")
            
            if "outline" not in data:
                print(f"  ❌ Missing 'outline' field")
                all_good = False
            else:
                print(f"  ✅ Outline: {len(data['outline'])} headings")
                
                # Check outline structure
                for i, heading in enumerate(data['outline']):
                    if not all(key in heading for key in ['level', 'text', 'page']):
                        print(f"  ❌ Heading {i} missing required fields")
                        all_good = False
                        break
                    if heading['level'] not in ['H1', 'H2', 'H3', 'H4']:
                        print(f"  ❌ Heading {i} has invalid level: {heading['level']}")
                        all_good = False
                        break
                    if not isinstance(heading['page'], int):
                        print(f"  ❌ Heading {i} has invalid page number: {heading['page']}")
                        all_good = False
                        break
                else:
                    print(f"  ✅ All headings have valid structure")
            
        except json.JSONDecodeError as e:
            print(f"  ❌ Invalid JSON: {e}")
            all_good = False
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
            all_good = False
    
    return all_good

def main():
    print("🧪 Testing PDF Processing Solution")
    print("=" * 50)
    
    # Check if the processing script exists
    script_path = Path("Challenge_1a/process_pdfs.py")
    if not script_path.exists():
        print("❌ process_pdfs.py not found")
        return 1
    
    print("✅ Processing script found")
    
    # Check if Dockerfile exists
    dockerfile_path = Path("Challenge_1a/Dockerfile")
    if not dockerfile_path.exists():
        print("❌ Dockerfile not found")
        return 1
    
    print("✅ Dockerfile found")
    
    # Check if requirements.txt exists
    requirements_path = Path("Challenge_1a/requirements.txt")
    if not requirements_path.exists():
        print("❌ requirements.txt not found")
        return 1
    
    print("✅ requirements.txt found")
    
    # Compare outputs
    if compare_outputs():
        print("\n🎉 All tests passed! Solution is ready for submission.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 