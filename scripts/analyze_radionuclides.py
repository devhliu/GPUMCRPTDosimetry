#!/usr/bin/env python3
"""
Analyze radionuclide emission data from ICRP 107 database
Find maximum energy and corresponding emission type for each radionuclide
"""

import json
import os
import glob
from collections import defaultdict

# Medical purpose radionuclide classification
MEDICAL_RADIONUCLIDES = {
    # Imaging radionuclides
    'imaging': {
        'F-18', 'Tc-99m', 'I-123', 'I-131', 'Ga-67', 'Ga-68', 'In-111', 'Tl-201', 
        'Xe-133', 'Kr-81m', 'Rb-82', 'C-11', 'N-13', 'O-15', 'Cu-64', 'Zr-89',
        'I-124', 'Y-86', 'Br-76'
    },
    # Therapy radionuclides
    'therapy': {
        'I-131', 'Y-90', 'Lu-177', 'Ra-223', 'Sm-153', 'Sr-89', 'P-32', 'Re-186',
        'Re-188', 'Ho-166', 'Cu-67', 'At-211', 'Bi-213', 'Ac-225', 'Tb-149',
        'Pb-212', 'Fr-221', 'Rn-219', 'Tb-161'
    },
    # Common parent radionuclides that produce medical daughters
    'parent': {
        'Ac-225',
        'Ra-223', 'Th-227'
    }
}

# Decay chains for medical radionuclides
MEDICAL_DECAY_CHAINS = {
    # Ac-225 decay chain (alpha therapy)
    'Ac-225': {'Fr-221', 'At-217', 'Bi-213', 'Po-213', 'Pb-209', 'Tl-209'},
    # Ra-223 decay chain (alpha therapy - Xofigo)
    'Ra-223': {'Rn-219', 'Po-215', 'Pb-211', 'Bi-211', 'Tl-207', 'Po-211'},
    # Th-227 decay chain
    'Th-227': {'Ra-223', 'Rn-219', 'Po-215', 'Pb-211', 'Bi-211', 'Tl-207'}
}

def classify_medical_purpose(radionuclide_name):
    """Classify radionuclide based on medical purpose"""
    
    # Check if it's a direct medical radionuclide for imaging
    if radionuclide_name in MEDICAL_RADIONUCLIDES['imaging']:
        return 'Medical - Imaging'
    
    # Check if it's a direct medical radionuclide for therapy
    elif radionuclide_name in MEDICAL_RADIONUCLIDES['therapy']:
        return 'Medical - Therapy'
    
    # Check if it's a parent radionuclide that produces medical daughters
    elif radionuclide_name in MEDICAL_RADIONUCLIDES['parent']:
        return 'Medical - Parent (Generator)'
    
    # Check if it's a daughter in medical decay chains
    for parent, daughters in MEDICAL_DECAY_CHAINS.items():
        if radionuclide_name in daughters:
            # Check if the parent is a medical radionuclide
            if (parent in MEDICAL_RADIONUCLIDES['imaging'] or 
                parent in MEDICAL_RADIONUCLIDES['therapy'] or 
                parent in MEDICAL_RADIONUCLIDES['parent']):
                return f'Medical - Daughter of {parent}'
    
    return 'Non-Medical'

def analyze_radionuclide_emissions():
    """Analyze all radionuclide JSON files and find max energy per radionuclide"""
    
    # Path to the radionuclide data files
    data_dir = "/mnt/d/WSL/workspace/devhliu/Dosimetry/MCGPURPTDosimetry/GPUMCRPTDosimetry/src/gpumcrpt/decaydb/icrp107_database/icrp107"
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    print(f"Found {len(json_files)} radionuclide files")
    
    # Dictionary to store results
    results = []
    
    # Process each radionuclide file
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                # Files contain JSON as a string within quotes
                content = f.read().strip()
                
                # Remove surrounding quotes if present
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                
                # Handle escaped quotes within the JSON
                content = content.replace('\\"', '"')
                
                # Parse the JSON data
                data = json.loads(content)
            
            radionuclide_name = data.get('name', os.path.basename(file_path).replace('.json', ''))
            emissions = data.get('emissions', {})
            
            max_energy = 0.0
            max_energy_type = ""
            
            # Check each emission type
            for emission_type, energy_data in emissions.items():
                if isinstance(energy_data, list) and energy_data:
                    # Find maximum energy for this emission type
                    for energy_entry in energy_data:
                        if isinstance(energy_entry, list) and len(energy_entry) >= 1:
                            energy = energy_entry[0]
                            if isinstance(energy, (int, float)) and energy > max_energy:
                                max_energy = energy
                                max_energy_type = emission_type
            
            if max_energy > 0:
                medical_purpose = classify_medical_purpose(radionuclide_name)
                results.append({
                    'radionuclide': radionuclide_name,
                    'max_energy': max_energy,
                    'emission_type': max_energy_type,
                    'half_life': data.get('half_life', 0.0),
                    'time_unit': data.get('time_unit', ''),
                    'medical_purpose': medical_purpose
                })
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Sort results by maximum energy (descending)
    results.sort(key=lambda x: x['max_energy'], reverse=True)
    
    return results

def generate_table(results):
    """Generate a formatted table of results"""
    
    print("\n" + "="*130)
    print("Radionuclide Maximum Energy Analysis")
    print("="*130)
    print(f"{'Radionuclide':<15} {'Max Energy (MeV)':<15} {'Emission Type':<15} {'Half-Life':<15} {'Unit':<10} {'Medical Purpose':<30}")
    print("-"*130)
    
    for result in results:
        half_life_str = f"{result['half_life']:.6f}" if result['half_life'] > 0 else "N/A"
        time_unit = result['time_unit'] if result['time_unit'] else "N/A"
        medical_purpose = result['medical_purpose']
        print(f"{result['radionuclide']:<15} {result['max_energy']:<15.6f} {result['emission_type']:<15} {half_life_str:<15} {time_unit:<10} {medical_purpose:<30}")
    
    print("-"*130)
    print(f"Total radionuclides analyzed: {len(results)}")
    print("="*130)

def main():
    """Main function to analyze and display results"""
    
    print("Starting radionuclide emission analysis...")
    
    results = analyze_radionuclide_emissions()
    
    if results:
        generate_table(results)
        
        # Save results to CSV
        csv_filename = "radionuclide_max_energies.csv"
        with open(csv_filename, 'w') as f:
            f.write("Radionuclide,Max_Energy_MeV,Emission_Type,Half_Life,Time_Unit,Medical_Purpose\n")
            for result in results:
                f.write(f"{result['radionuclide']},{result['max_energy']},{result['emission_type']},{result['half_life']},{result['time_unit']},{result['medical_purpose']}\n")
        
        print(f"\nResults saved to: {csv_filename}")
        
        # Show some statistics
        print("\nStatistics:")
        print(f"Highest energy: {results[0]['max_energy']:.6f} MeV ({results[0]['radionuclide']} - {results[0]['emission_type']})")
        print(f"Lowest energy (non-zero): {results[-1]['max_energy']:.6f} MeV ({results[-1]['radionuclide']} - {results[-1]['emission_type']})")
        
        # Count emission types
        emission_counts = defaultdict(int)
        for result in results:
            emission_counts[result['emission_type']] += 1
            
        print("\nEmission type distribution:")
        for emission_type, count in sorted(emission_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emission_type}: {count} radionuclides")
    else:
        print("No valid radionuclide data found.")

if __name__ == "__main__":
    main()