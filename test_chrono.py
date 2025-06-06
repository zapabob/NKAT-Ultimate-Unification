#!/usr/bin/env python3
"""
Simple test script for Chrono.py visualization
"""

import matplotlib.pyplot as plt
import matplotlib
print(f"Matplotlib backend: {matplotlib.get_backend()}")

# Try to set a suitable backend for Windows
try:
    matplotlib.use('Agg')  # Non-interactive backend
    print("Using Agg backend for safe visualization")
except:
    print("Using default backend")

from Chrono import QuantumSpacetimeCell

def test_visualization():
    """Test the visualization without showing plots"""
    print("🧮 Testing 2-bit Quantum Cell Calculations")
    
    # Create instance
    qst = QuantumSpacetimeCell()
    
    # Run calculations
    results = qst.calculate_2bit_cell_length()
    optimal_length, factor = qst.calculate_optimal_cell_length()
    properties = qst.calculate_cell_volume_and_properties()
    
    # Create visualization and save to file instead of showing
    plt.figure(figsize=(12, 8))
    
    methods = list(results.keys())
    lengths = [results[method] / qst.l_planck for method in methods]
    
    # Bar chart
    plt.subplot(2, 1, 1)
    bars = plt.bar(range(len(methods)), lengths, alpha=0.7, color='skyblue')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.ylabel('Length (Planck length units)')
    plt.title('2-bit Quantum Cell Length by Various Theories')
    plt.axhline(y=43.06, color='red', linestyle='--', 
               label='NKAT theory value (43.06 lₚ)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log plot
    plt.subplot(2, 1, 2)
    x_pos = range(len(methods))
    plt.semilogy(x_pos, lengths, 'bo-', markersize=8)
    plt.xticks(x_pos, methods, rotation=45, ha='right')
    plt.ylabel('Length (Planck length units, log)')
    plt.title('Comparison in Logarithmic Scale')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save instead of show
    plt.savefig('2bit_quantum_cell_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Graph saved as '2bit_quantum_cell_analysis.png'")
    
    plt.close()  # Close to free memory
    
    print("✅ Test completed successfully!")
    print(f"📏 Optimal cell length: {optimal_length:.3e} m")
    print(f"🔢 Factor: {factor:.2f} lₚ")
    
    return True

if __name__ == "__main__":
    test_visualization() 