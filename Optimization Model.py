"""
Production Planning Optimization Model
=======================================
Company: TechGadgets Inc. - Electronics Manufacturing
Problem: Maximize profit while considering resource constraints


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("PRODUCTION OPTIMIZATION MODEL")
print("TechGadgets Inc. - Electronics Manufacturing")
print("="*80)

# =============================================================================
# 1. DEFINE PROBLEM PARAMETERS
# =============================================================================

print("\n📊 PROBLEM SETUP")
print("-"*80)

# Products
products = ['Smartphones', 'Tablets', 'Laptops']

# Profit per unit ($)
profit = np.array([200, 300, 500])

# Resource requirements per unit
assembly_time = np.array([2, 3, 5])      # hours
testing_time = np.array([1, 2, 2])       # hours
raw_materials = np.array([3, 4, 6])      # units

# Available resources
assembly_capacity = 1000    # hours
testing_capacity = 500      # hours
material_capacity = 1500    # units

# Maximum demand
max_demand = np.array([300, 200, 150])

# Display problem data
problem_data = pd.DataFrame({
    'Product': products,
    'Profit ($)': profit,
    'Assembly (hrs)': assembly_time,
    'Testing (hrs)': testing_time,
    'Raw Materials': raw_materials,
    'Max Demand': max_demand
})

print(problem_data.to_string(index=False))
print("\nAvailable Resources:")
print(f"  Assembly Capacity: {assembly_capacity} hours")
print(f"  Testing Capacity: {testing_capacity} hours")
print(f"  Raw Materials: {material_capacity} units")

# =============================================================================
# 2. FORMULATE LINEAR PROGRAMMING MODEL
# =============================================================================

print("\n\n🔧 LINEAR PROGRAMMING MODEL")
print("="*80)

print("\nObjective Function (Maximize):")
print(f"  Z = {profit[0]}·x₁ + {profit[1]}·x₂ + {profit[2]}·x₃")

print("\nSubject to:")
print(f"  Assembly:     {assembly_time[0]}x₁ + {assembly_time[1]}x₂ + {assembly_time[2]}x₃ ≤ {assembly_capacity}")
print(f"  Testing:      {testing_time[0]}x₁ + {testing_time[1]}x₂ + {testing_time[2]}x₃ ≤ {testing_capacity}")
print(f"  Materials:    {raw_materials[0]}x₁ + {raw_materials[1]}x₂ + {raw_materials[2]}x₃ ≤ {material_capacity}")
print(f"  Demand (x₁):  x₁ ≤ {max_demand[0]}")
print(f"  Demand (x₂):  x₂ ≤ {max_demand[1]}")
print(f"  Demand (x₃):  x₃ ≤ {max_demand[2]}")
print(f"  Non-negative: x₁, x₂, x₃ ≥ 0")

# Set up for scipy.optimize.linprog
c = -profit  # Negate for minimization

A_ub = np.array([
    assembly_time,
    testing_time,
    raw_materials,
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

b_ub = np.array([
    assembly_capacity,
    testing_capacity,
    material_capacity,
    max_demand[0],
    max_demand[1],
    max_demand[2]
])

x_bounds = [(0, None), (0, None), (0, None)]

# =============================================================================
# 3. SOLVE THE OPTIMIZATION PROBLEM
# =============================================================================

print("\n\n⚙️  SOLVING OPTIMIZATION PROBLEM...")
print("="*80)

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='highs')

if result.success:
    optimal_production = result.x
    max_profit = -result.fun
    
    print("\n✅ OPTIMIZATION SUCCESSFUL!")
    print("\n🎯 OPTIMAL SOLUTION")
    print("-"*80)
    
    for i, product in enumerate(products):
        print(f"  {product:15s}: {optimal_production[i]:8.2f} units")
    
    print("-"*80)
    print(f"\n💰 Maximum Profit: ${max_profit:,.2f}")
    print("="*80)
    
    # Calculate resource utilization
    assembly_used = np.dot(assembly_time, optimal_production)
    testing_used = np.dot(testing_time, optimal_production)
    materials_used = np.dot(raw_materials, optimal_production)
    
    print("\n📊 RESOURCE UTILIZATION")
    print("-"*80)
    print(f"  Assembly Time:  {assembly_used:.2f} / {assembly_capacity} hours ({assembly_used/assembly_capacity*100:.1f}%)")
    print(f"  Testing Time:   {testing_used:.2f} / {testing_capacity} hours ({testing_used/testing_capacity*100:.1f}%)")
    print(f"  Raw Materials:  {materials_used:.2f} / {material_capacity} units ({materials_used/material_capacity*100:.1f}%)")
    
else:
    print("\n❌ OPTIMIZATION FAILED!")
    print(f"Reason: {result.message}")
    exit(1)

# =============================================================================
# 4. DETAILED ANALYSIS
# =============================================================================

print("\n\n📈 DETAILED ANALYSIS")
print("="*80)

# Results dataframe
profit_contribution = optimal_production * profit
results_df = pd.DataFrame({
    'Product': products,
    'Optimal Qty': optimal_production,
    'Profit/Unit ($)': profit,
    'Total Profit ($)': profit_contribution,
    'Max Demand': max_demand,
    'Demand Met (%)': (optimal_production / max_demand * 100)
})

print(results_df.to_string(index=False))

# Constraint analysis
print("\n\n🔍 CONSTRAINT ANALYSIS")
print("-"*80)

slack = b_ub - A_ub @ optimal_production
constraint_names = ['Assembly Time', 'Testing Time', 'Raw Materials', 
                   'Smartphone Demand', 'Tablet Demand', 'Laptop Demand']

binding_constraints = []
for i, name in enumerate(constraint_names):
    if abs(slack[i]) < 0.01:
        print(f"  ⚠️  {name:20s}: BINDING (fully utilized)")
        binding_constraints.append(name)
    else:
        print(f"  ✓  {name:20s}: Slack = {slack[i]:.2f}")

# =============================================================================
# 5. SENSITIVITY ANALYSIS
# =============================================================================

print("\n\n🔬 SENSITIVITY ANALYSIS")
print("="*80)
print("\nImpact of increasing each resource by 10%:\n")

resources_test = [
    ('Assembly Capacity', 0, assembly_capacity),
    ('Testing Capacity', 1, testing_capacity),
    ('Material Capacity', 2, material_capacity)
]

for resource_name, idx, original_value in resources_test:
    b_ub_modified = b_ub.copy()
    b_ub_modified[idx] = original_value * 1.1
    
    result_modified = linprog(c, A_ub=A_ub, b_ub=b_ub_modified, bounds=x_bounds, method='highs')
    
    if result_modified.success:
        new_profit = -result_modified.fun
        profit_increase = new_profit - max_profit
        pct_increase = (profit_increase / max_profit) * 100
        shadow_price = profit_increase / (original_value * 0.1)
        
        print(f"📊 {resource_name}:")
        print(f"   Original: {original_value} → New: {original_value * 1.1:.0f}")
        print(f"   Profit Impact: ${profit_increase:,.2f} (+{pct_increase:.2f}%)")
        print(f"   Shadow Price: ${shadow_price:.2f} per unit\n")

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================

print("\n📊 GENERATING VISUALIZATIONS...")

fig = plt.figure(figsize=(16, 12))

# 1. Optimal Production
ax1 = plt.subplot(2, 3, 1)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars1 = ax1.bar(products, optimal_production, color=colors, edgecolor='black', linewidth=2)
ax1.set_title('Optimal Production Quantities', fontsize=14, fontweight='bold')
ax1.set_ylabel('Units', fontsize=12)
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# 2. Profit Contribution
ax2 = plt.subplot(2, 3, 2)
bars2 = ax2.bar(products, profit_contribution, color=colors, edgecolor='black', linewidth=2)
ax2.set_title('Profit Contribution', fontsize=14, fontweight='bold')
ax2.set_ylabel('Profit ($)', fontsize=12)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'${int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3. Resource Utilization
ax3 = plt.subplot(2, 3, 3)
resources = ['Assembly', 'Testing', 'Materials']
utilization = [assembly_used/assembly_capacity*100, 
               testing_used/testing_capacity*100,
               materials_used/material_capacity*100]
bars3 = ax3.bar(resources, utilization, color=colors, edgecolor='black', linewidth=2)
ax3.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Full Capacity')
ax3.set_title('Resource Utilization', fontsize=14, fontweight='bold')
ax3.set_ylabel('Utilization (%)', fontsize=12)
ax3.legend()
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# 4. Production vs Demand
ax4 = plt.subplot(2, 3, 4)
x_pos = np.arange(len(products))
width = 0.35
ax4.bar(x_pos - width/2, optimal_production, width, label='Production', color='#4ECDC4')
ax4.bar(x_pos + width/2, max_demand, width, label='Max Demand', color='#95E1D3', alpha=0.7)
ax4.set_title('Production vs Demand', fontsize=14, fontweight='bold')
ax4.set_ylabel('Units', fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(products)
ax4.legend()

# 5. Profit Distribution
ax5 = plt.subplot(2, 3, 5)
ax5.pie(profit_contribution, labels=products, autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontweight': 'bold'})
ax5.set_title('Profit Distribution', fontsize=14, fontweight='bold')

# 6. Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary = f"""
╔════════════════════════════════════╗
║      OPTIMIZATION SUMMARY          ║
╠════════════════════════════════════╣
║                                    ║
║  Max Profit: ${max_profit:,.0f}         ║
║                                    ║
║  Total Units: {optimal_production.sum():.0f}              ║
║                                    ║
║  Assembly: {assembly_used/assembly_capacity*100:.1f}%              ║
║  Testing: {testing_used/testing_capacity*100:.1f}%               ║
║  Materials: {materials_used/material_capacity*100:.1f}%             ║
║                                    ║
║  Status: ✅ OPTIMAL                ║
╚════════════════════════════════════╝
"""
ax6.text(0.5, 0.5, summary, ha='center', va='center', 
         fontsize=11, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('Production Optimization - TechGadgets Inc.', 
             fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved: optimization_results.png")

# =============================================================================
# 7. KEY INSIGHTS
# =============================================================================

print("\n\n💡 KEY BUSINESS INSIGHTS")
print("="*80)

insights = [
    f"1. {products[np.argmax(profit_contribution)]} generates the highest profit contribution",
    f"2. Bottleneck resources: {', '.join(binding_constraints) if binding_constraints else 'None'}",
    f"3. Total production: {optimal_production.sum():.0f} units",
    f"4. Average profit per unit: ${max_profit/optimal_production.sum():.2f}",
    f"5. Most efficient product (profit/resources): {products[np.argmax(profit/(assembly_time+testing_time+raw_materials))]}"
]

for insight in insights:
    print(f"   {insight}")

print("\n\n📋 RECOMMENDATIONS")
print("="*80)

recommendations = [
    "1. ⭐ Implement the optimal production mix calculated above",
    f"2. 🏭 Focus capacity expansion on: {binding_constraints[0] if binding_constraints else 'all resources equally'}",
    f"3. 📈 Expected profit: ${max_profit:,.2f} with current resources",
    "4. 🔄 Review and update optimization monthly with actual data",
    "5. 💰 Investigate opportunities to reduce production costs",
    "6. 📊 Monitor KPIs: profit margin, resource utilization, production efficiency"
]

for rec in recommendations:
    print(f"   {rec}")

# =============================================================================
# 8. EXPORT RESULTS
# =============================================================================

print("\n\n💾 EXPORTING RESULTS...")

# Export to CSV
results_df.to_csv('optimization_results.csv', index=False)
print("✅ Saved: optimization_results.csv")

# Export summary
with open('optimization_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("PRODUCTION OPTIMIZATION SUMMARY\n")
    f.write("TechGadgets Inc. - Electronics Manufacturing\n")
    f.write("="*80 + "\n\n")
    
    f.write("OPTIMAL PRODUCTION:\n")
    for i, product in enumerate(products):
        f.write(f"  {product:15s}: {optimal_production[i]:8.2f} units\n")
    f.write(f"\nMAXIMUM PROFIT: ${max_profit:,.2f}\n")
    f.write("\n" + "="*80 + "\n")

print("✅ Saved: optimization_summary.txt")

print("\n\n🎉 OPTIMIZATION COMPLETE!")
print("="*80)
print(f"\nMaximum Profit: ${max_profit:,.2f}")
print(f"Solution Status: OPTIMAL ✅")
print("\nFiles generated:")
print("  - optimization_results.png")
print("  - optimization_results.csv")
print("  - optimization_summary.txt")
print("\n" + "="*80)
